"""Strict YAML runtime profiles for the SONIC process graph."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import yaml

from gear_sonic.runtime.endpoints import ENDPOINTS, Transport


PROFILE_SCHEMA = "sonic.runtime_profile"
PROFILE_VERSION = 1
_TOP_LEVEL_KEYS = {
    "schema",
    "version",
    "profile",
    "endpoints",
    "ros_topics",
    "components",
    "launch_inference",
}


@dataclass(frozen=True)
class EndpointAddress:
    """Connect address selected for one endpoint in a runtime profile."""

    host: str
    port: int

    def __post_init__(self) -> None:
        if not self.host:
            raise ValueError("endpoint host cannot be empty")
        if not 1 <= self.port <= 65535:
            raise ValueError(f"invalid endpoint port: {self.port}")


@dataclass(frozen=True)
class RuntimeProfile:
    """Validated configuration snapshot shared by staged runtime migrations."""

    name: str
    endpoints: Mapping[str, EndpointAddress]
    ros_topics: Mapping[str, str]
    components: Mapping[str, Mapping[str, Any]]
    source_files: tuple[Path, ...]

    def endpoint(self, name: str) -> EndpointAddress:
        try:
            return self.endpoints[name]
        except KeyError as exc:
            raise KeyError(f"runtime profile {self.name!r} has no endpoint {name!r}") from exc

    def endpoint_uri(self, name: str) -> str:
        address = self.endpoint(name)
        scheme = "tcp" if ENDPOINTS[name].transport is Transport.ZMQ else "http"
        return f"{scheme}://{address.host}:{address.port}"

    def component(self, name: str) -> Mapping[str, Any]:
        try:
            return self.components[name]
        except KeyError as exc:
            raise KeyError(f"runtime profile {self.name!r} has no component {name!r}") from exc


def default_runtime_profile_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "launch_inference.yaml"


def _load_yaml_object(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML runtime profile {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"runtime profile must contain a YAML object: {path}")
    return payload


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _validate_top_level(payload: Mapping[str, Any]) -> None:
    unknown = set(payload) - _TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(f"unknown runtime profile fields: {', '.join(sorted(unknown))}")
    if payload.get("schema") != PROFILE_SCHEMA:
        raise ValueError(f"runtime profile schema must be {PROFILE_SCHEMA!r}")
    if int(payload.get("version", -1)) != PROFILE_VERSION:
        raise ValueError(f"runtime profile version must be {PROFILE_VERSION}")
    if not str(payload.get("profile", "")):
        raise ValueError("runtime profile name cannot be empty")


def _parse_endpoints(payload: Any) -> Mapping[str, EndpointAddress]:
    if not isinstance(payload, Mapping):
        raise ValueError("runtime profile endpoints must be an object")
    result: dict[str, EndpointAddress] = {}
    used_addresses: dict[tuple[str, int], str] = {}
    for name, raw_address in payload.items():
        if name not in ENDPOINTS:
            raise ValueError(f"runtime profile contains unknown endpoint {name!r}")
        if not isinstance(raw_address, Mapping):
            raise ValueError(f"endpoint {name!r} must be an object")
        unknown = set(raw_address) - {"host", "port"}
        if unknown:
            raise ValueError(
                f"endpoint {name!r} has unknown fields: {', '.join(sorted(unknown))}"
            )
        address = EndpointAddress(
            host=str(raw_address.get("host", "")),
            port=int(raw_address.get("port", -1)),
        )
        address_key = (address.host, address.port)
        previous = used_addresses.setdefault(address_key, name)
        if previous != name:
            raise ValueError(
                f"runtime profile address {address.host}:{address.port} is shared by "
                f"{previous!r} and {name!r}"
            )
        result[name] = address
    return MappingProxyType(result)


def _parse_ros_topics(payload: Any) -> Mapping[str, str]:
    if not isinstance(payload, Mapping):
        raise ValueError("runtime profile ros_topics must be an object")
    result: dict[str, str] = {}
    for role, raw_name in payload.items():
        name = str(raw_name)
        if not name.startswith("/"):
            raise ValueError(f"runtime profile ROS topic must be absolute: {name!r}")
        result[str(role)] = name
    return MappingProxyType(result)


def _parse_components(payload: Any) -> Mapping[str, Mapping[str, Any]]:
    if not isinstance(payload, Mapping):
        raise ValueError("runtime profile components must be an object")
    components: dict[str, Mapping[str, Any]] = {}
    for name, parameters in payload.items():
        if not str(name):
            raise ValueError("runtime component name cannot be empty")
        if not isinstance(parameters, Mapping):
            raise ValueError(f"runtime component {name!r} must be an object")
        components[str(name)] = MappingProxyType(copy.deepcopy(dict(parameters)))
    return MappingProxyType(components)


def load_runtime_profile(
    path: str | Path | None = None,
    *,
    overlays: Sequence[str | Path] = (),
) -> RuntimeProfile:
    """Load a base YAML profile and optional partial YAML overlays.

    Precedence is base profile, then overlays from left to right. CLI overrides
    remain the final layer for processes as they migrate onto the profile.
    """

    base_path = default_runtime_profile_path() if path is None else Path(path).expanduser()
    source_files = [base_path.resolve()]
    payload = _load_yaml_object(base_path)
    for overlay in overlays:
        overlay_path = Path(overlay).expanduser()
        payload = _deep_merge(payload, _load_yaml_object(overlay_path))
        source_files.append(overlay_path.resolve())

    _validate_top_level(payload)
    return RuntimeProfile(
        name=str(payload["profile"]),
        endpoints=_parse_endpoints(payload.get("endpoints")),
        ros_topics=_parse_ros_topics(payload.get("ros_topics")),
        components=_parse_components(payload.get("components")),
        source_files=tuple(source_files),
    )
