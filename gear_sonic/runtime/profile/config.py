"""Strict YAML runtime profiles for the SONIC process graph."""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import (
    Any,
    Literal,
    Mapping,
    Sequence,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

import yaml

from gear_sonic.runtime.profile.endpoints import ENDPOINT_SCHEMES

PROFILE_SCHEMA = "sonic.runtime_profile"
PROFILE_VERSION = 1
ComponentConfig = TypeVar("ComponentConfig")
_TOP_LEVEL_KEYS = {
    "schema",
    "version",
    "profile",
    "endpoints",
    "ros_topics",
    "components",
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
        return f"{ENDPOINT_SCHEMES[name]}://{address.host}:{address.port}"

    def component(self, name: str) -> Mapping[str, Any]:
        try:
            return self.components[name]
        except KeyError as exc:
            raise KeyError(f"runtime profile {self.name!r} has no component {name!r}") from exc


@dataclass(frozen=True)
class RuntimeProfileSelection:
    """The only process-level CLI selection shared by production workers."""

    profile: str = ""
    """Base runtime YAML profile; empty selects the repository default."""

    overlay: tuple[str, ...] = ()
    """Optional partial profiles applied from left to right."""


def default_runtime_profile_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "launch_inference.yaml"


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
        if name not in ENDPOINT_SCHEMES:
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

    Precedence is the base profile followed by overlays from left to right.
    Production process endpoints are resolved from this merged snapshot.
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


def load_component_config(
    config_type: type[ComponentConfig],
    component: str,
    path: str | Path | None = None,
    *,
    overlays: Sequence[str | Path] = (),
    ignored_fields: Sequence[str] = (),
) -> ComponentConfig:
    """Build one worker config strictly from its runtime-profile component.

    Worker dataclasses retain their explicit fields for typed internal use and
    focused tests. Production CLIs expose only ``profile`` and ``overlay``;
    every other value must be present exactly once under ``components``.
    """

    profile = load_runtime_profile(path, overlays=overlays)
    values = dict(profile.component(component))
    for name in ignored_fields:
        values.pop(name)
    config_fields = {field.name for field in fields(config_type)}
    selection_fields = {"profile", "overlay", "config"}
    expected = config_fields - selection_fields
    unknown = set(values) - expected
    missing = expected - set(values)
    if unknown:
        raise ValueError(
            f"runtime component {component!r} has unknown fields: "
            + ", ".join(sorted(unknown))
        )
    if missing:
        raise ValueError(
            f"runtime component {component!r} is missing fields: "
            + ", ".join(sorted(missing))
        )
    annotations = get_type_hints(config_type)
    for name, value in tuple(values.items()):
        annotation = annotations[name]
        origin = get_origin(annotation)
        if origin is Literal:
            allowed = get_args(annotation)
            if value not in allowed:
                raise ValueError(
                    f"runtime component {component!r}.{name} must be one of "
                    f"{allowed}, got {value!r}"
                )
        elif annotation is bool:
            if not isinstance(value, bool):
                raise ValueError(
                    f"runtime component {component!r}.{name} must be a boolean"
                )
        elif annotation is int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"runtime component {component!r}.{name} must be an integer"
                )
        elif annotation is float:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"runtime component {component!r}.{name} must be a number"
                )
            values[name] = float(value)
        elif annotation is str and not isinstance(value, str):
            raise ValueError(
                f"runtime component {component!r}.{name} must be a string"
            )
    selection_values: dict[str, Any] = {}
    if "profile" in config_fields:
        selection_values["profile"] = str(profile.source_files[0])
    if "overlay" in config_fields:
        selection_values["overlay"] = tuple(
            str(source) for source in profile.source_files[1:]
        )
    if "config" in config_fields:
        selection_values["config"] = str(profile.source_files[0])
    return config_type(**selection_values, **values)


def parse_component_config(
    config_type: type[ComponentConfig],
    component: str,
    args: list[str] | None = None,
    *,
    ignored_fields: Sequence[str] = (),
) -> ComponentConfig:
    """Parse the shared profile CLI and build one component config."""
    import tyro

    selection = tyro.cli(RuntimeProfileSelection, args=args)
    return load_component_config(
        config_type,
        component,
        selection.profile or None,
        overlays=selection.overlay,
        ignored_fields=ignored_fields,
    )
