"""A single request lane for resident VLA backends, with thread-owned ZMQ clients."""

from dataclasses import dataclass
import threading


@dataclass(frozen=True)
class Backend:
    host: str
    port: int
    prompt: str
    timeout_ms: int = 15000

    def __post_init__(self):
        if (not isinstance(self.host, str) or not self.host.strip()
            or not isinstance(self.prompt, str) or not self.prompt.strip()
            or type(self.port) is not int or not 0 < self.port < 65536
            or type(self.timeout_ms) is not int or self.timeout_ms <= 0):
            raise ValueError("VLA backend requires host, port, prompt and a positive timeout")


def parse_backends(raw):
    if not isinstance(raw, dict):
        raise ValueError("vla.backends must be a mapping")
    if any(not isinstance(key, str) or not key.strip() for key in raw):
        raise ValueError("VLA skill keys must be nonempty strings")
    return {key: Backend(**value) for key, value in raw.items()}


class PolicyRouter:
    def __init__(self, backends, default_skill, client_factory, *, require_warmup=False, disabled=False):
        self.backends = dict(backends)
        if default_skill not in backends:
            raise ValueError(f"unknown default VLA skill: {default_skill}")
        self.default_skill = self.selected_skill = default_skill
        self.client_factory = client_factory
        self.require_warmup = require_warmup
        self.disabled = disabled
        self.warmed = set()
        self.required_warmups = (set(backends) - {""}) | {default_skill}
        self._clients = {}
        self._requests = threading.Lock()
        self._selection = threading.Lock()

    def select(self, skill_key):
        if self.disabled:
            raise ValueError("VLA backends disabled for smoke_assume_vla_success")
        key = skill_key or self.default_skill
        if key not in self.backends:
            raise ValueError(f"unknown VLA skill_key: {key}")
        if not self.ready() or (self.require_warmup and key not in self.warmed):
            raise ValueError(f"VLA backend is not prewarmed: {key}")
        with self._selection:
            self.selected_skill = key
        return key

    def ready(self):
        return not self.disabled and (not self.require_warmup or self.required_warmups <= self.warmed)

    def _client(self, key):
        if self.disabled:
            raise ValueError("VLA backends disabled for smoke_assume_vla_success")
        identity = (threading.get_ident(), key)
        if identity not in self._clients:
            backend = self.backends[key]
            self._clients[identity] = self.client_factory(backend.host, backend.port, timeout_ms=backend.timeout_ms)
        return self._clients[identity]

    def snapshot(self):
        with self._selection:
            return self.selected_skill

    def ping(self, *, timeout_ms=1000):
        if self.disabled:
            return False
        key = self.snapshot()
        if not self._requests.acquire(timeout=timeout_ms / 1000):
            return False
        try:
            return self._client(key).ping(timeout_ms=timeout_ms)
        finally:
            self._requests.release()

    def infer(self, key, observation, *, current, run):
        # Check the epoch *after* waiting for the request lane. No old observation
        # can migrate to the newly selected model or start after cancellation.
        with self._requests:
            if not current():
                return None
            return run(self._client(key), observation)

    def warmup(self, observation, *, stopped, keys=None):
        if self.disabled:
            return
        for key in self.backends if keys is None else keys:
            backend = self.backends[key]
            if stopped():
                return
            obs = dict(observation)
            # Same observation contract as prepare_observation_from_sensors.
            obs["language"] = {**obs.get("language", {}), "annotation.human.task_description": [[backend.prompt]]}
            with self._requests:
                self._client(key).get_action(obs)
            self.warmed.add(key)

    def close(self):
        for client in self._clients.values():
            client.close()
