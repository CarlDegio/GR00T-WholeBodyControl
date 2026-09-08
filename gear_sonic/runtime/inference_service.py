"""Shared startup resources for inference services."""

import queue

from gear_sonic.runtime.profile import load_runtime_profile
from gear_sonic.experiments.recording import Recorder
from gear_sonic.runtime.telemetry import (
    build_event,
    configure_file_logging,
    emit_event,
    open_telemetry_publisher,
    publish_metrics,
)


class InferenceServiceContext:
    """Own profile, logging, telemetry publishers, and queued runtime events."""

    def __init__(self, component: str, config: object, *, enable_metrics=True):
        self.component = str(component)
        self.logger = configure_file_logging(self.component)
        self.profile = load_runtime_profile(
            getattr(config, "profile", "") or None,
            overlays=tuple(getattr(config, "overlay", ())),
        )
        self.pending_events = queue.SimpleQueue()
        self.experiment = Recorder(self.profile)
        endpoint = self.profile.endpoint_uri
        self._event_socket = open_telemetry_publisher(endpoint("runtime_event_ingress"))
        self._metrics_socket = (
            open_telemetry_publisher(endpoint("runtime_metrics_ingress"))
            if enable_metrics
            else None
        )
        self._closed = False

    def event(
        self, level, code, message, *, queued=False, write_log=True, **fields
    ):
        self.experiment.runtime(self.component, code, **fields)
        payload = build_event(self.component, level, code, message, **fields)
        if write_log:
            emit_event(payload, logger=self.logger)
        if queued:
            self.pending_events.put(payload)
        else:
            emit_event(payload, socket=self._event_socket)

    def flush_events(self):
        while True:
            try:
                payload = self.pending_events.get_nowait()
            except queue.Empty:
                return
            emit_event(payload, socket=self._event_socket)

    def publish_metrics(self, values, *, allowed_names, activate=False):
        if self._metrics_socket is None:
            raise RuntimeError(f"metrics are disabled for {self.component}")
        return publish_metrics(
            self._metrics_socket, self.component, values,
            allowed_names=allowed_names, activate=activate,
        )

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.flush_events()
        self._event_socket.close(linger=0)
        if self._metrics_socket is not None:
            self._metrics_socket.close(linger=0)
