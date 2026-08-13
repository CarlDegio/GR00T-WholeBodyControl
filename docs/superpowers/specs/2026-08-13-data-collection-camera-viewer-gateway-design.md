# Data Collection Camera Viewer Gateway Design

## Goal

Make the data-collection camera viewer consume decoded RGB frames exclusively
from `SensorGateway`. The viewer must no longer connect directly to the camera
server on port 5555.

## Scope

Modify `gear_sonic/scripts/run_camera_viewer.py` and the viewer command launched
by `gear_sonic/scripts/launch_data_collection.py`.

Preserve the current viewer behavior:

- display every available RGB camera in the existing horizontal tiled layout;
- ignore depth streams;
- convert RGB arrays to BGR for OpenCV;
- keep `R` start/stop MP4 recording and `Q` quit controls;
- keep the current recording directory, codec, FPS, labels, and status overlay.

Do not retain a direct-camera compatibility mode. Do not modify
`run_operator_cv_viewer.py`, camera-server startup, endpoint defaults, dataset
depth recording, readiness gating, or camera installation workflows.

## Data Flow

```text
camera server :5555
        |
        v
SensorGateway camera/* shared-memory streams + RPC :5560
        |
        v
run_camera_viewer.py -> OpenCV display / optional MP4 recording
```

The viewer loads the same runtime profile as the launcher and resolves the
`sensor_gateway_metadata` endpoint from it. The launcher passes its
`runtime_profile` value to the viewer instead of passing `camera_host` and
`camera_port`.

## Gateway Reader

The viewer owns a `SensorGatewayClient`. It discovers available decoded RGB
streams from the Gateway health response by selecting names under `camera/`
and excluding names ending in `_depth`. The names are sorted to keep the
existing deterministic tile and output-file order.

After discovery, the viewer requests a snapshot containing those streams and
returns an `images` mapping keyed by the camera name without the `camera/`
prefix. Only `HxWx3 uint8` arrays are accepted as RGB frames.

The first-frame wait retains the current ten-second behavior. Gateway timeout,
missing-stream, stale-frame, and overwritten-frame errors are treated as a
temporarily unavailable frame while waiting or displaying. They do not crash
the viewer. Cleanup always closes the Gateway client, releases active video
writers, and destroys OpenCV windows.

## CLI

Remove `camera_host` and `camera_port` from `CameraViewerConfig`. Add a runtime
profile option with the same default profile used elsewhere in the repository.
No direct-camera CLI is retained.

## Testing

Add focused tests that prove:

1. Gateway health discovery selects sorted `camera/*` RGB streams and ignores
   depth and non-camera streams.
2. Snapshot materialization returns camera-name keyed RGB arrays and rejects
   non-RGB payloads.
3. Temporary Gateway errors return no new frame instead of escaping the viewer
   loop.
4. The data-collection launcher builds a viewer command containing the runtime
   profile and no `camera-host`, `camera-port`, or direct camera client path.
5. Existing RGB filtering, viewer layout/recording behavior, SensorGateway, and
   data-collection tests remain green.

## Success Criteria

When `launch_data_collection.py` starts its camera viewer, the only camera
network subscriber is `SensorGateway`. The viewer obtains decoded RGB arrays
through the local Gateway Snapshot API, displays the same camera tiles, and
retains independent MP4 recording controls.
