# Recenter Threshold Recalibration Design

## Goal

Prevent the horizontal and vertical visibility guards from activating on the
healthy first frames observed in the latest g1 and g3 runs while retaining a
measured margin before historical YOLOE target loss.

## Evidence

- Latest g1 began with target bbox `[292, 0, 560, 115.875]`. Its bottom edge
  was still at 115.875 px, so substantial target content remained visible.
- Latest g3 began with target bbox `[267.75, 82, 621, 317]`. Its center was at
  69.4% image width even though its right edge was at 97.0% because the basket
  occupied more than half the image width.
- Historical top-only terminal losses had last-valid bbox bottom edges near
  46--52 px.
- Historical right-side terminal loss had a last-valid bbox center near 89.2%
  image width; transient right-edge misses occurred farther right.

The existing horizontal outer-edge test is therefore size-sensitive, and the
existing vertical center test is crop/height-sensitive.

## Approved behavior

- Horizontal entry uses bbox center x, not bbox outer edges.
- Enter horizontal `RECENTER` when center x is below 20% or above 80% of
  image width.
- Recover horizontally when center x remains inside 25%--75% of image width
  for three consecutive visual frames.
- Vertical entry uses the bbox bottom edge `y2`, not bbox center y.
- Enter `VERTICAL_RECENTER` when `y2 < 75 px`.
- Recover vertically when `y2 >= 110 px` for three consecutive visual frames.
- Transition-frame zero commands, forward-only vertical motion, lateral-only
  horizontal motion, vertical priority, saved resume phases, and all existing
  safety behavior remain unchanged.

At 640x480, the horizontal entry leaves at least 59 px relative to the
earliest observed right-only terminal loss. The 75 px vertical entry leaves
23--29 px relative to observed top-only last-valid bottom edges.

## Test strategy

- Reproduce latest g1 and g3 first-frame boxes and assert neither guard fires.
- Assert literal horizontal entry boundaries at center x just outside 20% and
  80%, and literal recovery boundaries at 25% and 75%.
- Assert literal vertical bottom-edge entry at 74 px versus 75 px and recovery
  at 109 px versus 110 px.
- Update vertical-priority and resume-path tests to use the new metrics.
- Run focused visual-servo tests and the complete `gear_sonic/tests` suite.

