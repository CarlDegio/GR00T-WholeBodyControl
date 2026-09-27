# Teaser: front rim correction

Compared the original head frame plus 8 neighboring recorded frames on each side (17 total). Selected 3 frames later: sequence 8587, source-video time 332.091409 s, offset +0.255685 s from V7.

Same offline YOLOE prompt `green rubbish bin` and existing RGB/mask Hough detector. The displayed segment [342.0, 117.0] → [403.0, 118.0] is a detected line on the front lip, not manually invented coordinates. It is also the longest candidate in this frame. Only the second inset changed; exact pixel comparison confirms the rest is identical to V7. No experiment annotations changed.
