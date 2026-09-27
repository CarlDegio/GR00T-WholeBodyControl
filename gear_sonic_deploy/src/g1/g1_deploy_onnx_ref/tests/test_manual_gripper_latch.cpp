#include "input_interface/manual_gripper_latch.hpp"
#include <cassert>

int main() {
  ManualGripperLatch collection;
  ManualGripperLatch explicitly_disabled(false);
  ManualGripperLatch demo(true);
  assert(!demo.IsActive());
  assert(!demo.GetHandPose(true).first);
  assert(!demo.GetHandPose(false).first);

  for (bool left_closed : {false, true}) {
    for (bool right_closed : {false, true}) {
      // Even stray demo commands must never take hand control away from PICO.
      for (auto* latch : {&collection, &explicitly_disabled}) {
        latch->Set(left_closed, right_closed);
        assert(!latch->IsActive());
        assert(!latch->GetHandPose(true).first);
        assert(!latch->GetHandPose(false).first);
      }

      demo.Set(left_closed, right_closed);
      assert(demo.IsActive());
      const auto left = demo.GetHandPose(true);
      const auto right = demo.GetHandPose(false);
      assert(left.first && right.first);
      assert(left.second[0] == (left_closed ? 1.0 : 0.0));
      assert(right.second[0] == (right_closed ? 1.0 : 0.0));
      for (std::size_t i = 1; i < left.second.size(); ++i) {
        assert(left.second[i] == 0.0 && right.second[i] == 0.0);
      }
      // VLA releases the latch, and pause makes no further latch changes.
      demo.Release();
      demo.Release(); // Repeated PUB/SUB releases are idempotent.
      assert(!demo.IsActive());
      assert(!demo.GetHandPose(true).first && !demo.GetHandPose(false).first);
      // The next explicit manual command takes control back.
      demo.Set(left_closed, right_closed);
      assert(demo.IsActive());
      assert(demo.GetHandPose(true).second[0] == (left_closed ? 1.0 : 0.0));
      assert(demo.GetHandPose(false).second[0] == (right_closed ? 1.0 : 0.0));
    }
  }
}
