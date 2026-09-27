#include "planner_upper_body_hold.hpp"
#include <cassert>
#include <cmath>

int main() {
  // Collection must preserve PICO references in every planner/POSE state,
  // including PLANNER_FROZEN_UPPER_BODY without a pose-entry extension.
  for (bool planner_active : {false, true}) {
    for (bool has_upper_body : {false, true}) {
      for (bool pose_entry : {false, true}) {
        assert(!PlannerUpperBodyHold::ShouldHold(
            false, planner_active, has_upper_body, pose_entry));
      }
    }
  }
  assert(PlannerUpperBodyHold::ShouldHold(true, true, false, false));
  assert(PlannerUpperBodyHold::ShouldHold(true, true, true, false));
  assert(!PlannerUpperBodyHold::ShouldHold(true, true, true, true));
  assert(PlannerUpperBodyHold::ShouldHold(true, true, false, true));
  assert(!PlannerUpperBodyHold::ShouldHold(true, false, true, false));
  PlannerUpperBodyHold hold;
  PlannerUpperBodyHold::Joints measured{}, target{}, q{}, dq{};
  measured.fill(1.0);
  target.fill(0.2);
  hold.Update(measured, target, 10.0, q, dq);
  assert(q == measured);
  for (double v : dq) assert(v == 0.0);
  measured.fill(-1.0);  // Feedback noise must not restart the ramp.
  hold.Update(measured, target, 11.0, q, dq);
  for (double v : q) assert(std::abs(v - 0.6) < 1e-12);
  for (double v : dq) assert(std::abs(v + 0.6) < 1e-12);
  hold.Update(measured, target, 12.0, q, dq);
  for (double v : q) assert(std::abs(v - 0.2) < 1e-12);
  for (double v : dq) assert(v == 0.0);
  hold.Update(measured, target, 100.0, q, dq);
  for (double v : q) assert(std::abs(v - 0.2) < 1e-12);
  hold.Reset();
  hold.Update(measured, target, 101.0, q, dq);
  assert(q == measured);  // A new planner entry captures new measured angles.
  for (double v : dq) assert(v == 0.0);
  hold.Reset();
  measured.fill(0.8);
  hold.Update(measured, target, 102.0, q, dq, true);
  measured.fill(0.9);
  hold.Update(measured, target, 104.0, q, dq, true);
  for (double v : q) assert(std::abs(v - 0.8) < 1e-12);
  for (double v : dq) assert(v == 0.0);
  // Explicitly returning to default posture begins a new measured-angle ramp.
  hold.Update(measured, target, 105.0, q, dq, false);
  assert(q == measured);
}
