#pragma once

#include <algorithm>
#include <array>

// Control-thread-owned reference ramp. Re-entering planner starts from measured
// joint angles; once settled, every planner frame uses the same standing target.
class PlannerUpperBodyHold {
 public:
  using Joints = std::array<double, 17>;

  static bool ShouldHold(bool demo_controls, bool planner_active,
                         bool has_upper_body, bool pose_entry) {
    return demo_controls && planner_active && !(has_upper_body && pose_entry);
  }

  void Reset() { active_ = false; }

  void Update(const Joints& measured, const Joints& target, double now,
              Joints& positions, Joints& velocities, bool hold_current = false) {
    if (!active_ || hold_current != hold_current_) {
      start_ = measured;
      target_ = hold_current ? measured : target;
      hold_current_ = hold_current;
      started_at_ = now;
      active_ = true;
    }
    constexpr double duration = 2.0;
    const double t = std::clamp((now - started_at_) / duration, 0.0, 1.0);
    const double alpha = t * t * (3.0 - 2.0 * t);
    const double rate = 6.0 * t * (1.0 - t) / duration;
    for (size_t i = 0; i < positions.size(); ++i) {
      positions[i] = start_[i] + alpha * (target_[i] - start_[i]);
      velocities[i] = rate * (target_[i] - start_[i]);
    }
  }

 private:
  bool active_ = false;
  bool hold_current_ = false;
  double started_at_ = 0.0;
  Joints start_{};
  Joints target_{};
};
