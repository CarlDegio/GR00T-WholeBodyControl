#pragma once

#include <array>
#include <atomic>
#include <utility>

// A demo may retain manual hand targets across body-mode switches. Collection
// leaves this disabled, so all hand targets continue to come from PICO.
class ManualGripperLatch {
 public:
  explicit ManualGripperLatch(bool enabled = false) : enabled_(enabled) {}

  void Set(bool left_closed, bool right_closed) {
    if (enabled_) {
      targets_.store(static_cast<int>(left_closed) | (static_cast<int>(right_closed) << 1));
    }
  }

  bool IsActive() const { return targets_.load() >= 0; }

  void Release() { targets_.store(-1); }

  std::pair<bool, std::array<double, 7>> GetHandPose(bool is_left) const {
    const int targets = targets_.load();
    std::array<double, 7> joints{};
    if (targets < 0) return {false, joints};
    joints[0] = (targets & (is_left ? 1 : 2)) ? 1.0 : 0.0;
    return {true, joints};
  }

 private:
  const bool enabled_;
  std::atomic<int> targets_{-1};
};
