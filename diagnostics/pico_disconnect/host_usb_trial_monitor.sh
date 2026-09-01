#!/usr/bin/env bash

# Host-side monitor for controlled PICO cable A/B trials.
# This intentionally does not invoke adb or change any device/service state.

set -uo pipefail

if [[ $# -ne 1 || -z "$1" ]]; then
  echo "usage: $0 OUTPUT_DIRECTORY" >&2
  exit 2
fi

trial_output_dir=$1
if [[ -e "$trial_output_dir" ]]; then
  echo "output directory already exists: $trial_output_dir" >&2
  exit 2
fi

/usr/bin/install -d "$trial_output_dir"
echo "$$" > "$trial_output_dir/monitor.pid"
/usr/bin/date -Ins > "$trial_output_dir/started_at.txt"

cleanup_trial_monitor() {
  /usr/bin/jobs -pr | /usr/bin/xargs -r /usr/bin/kill 2>/dev/null || true
  /usr/bin/date -Ins > "$trial_output_dir/stopped_at.txt"
}
trap cleanup_trial_monitor EXIT INT TERM

/usr/bin/stdbuf -oL -eL /usr/bin/journalctl -kf -o short-precise \
  > "$trial_output_dir/kernel_follow.log" 2>&1 &
/usr/bin/stdbuf -oL -eL /usr/bin/udevadm monitor --kernel --udev --property \
  --subsystem-match=usb --subsystem-match=net \
  > "$trial_output_dir/udev_follow.log" 2>&1 &
/usr/bin/stdbuf -oL -eL /usr/sbin/ip -ts monitor link address route \
  > "$trial_output_dir/ip_follow.log" 2>&1 &

/usr/bin/lsusb > "$trial_output_dir/lsusb_start.txt" 2>&1 || true
/usr/bin/lsusb -t > "$trial_output_dir/lsusb_tree_start.txt" 2>&1 || true
/usr/sbin/ip -details -statistics address show \
  > "$trial_output_dir/ip_start.txt" 2>&1 || true
/usr/bin/ps -eo pid,ppid,lstart,etimes,pcpu,pmem,stat,args --sort=pid \
  > "$trial_output_dir/processes_start.txt" 2>&1 || true

while true; do
  {
    printf '=== host=%s ===\n' "$(/usr/bin/date -Ins)"
    printf 'loadavg='
    /usr/bin/tr -d '\n' < /proc/loadavg
    printf '\n'

    pico_present=0
    for usb_device_path in /sys/bus/usb/devices/*; do
      [[ -r "$usb_device_path/idVendor" ]] || continue
      usb_vendor=$(/usr/bin/tr -d '\n' < "$usb_device_path/idVendor")
      [[ "$usb_vendor" == "05c6" || "$usb_vendor" == "2d40" ]] || continue
      usb_product=$(/usr/bin/tr -d '\n' < "$usb_device_path/idProduct")
      [[ "$usb_product" == "9024" || "$usb_product" == "00b5" ]] || continue
      pico_present=1
      printf 'pico path=%s vidpid=%s:%s' \
        "${usb_device_path##*/}" "$usb_vendor" "$usb_product"
      for usb_attribute in speed authorized bMaxPower; do
        if [[ -r "$usb_device_path/$usb_attribute" ]]; then
          printf ' %s=' "$usb_attribute"
          /usr/bin/tr -d '\n' < "$usb_device_path/$usb_attribute"
        fi
      done
      if [[ -r "$usb_device_path/power/runtime_status" ]]; then
        printf ' runtime_status='
        /usr/bin/tr -d '\n' < "$usb_device_path/power/runtime_status"
      fi
      printf '\n'
    done
    [[ "$pico_present" == 1 ]] || printf 'pico present=0\n'

    for network_path in /sys/class/net/*; do
      network_name=${network_path##*/}
      network_driver=$(
        /usr/bin/basename "$(/usr/bin/readlink -f "$network_path/device/driver" 2>/dev/null)" \
          2>/dev/null || true
      )
      [[ "$network_driver" == "rndis_host" || "$network_name" == enx* || "$network_name" == usb* ]] || continue
      printf 'net name=%s driver=%s' "$network_name" "${network_driver:-none}"
      for network_attribute in operstate carrier; do
        if [[ -r "$network_path/$network_attribute" ]]; then
          printf ' %s=' "$network_attribute"
          /usr/bin/tr -d '\n' < "$network_path/$network_attribute" 2>/dev/null || printf '?'
        fi
      done
      for network_stat in rx_bytes tx_bytes rx_packets tx_packets rx_errors tx_errors rx_dropped tx_dropped; do
        if [[ -r "$network_path/statistics/$network_stat" ]]; then
          printf ' %s=' "$network_stat"
          /usr/bin/tr -d '\n' < "$network_path/statistics/$network_stat"
        fi
      done
      printf '\n'
    done

    /usr/sbin/ip -brief address show 2>/dev/null || true
    /usr/bin/ss -Htanp 2>/dev/null | /usr/bin/grep -E \
      'RoboticsService|192\.168\.30\.|:13579|:60061' || true
    /usr/bin/ps -eo pid,ppid,pcpu,pmem,stat,args --sort=-pcpu 2>/dev/null \
      | /usr/bin/grep -E \
        'RoboticsService|launch_data_collection|pico_manager|pico_video\.service|data_collection\.service|gateway\.services\.(sensor|control)' \
      | /usr/bin/grep -v grep || true
    /usr/bin/nvidia-smi \
      --query-gpu=timestamp,temperature.gpu,utilization.gpu,memory.used,power.draw,clocks.current.graphics \
      --format=csv,noheader,nounits 2>/dev/null || true
  } >> "$trial_output_dir/snapshots.log" 2>&1
  /usr/bin/sleep 0.5
done
