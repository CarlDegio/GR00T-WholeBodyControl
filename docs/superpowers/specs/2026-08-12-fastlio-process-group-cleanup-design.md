# FAST-LIO Process-Group Cleanup Design

## Problem

`run_fastlio_supervisor.py` starts `ros2 launch fast_lio ...` in a new session.
The ROS launch process then starts `fastlio_mapping` in the same process group.

The current stop path sends a signal to the process group but waits only for the
ROS launch parent. ROS launch can exit before `fastlio_mapping` has stopped. The
stop function then returns, the supervisor starts another FAST-LIO instance, and
the surviving mapping process becomes an orphan. On the affected PC this produced
nine concurrent `fastlio_mapping` processes, several consuming multiple GiB of
RSS and a large fraction of one CPU core.

## Scope

This change fixes ownership and cleanup of the FAST-LIO process group only. It
does not tune FAST-LIO, change LiDAR or IMU topics, adjust time synchronization,
or change localization behavior. Localization drift is a separate subsequent
task and will be diagnosed using a single leak-free FAST-LIO instance.

## Process Ownership

The supervisor remains the owner of a dedicated FAST-LIO session created with
`start_new_session=True`. The ROS launch PID is therefore also the process-group
ID used for all cleanup operations.

The cleanup code must treat the group, not the ROS launch parent, as the unit of
liveness:

- A process group is alive while `os.killpg(pgid, 0)` succeeds or reports a
  permission error.
- The ROS launch `Popen` object is polled during waits so its exit status is
  reaped, but its exit alone does not complete cleanup.
- Cleanup sends SIGINT, waits for the entire group to disappear, then escalates
  to SIGTERM and SIGKILL with the existing bounded timeouts.
- If the group is still alive after SIGKILL, cleanup raises an error. The
  supervisor must not start a replacement while any member of the old group is
  known to remain.
- Calling cleanup after the whole group has already exited is idempotent.

## Restart and Shutdown Behavior

Both malicious-drift restart and normal supervisor shutdown use the same group
cleanup function. A drift restart launches the replacement only after group
cleanup has returned successfully. A cleanup failure stops the supervisor rather
than stacking another FAST-LIO instance.

This design does not add broad `pkill` or process-name matching. The supervisor
signals only the process group it created, so independently launched ROS or
FAST-LIO processes remain outside its scope.

## Testing

Unit tests will model the observed failure mode directly:

1. The ROS launch parent reports that it exited after SIGINT while the process
   group remains alive.
2. Cleanup must continue to SIGTERM instead of returning after the parent exit.
3. Cleanup returns only after the group disappears.
4. An already-dead group is a no-op.
5. A group surviving SIGKILL produces a clear failure and therefore prevents a
   restart.

After unit tests pass, a bounded PC integration check will repeatedly start and
stop a harmless process group and verify that no child remains. The FAST-LIO
runtime test will then verify the host has exactly one mapping process while the
supervisor is running and zero after shutdown.
