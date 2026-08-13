# AgentsDock Stable User Installation Design

## Goal

Install the official AgentsDock stable release on this Ubuntu 22.04 x86_64
workstation without requiring root access. The application must discover the
existing Codex CLI and open `/home/user/Project` as its initial workspace.

## Selected release and trust boundary

- Install immutable release `v0.2.9` from `ZhengyiLuo/AgentsDock-Releases`.
- Use `AgentsDock-0.2.9-linux-x86_64.AppImage`.
- Require the published SHA-256 digest
  `e1a7e716669ade27c740a9da8682ffa45e5aae73697435255980c6fee05fe78a`
  to match before installation.
- Do not copy, edit, print, or migrate files under `~/.codex`. AgentsDock will
  discover `/home/user/.local/bin/codex` and use that CLI's existing sign-in
  state.

## Files and launch behavior

The user-owned installation has four parts:

1. `~/.local/opt/agentsdock/AgentsDock.AppImage` is the fixed executable path.
2. `~/.local/bin/agentsdock` is a small launcher that adds `~/.local/bin` to
   `PATH`, changes to `/home/user/Project`, and executes the AppImage.
3. `~/.local/share/applications/agentsdock.desktop` exposes the same launcher
   in the GNOME application menu.
4. An icon extracted from the signed AppImage is stored below the user data
   directory and referenced by the desktop entry when extraction succeeds.

The installation does not write `/opt`, `/usr`, or other system-owned paths.

## Update and rollback

The fixed AppImage path lets the in-app stable updater replace a user-owned
file without `sudo`. A manual upgrade follows the same safe sequence: download
to a temporary file, verify the release digest, retain the current AppImage as
`AgentsDock.AppImage.previous`, and atomically replace the fixed path. Only one
previous version is retained.

The desktop entry and CLI launcher do not contain a version number, so neither
needs to change during ordinary updates. The stable update channel remains in
use; beta releases are not installed automatically as part of this task.

## Failure handling and verification

Installation stops without replacing the current executable if download or
SHA-256 validation fails. Existing user files are preserved.

Verification consists of:

- confirming the installed AppImage digest and executable permission;
- confirming the launcher resolves `codex-cli 0.147.0` through its `PATH`;
- validating the desktop entry;
- running a bounded non-interactive application smoke test and inspecting its
  exit/output for missing runtime libraries;
- launching the GUI from `/home/user/Project` for the user to complete any
  first-run UI confirmation that AgentsDock itself requires.

No provider credential is read or displayed during verification.
