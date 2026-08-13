# AgentsDock Stable User Installation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Install official AgentsDock v0.2.9 for the current Ubuntu user, expose CLI and GNOME launchers, reuse the existing Codex CLI environment, and default launches to `/home/user/Project`.

**Architecture:** Treat the official AppImage as the immutable application payload and install it at one user-owned fixed path. A small launcher supplies the desktop process environment and working directory, while a version-independent desktop entry integrates it with GNOME. Download and verification happen in `/tmp`; the installed executable is replaced only after its published SHA-256 matches.

**Tech Stack:** Ubuntu 22.04 x86_64, AppImage, POSIX shell, freedesktop desktop entry, SHA-256, Codex CLI 0.147.0.

## Global Constraints

- Install immutable stable release `v0.2.9`, not a beta release.
- Use `AgentsDock-0.2.9-linux-x86_64.AppImage` from `ZhengyiLuo/AgentsDock-Releases`.
- Require SHA-256 `e1a7e716669ade27c740a9da8682ffa45e5aae73697435255980c6fee05fe78a` before installation.
- Install only below `/home/user/.local`; do not write `/opt`, `/usr`, or another system-owned path.
- Do not copy, edit, print, or migrate `/home/user/.codex` or provider credentials.
- Make `/home/user/.local/bin/codex` visible to AgentsDock and launch from `/home/user/Project`.
- Retain at most one previous AppImage for rollback.

---

### Task 1: Stage and verify the official stable payload

**Files:**
- Download: `/tmp/agentsdock-v0.2.9/AgentsDock-0.2.9-linux-x86_64.AppImage`
- Download: `/tmp/agentsdock-v0.2.9/SHA256SUMS`

**Interfaces:**
- Consumes: official immutable GitHub release assets for tag `v0.2.9`.
- Produces: a verified executable whose digest exactly matches the Global Constraints.

- [ ] **Step 1: Confirm there is no unreviewed installation**

Run:

```bash
find /home/user/.local/opt/agentsdock /home/user/.local/bin/agentsdock /home/user/.local/share/applications/agentsdock.desktop -maxdepth 1 -print 2>/dev/null
```

Expected: no output on this first installation. If files exist, inspect them and preserve the current AppImage as the single rollback file before replacement.

- [ ] **Step 2: Create the release staging directory**

Run: `mkdir -p /tmp/agentsdock-v0.2.9`

Expected: the directory exists and is owned by `user`.

- [ ] **Step 3: Download the official artifacts**

Run:

```bash
curl -fL --retry 3 -o /tmp/agentsdock-v0.2.9/AgentsDock-0.2.9-linux-x86_64.AppImage https://github.com/ZhengyiLuo/AgentsDock-Releases/releases/download/v0.2.9/AgentsDock-0.2.9-linux-x86_64.AppImage
curl -fL --retry 3 -o /tmp/agentsdock-v0.2.9/SHA256SUMS https://github.com/ZhengyiLuo/AgentsDock-Releases/releases/download/v0.2.9/SHA256SUMS
```

Expected: the AppImage size is `129118393` bytes and both requests complete successfully.

- [ ] **Step 4: Verify the artifact**

Run:

```bash
sha256sum /tmp/agentsdock-v0.2.9/AgentsDock-0.2.9-linux-x86_64.AppImage
grep 'AgentsDock-0.2.9-linux-x86_64.AppImage' /tmp/agentsdock-v0.2.9/SHA256SUMS
```

Expected: both values equal `e1a7e716669ade27c740a9da8682ffa45e5aae73697435255980c6fee05fe78a`. Stop without installing on any mismatch.

### Task 2: Install executable and desktop integration

**Files:**
- Install: `/home/user/.local/opt/agentsdock/AgentsDock.AppImage`
- Create: `/home/user/.local/bin/agentsdock`
- Create: `/home/user/.local/share/applications/agentsdock.desktop`
- Create if bundled: `/home/user/.local/share/icons/hicolor/512x512/apps/agentsdock.png`

**Interfaces:**
- Consumes: verified AppImage from Task 1 and `/home/user/.local/bin/codex`.
- Produces: `agentsdock [args...]` and a GNOME entry that both start the fixed AppImage from `/home/user/Project`.

- [ ] **Step 1: Stage the exact launcher**

Create `/tmp/agentsdock-v0.2.9/agentsdock` with mode `0755`:

```sh
#!/bin/sh
export PATH="/home/user/.local/bin:/usr/local/bin:/usr/bin:/bin${PATH:+:$PATH}"
cd /home/user/Project || exit 1
exec /home/user/.local/opt/agentsdock/AgentsDock.AppImage "$@"
```

- [ ] **Step 2: Stage the desktop entry**

Create `/tmp/agentsdock-v0.2.9/agentsdock.desktop` with mode `0644`:

```ini
[Desktop Entry]
Name=AgentsDock
Comment=Desktop workspace for coding agents
Exec=/home/user/.local/bin/agentsdock %U
Icon=agentsdock
Terminal=false
Type=Application
Categories=Development;IDE;
StartupNotify=true
MimeType=x-scheme-handler/agentsdock;
```

- [ ] **Step 3: Extract and locate the official icon**

Run from `/tmp/agentsdock-v0.2.9`:

```bash
chmod 0755 AgentsDock-0.2.9-linux-x86_64.AppImage
./AgentsDock-0.2.9-linux-x86_64.AppImage --appimage-extract >extract.log
find squashfs-root/usr/share/icons -type f -iname '*agentsdock*.png' -print | sort
```

Expected: extraction succeeds. Prefer the bundled 512x512 application PNG. If no PNG exists, omit the icon file and retain `Icon=agentsdock` for a safe desktop fallback.

- [ ] **Step 4: Install version-independent paths**

Run:

```bash
install -d -m 0755 /home/user/.local/opt/agentsdock /home/user/.local/bin /home/user/.local/share/applications
install -m 0755 /tmp/agentsdock-v0.2.9/AgentsDock-0.2.9-linux-x86_64.AppImage /home/user/.local/opt/agentsdock/AgentsDock.AppImage
install -m 0755 /tmp/agentsdock-v0.2.9/agentsdock /home/user/.local/bin/agentsdock
install -m 0644 /tmp/agentsdock-v0.2.9/agentsdock.desktop /home/user/.local/share/applications/agentsdock.desktop
```

Expected: all installed files are owned by `user`; no `sudo` is used.

- [ ] **Step 5: Install the icon and refresh desktop caches**

When the exact bundled icon is found, install it as:

```bash
install -d -m 0755 /home/user/.local/share/icons/hicolor/512x512/apps
install -m 0644 /tmp/agentsdock-v0.2.9/squashfs-root/usr/share/icons/hicolor/512x512/apps/agentsdock.png /home/user/.local/share/icons/hicolor/512x512/apps/agentsdock.png
update-desktop-database /home/user/.local/share/applications
gtk-update-icon-cache -f -t /home/user/.local/share/icons/hicolor
```

Expected: GNOME accepts the entry. Missing optional cache utilities are non-fatal because GNOME also discovers user entries lazily.

### Task 3: Verify runtime, Codex discovery, and updates

**Files:**
- Verify: `/home/user/.local/opt/agentsdock/AgentsDock.AppImage`
- Verify: `/home/user/.local/bin/agentsdock`
- Verify: `/home/user/.local/share/applications/agentsdock.desktop`

**Interfaces:**
- Consumes: installed artifacts from Task 2.
- Produces: evidence that the payload, launcher environment, desktop entry, and GUI runtime work.

- [ ] **Step 1: Verify digest, modes, and ownership**

Run:

```bash
sha256sum /home/user/.local/opt/agentsdock/AgentsDock.AppImage
stat -c '%a %U:%G %n' /home/user/.local/opt/agentsdock/AgentsDock.AppImage /home/user/.local/bin/agentsdock /home/user/.local/share/applications/agentsdock.desktop
```

Expected: the digest matches the Global Constraints; modes are `755`, `755`, and `644`; ownership is `user:user`.

- [ ] **Step 2: Validate launcher and desktop syntax**

Run:

```bash
sh -n /home/user/.local/bin/agentsdock
desktop-file-validate /home/user/.local/share/applications/agentsdock.desktop
```

Expected: both return zero. If the optional desktop validator is absent, verify the required keys with `grep` and report that limitation.

- [ ] **Step 3: Confirm Codex and workspace discovery**

Run:

```bash
env PATH=/home/user/.local/bin:/usr/local/bin:/usr/bin:/bin sh -c 'cd /home/user/Project && command -v codex && codex --version && pwd'
```

Expected:

```text
/home/user/.local/bin/codex
codex-cli 0.147.0
/home/user/Project
```

- [ ] **Step 4: Run bounded AppImage smoke checks**

Run:

```bash
/home/user/.local/opt/agentsdock/AgentsDock.AppImage --appimage-version
timeout 15s /home/user/.local/bin/agentsdock --version
```

Expected: metadata is readable. The application either prints its version or reaches the timeout without missing-library, sandbox, or executable-format errors.

- [ ] **Step 5: Launch the GUI**

Run from the active X11 session:

```bash
nohup /home/user/.local/bin/agentsdock >/tmp/agentsdock-v0.2.9/gui.log 2>&1 &
```

Expected: AgentsDock opens with `/home/user/Project` as launch context and can select the existing Codex provider. Any provider login confirmation remains an explicit user UI action.

- [ ] **Step 6: Confirm update and rollback invariants**

Run:

```bash
test -w /home/user/.local/opt/agentsdock/AgentsDock.AppImage
find /home/user/.local/opt/agentsdock -maxdepth 1 -type f -printf '%f\n' | sort
```

Expected: the AppImage is writable by `user`. The directory contains `AgentsDock.AppImage` and at most `AgentsDock.AppImage.previous`.
