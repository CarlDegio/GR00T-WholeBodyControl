# 时间同步（Workstation <-> ORIN）

本文档记录当前项目使用 `chrony` 进行时间同步的方式。目标是在有线网络直连时，让 ORIN 优先跟随工作站时间，同时保留公网 NTP 作为兜底。

## 1. 网络与目标

- 工作站与 ORIN 通过网线连接（网段：`192.168.123.0/24`）。
- 工作站作为 ORIN 的首选时间源。
- 外部 NTP 源作为备用，保证在断开直连时也能继续校时。

## 2. 工作站配置

### 2.1 安装 chrony

```bash
sudo apt update
sudo apt install -y chrony
```

### 2.2 修改配置文件

编辑 `/etc/chrony/chrony.conf`，确保包含以下内容：

```conf
server ntp.aliyun.com iburst
server ntp1.ntsc.ac.cn iburst

allow 192.168.123.0/24
```

说明：
- `server ... iburst`：启动时加速同步。
- `allow 192.168.123.0/24`：允许该有线网段内的设备（ORIN）向本机请求时间。

### 2.3 重启服务

```bash
sudo systemctl restart chrony
sudo systemctl enable chrony
```

## 3. ORIN 配置

### 3.1 安装 chrony

```bash
sudo apt update
sudo apt install -y chrony
```

### 3.2 修改配置文件

编辑 `/etc/chrony/chrony.conf`，设置为：

```conf
server [workstation IP] iburst prefer

server ntp.aliyun.com iburst
server ntp.ntsc.ac.cn iburst
```

说明：
- `[workstation IP]` 替换为工作站在有线网卡上的实际 IP（例如 `192.168.123.1`）。
- `prefer` 表示优先使用工作站时间源。
- 后两条公网 NTP 为备用源。

### 3.3 重启服务

```bash
sudo systemctl restart chrony
sudo systemctl enable chrony
```

## 4. 验证

可先快速判断当前系统实际启用的是哪套时间同步服务：

```bash
timedatectl status
systemctl list-units --type=service | grep -E 'chrony|timesyncd|ntp'
```

判断参考：
- 若看到 `chronyd.service` 处于 `active (running)`，通常表示正在使用 `chrony`。
- 若看到 `systemd-timesyncd.service` 为 `active (running)`，则系统可能在使用 `timesyncd`（不是 `chrony`）。
- 若看到 `ntp.service` / `ntpd.service` 为 `active (running)`，则系统可能在使用 `ntp`。
- 实践中建议只保留一种时间同步服务为 `active`，避免多个服务同时校时导致冲突。

之后在两台机器上可用以下命令检查状态：

```bash
chronyc sources -v
chronyc tracking
```

预期现象：
- 在有线网络连接正常时，ORIN 会优先锁定工作站时间源。
- 两台机器时间误差会缩小到十几个 ppm 以内（按当前实测经验）。

## 5. 备注

- 如果切换网络环境，确认 `allow` 网段和 ORIN 侧的工作站 IP 是否仍然正确。
- 若同步不稳定，优先检查：网线链路、双方防火墙策略、`chrony` 服务状态。
