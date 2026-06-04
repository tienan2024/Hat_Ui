# 传感器中继服务 (relay-server)

运行在工控机上的轻量级 Node.js 服务，负责桥接 MCU 串口与前端服务器之间的通信。

## 架构

```
┌─────────┐  串口  ┌──────────────────┐  WiFi/TCP  ┌──────────────────┐
│  MCU    │──────→│  relay-server    │ ─────────→ │  前端服务器(PC)   │
│ 传感器   │       │  (工控机)         │            │  Node.js         │
│         │←──────│  串口 ↔ TCP 桥接  │ ←───────── │                  │
└─────────┘ $ACK  └──────────────────┘  报警指令   └──────────────────┘
```

## 功能

- 通过串口连接 MCU，接收传感器数据 (`$DATA` 帧) 和报警应答 (`$ACK` 帧)
- 通过 TCP 长连接暴露给前端服务器，透传所有原始协议帧
- 转发前端服务器的报警指令 (`$ALARM`/`$NORMAL` 帧) 到 MCU
- 串口断线自动重连（3秒间隔）
- 心跳检测（30秒无响应断开 PC 连接）
- TCP 断线自动重连（3秒间隔）

## 部署步骤

### 1. 安装依赖

```bash
cd relay
npm install
```

### 2. 修改配置

编辑 `config.json`：

```json
{
  "serial": {
    "port": "COM3",        ← 改为工控机连接MCU的串口号
    "baudRate": 115200
  },
  "tcp": {
    "port": 6001           ← TCP监听端口（前端服务器连接此端口）
  }
}
```

### 3. 启动服务

```bash
node relay-server.js
```

看到以下输出表示启动成功：

```
[Relay] Config loaded: serial=COM3 @ 115200, tcp=6001
[Relay] Opening serial port COM3...
[Relay] Serial port opened: COM3
[Relay] TCP server listening on port 6001, waiting for PC connection...
```

### 4. 前端服务器连接

在前端电脑的网页控制面板中：
1. 选择「远程工控机」连接模式
2. 输入工控机的 IP 地址和端口（默认 6001）
3. 点击「连接工控机」

## 自动启动（Windows）

创建快捷方式放到启动文件夹：

1. 右键 `relay-server.js` → 创建快捷方式
2. 修改快捷方式属性 → 目标改为：
   ```
   wscript.exe "C:\path\to\relay\start-hidden.vbs"
   ```
3. 将快捷方式放入 `shell:startup`

或使用 `start-hidden.vbs`：

```vbs
Set objShell = CreateObject("WScript.Shell")
objShell.Run "cmd /c cd /d C:\path\to\relay && node relay-server.js", 0, False
```

## 远程串口选择

连接工控机后，前端网页可以：
1. 点击「↻」按钮查询工控机上所有可用的串口
2. 从下拉列表中选择 MCU 连接的串口号
3. 点击「打开串口」让工控机连接该串口

这样无需手动编辑 `config.json`，直接在网页上就能切换串口。

## 故障排查

| 现象 | 原因 | 解决 |
|------|------|------|
| `Serial port open error` | 串口号错误或被占用 | 检查 `config.json` 中的串口号 |
| `No PC client connected` | 前端服务器未连接 | 在前端网页点击「连接工控机」 |
| 传感器数据不更新 | MCU 未发送数据 | 检查 MCU 串口连线和程序 |
| 频繁断线重连 | 网络不稳定 | 检查 WiFi 信号强度 |
