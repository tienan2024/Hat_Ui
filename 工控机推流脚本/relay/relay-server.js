/**
 * 传感器 + 热像仪中继服务 - 运行在工控机上
 * 功能：串口(MCU/Lepton) <-> TCP(PC) 双向透传
 *
 * 数据流：
 *   MCU --串口--> 工控机 --TCP(6001)--> PC (传感器数据 $DATA、应答 $ACK)
 *   PC  --TCP(6001)--> 工控机 --串口--> MCU (报警指令 $ALARM、$NORMAL)
 *   Lepton --串口--> 工控机 --TCP(6002)--> PC (原始二进制热成像数据)
 *   PC  --TCP(6002)--> 工控机 --串口--> Lepton (初始化指令)
 *
 * 视频流：USB摄像头通过 Python/OpenCV video-streamer.py 单独运行，不在此服务中管理
 */

import { SerialPort } from 'serialport';
import { createServer as createTcpServer } from 'node:net';
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// 读取配置
const CONFIG_PATH = join(__dirname, 'config.json');
let config = {
  serial: { port: 'COM3', baudRate: 115200 },
  tcp: { port: 6001 },
  lepton: { enabled: true, port: 'COM5', baudRate: 921600, tcpPort: 6002 }
};
try {
  const loaded = JSON.parse(readFileSync(CONFIG_PATH, 'utf8'));
  config = {
    serial: loaded.serial || config.serial,
    tcp: loaded.tcp || config.tcp,
    lepton: loaded.lepton || config.lepton
  };
} catch {
  console.warn('[Relay] config.json not found, using defaults');
}

const SERIAL_PORT = config.serial.port;
const BAUD_RATE = config.serial.baudRate;
const TCP_PORT = config.tcp.port;
const LEPTON_ENABLED = config.lepton.enabled;
const LEPTON_PORT = config.lepton.port;
const LEPTON_BAUD = config.lepton.baudRate;
const LEPTON_TCP_PORT = config.lepton.tcpPort;

// ============ 日志 ============

function log(msg) {
  const ts = new Date().toLocaleTimeString('zh-CN', { hour12: false });
  console.log(`[${ts}] ${msg}`);
}

function logMCU(msg) { log(`[MCU] ${msg}`); }
function logLepton(msg) { log(`[Lepton] ${msg}`); }

// ============ MCU 串口侧 ============

let mcuSerialPort = null;
let currentMCUSerialPath = SERIAL_PORT;
let mcuTcpClient = null;
let mcuTcpServer = null;
let mcuSerialBuffer = '';
let mcuPingTimer = null;
const MCU_PING_TIMEOUT = 30000;

function openMCUSerial(portPath) {
  if (portPath) currentMCUSerialPath = portPath;
  logMCU(`Opening ${currentMCUSerialPath} @ ${BAUD_RATE}...`);

  if (mcuSerialPort) {
    mcuSerialPort.removeAllListeners();
    if (mcuSerialPort.isOpen) {
      mcuSerialPort.close(() => {
        mcuSerialPort = null;
        doOpenMCUSerial();
      });
      return;
    }
    mcuSerialPort = null;
  }
  doOpenMCUSerial();
}

function doOpenMCUSerial() {
  const port = new SerialPort({
    path: currentMCUSerialPath,
    baudRate: BAUD_RATE,
    dataBits: 8,
    parity: 'none',
    stopBits: 1,
    autoOpen: false
  });

  mcuSerialPort = port;

  port.on('data', (data) => {
    mcuSerialBuffer += data.toString();
    const lines = mcuSerialBuffer.split('\r\n');
    mcuSerialBuffer = lines.pop();
    for (const line of lines) {
      if (line.length > 0) handleMCUSerialLine(line);
    }
  });

  port.on('error', (err) => { logMCU(`Error: ${err.message}`); });

  port.on('close', () => {
    logMCU('Port closed');
    if (mcuSerialPort === port) mcuSerialPort = null;
    sendToMCUPC('$STATUS,DISCONNECTED\r\n');
    setTimeout(() => openMCUSerial(), 3000);
  });

  port.open((err) => {
    if (err) {
      logMCU(`Open error: ${err.message}`);
      setTimeout(() => openMCUSerial(), 3000);
      return;
    }
    logMCU(`Opened: ${currentMCUSerialPath}`);
    sendToMCUPC('$STATUS,CONNECTED\r\n');
  });
}

function handleMCUSerialLine(line) {
  logMCU(`RX: ${line}`);
  sendToMCUPC(line + '\r\n');
}

function sendToMCU(data) {
  if (mcuSerialPort?.isOpen) {
    mcuSerialPort.write(data, (err) => {
      if (err) logMCU(`TX Error: ${err.message}`);
      else logMCU(`TX: ${data.trim()}`);
    });
  } else {
    logMCU(`TX: Port not open, dropped`);
  }
}

// ============ MCU TCP 侧 ============

function cleanupMCUTcpClient(reason) {
  if (mcuTcpClient) {
    log(`[MCU-TCP] Cleaning up PC connection: ${reason}`);
    mcuTcpClient.removeAllListeners();
    mcuTcpClient.destroy();
    mcuTcpClient = null;
  }
  clearMCUPingTimer();
}

function startMCUTCPServer() {
  if (mcuTcpServer) {
    try { mcuTcpServer.close(); } catch {}
    mcuTcpServer = null;
  }

  const server = createTcpServer({ allowHalfOpen: false }, (socket) => {
    cleanupMCUTcpClient('new connection');
    mcuTcpClient = socket;
    socket.setKeepAlive(true, 10000);
    socket.setNoDelay(true);

    log(`[MCU-TCP] PC connected: ${socket.remoteAddress}:${socket.remotePort}`);
    resetMCUPingTimer();

    socket.on('data', (data) => {
      const lines = data.toString().split('\r\n');
      for (const line of lines) {
        if (line.length > 0) handleMCUTCPCommand(line);
      }
    });

    socket.on('error', (err) => { log(`[MCU-TCP] Socket error: ${err.message}`); cleanupMCUTcpClient('socket error'); });
    socket.on('close', () => { log('[MCU-TCP] PC disconnected'); cleanupMCUTcpClient('socket close'); });
    socket.on('end', () => { log('[MCU-TCP] PC sent FIN'); });
  });

  server.on('error', (err) => {
    log(`[MCU-TCP] Server error: ${err.message}`);
    if (err.code === 'EADDRINUSE') log('[MCU-TCP] Port in use, retrying in 3s...');
    setTimeout(startMCUTCPServer, 3000);
  });

  server.listen(TCP_PORT, () => { log(`[MCU-TCP] Server listening on port ${TCP_PORT}`); });
  mcuTcpServer = server;
}

function handleMCUTCPCommand(line) {
  log(`[MCU-TCP RX] ${line}`);

  if (line === '$PING') {
    sendToMCUPC('$PONG\r\n');
    resetMCUPingTimer();
    return;
  }

  if (line.startsWith('$CMD,')) {
    const cmd = line.substring(5);
    resetMCUPingTimer();
    if (cmd === 'LIST_PORTS') {
      handleMCUMListPorts();
    } else if (cmd.startsWith('OPEN,')) {
      handleMCUOpenPort(cmd.substring(5));
    }
    return;
  }

  if (line.startsWith('$') && !line.startsWith('$CMD,')) {
    sendToMCU(line + '\r\n');
    resetMCUPingTimer();
  }
}

function sendToMCUPC(data) {
  if (mcuTcpClient && !mcuTcpClient.destroyed && mcuTcpClient.writable) {
    mcuTcpClient.write(data, (err) => { if (err) log(`[MCU-TCP TX] Error: ${err.message}`); });
  }
}

function resetMCUPingTimer() {
  clearMCUPingTimer();
  mcuPingTimer = setTimeout(() => {
    log('[MCU-TCP] Ping timeout, disconnecting PC');
    cleanupMCUTcpClient('ping timeout');
  }, MCU_PING_TIMEOUT);
}

function clearMCUPingTimer() {
  if (mcuPingTimer) { clearTimeout(mcuPingTimer); mcuPingTimer = null; }
}

async function handleMCUMListPorts() {
  try {
    const ports = await SerialPort.list();
    const portPaths = ports.map(p => p.path);
    const resp = `$PORTS,${portPaths.join(',')}\r\n`;
    logMCU(`LIST_PORTS -> ${portPaths.length} ports: ${portPaths.join(', ')}`);
    sendToMCUPC(resp);
  } catch (err) {
    logMCU(`LIST_PORTS error: ${err.message}`);
    sendToMCUPC('$PORTS,\r\n');
  }
}

function handleMCUOpenPort(portPath) {
  if (!portPath || portPath.trim() === '') {
    sendToMCUPC('$CMD_ERR,OPEN,invalid port path\r\n');
    return;
  }
  portPath = portPath.trim();
  if (mcuSerialPort?.isOpen && currentMCUSerialPath === portPath) {
    sendToMCUPC('$CMD_OK,OPEN\r\n');
    return;
  }
  if (mcuSerialPort) {
    mcuSerialPort.removeAllListeners();
    if (mcuSerialPort.isOpen) {
      mcuSerialPort.close(() => {
        mcuSerialPort = null;
        openMCUSerial(portPath);
      });
    } else {
      mcuSerialPort = null;
      openMCUSerial(portPath);
    }
  } else {
    openMCUSerial(portPath);
  }
  const checkTimer = setInterval(() => {
    if (mcuSerialPort?.isOpen && currentMCUSerialPath === portPath) {
      clearInterval(checkTimer);
      sendToMCUPC('$CMD_OK,OPEN\r\n');
    }
  }, 100);
  setTimeout(() => clearInterval(checkTimer), 2000);
}

// ============ Lepton 串口侧 ============

let leptonSerialPort = null;
let currentLeptonSerialPath = LEPTON_PORT;
let leptonTcpClient = null;
let leptonTcpServer = null;
let leptonBuffer = Buffer.alloc(0);
let leptonPingTimer = null;
const LEPTON_PING_TIMEOUT = 30000;

function openLeptonSerial(portPath) {
  if (portPath) currentLeptonSerialPath = portPath;
  logLepton(`Opening ${currentLeptonSerialPath} @ ${LEPTON_BAUD}...`);

  if (leptonSerialPort) {
    leptonSerialPort.removeAllListeners();
    if (leptonSerialPort.isOpen) {
      leptonSerialPort.close(() => {
        leptonSerialPort = null;
        doOpenLeptonSerial();
      });
      return;
    }
    leptonSerialPort = null;
  }
  doOpenLeptonSerial();
}

function doOpenLeptonSerial() {
  const port = new SerialPort({
    path: currentLeptonSerialPath,
    baudRate: LEPTON_BAUD,
    dataBits: 8,
    parity: 'none',
    stopBits: 1,
    autoOpen: false
  });

  leptonSerialPort = port;

  // Lepton 是纯二进制协议，直接透传所有数据到 TCP
  port.on('data', (data) => {
    if (leptonTcpClient && !leptonTcpClient.destroyed && leptonTcpClient.writable) {
      leptonTcpClient.write(data, (err) => {
        if (err) logLepton(`TCP TX Error: ${err.message}`);
      });
    }
  });

  port.on('error', (err) => { logLepton(`Error: ${err.message}`); });

  port.on('close', () => {
    logLepton('Port closed');
    if (leptonSerialPort === port) leptonSerialPort = null;
    sendToLeptonPC('$STATUS,DISCONNECTED\r\n');
    setTimeout(() => openLeptonSerial(), 3000);
  });

  port.open((err) => {
    if (err) {
      logLepton(`Open error: ${err.message}`);
      setTimeout(() => openLeptonSerial(), 3000);
      return;
    }
    logLepton(`Opened: ${currentLeptonSerialPath}`);
    // 发送 Lepton 初始化命令：自动输出
    const initCmd = Buffer.from([0x5A, 0x01, 0x01]);
    port.write(initCmd, (err) => {
      if (err) logLepton(`Init cmd error: ${err.message}`);
      else logLepton('Init cmd sent: 0x5A 0x01 0x01');
    });
    sendToLeptonPC('$STATUS,CONNECTED\r\n');
  });
}

// ============ Lepton TCP 侧 ============

function cleanupLeptonTcpClient(reason) {
  if (leptonTcpClient) {
    log(`[Lepton-TCP] Cleaning up PC connection: ${reason}`);
    leptonTcpClient.removeAllListeners();
    leptonTcpClient.destroy();
    leptonTcpClient = null;
  }
  clearLeptonPingTimer();
}

function startLeptonTCPServer() {
  if (leptonTcpServer) {
    try { leptonTcpServer.close(); } catch {}
    leptonTcpServer = null;
  }

  const server = createTcpServer({ allowHalfOpen: false }, (socket) => {
    cleanupLeptonTcpClient('new connection');
    leptonTcpClient = socket;
    socket.setKeepAlive(true, 10000);
    socket.setNoDelay(true);

    log(`[Lepton-TCP] PC connected: ${socket.remoteAddress}:${socket.remotePort}`);
    resetLeptonPingTimer();

    socket.on('data', (data) => {
      if (data.length > 0 && data[0] === 0x24) {
        const str = data.toString('ascii', 0, data.length);
        const lines = str.split('\r\n');
        for (const line of lines) {
          if (line.length > 0) {
            handleLeptonTCPCommand(line);
          }
        }
      } else {
        if (leptonSerialPort?.isOpen) {
          leptonSerialPort.write(data, (err) => {
            if (err) logLepton(`Serial TX Error: ${err.message}`);
          });
        }
      }
      resetLeptonPingTimer();
    });

    socket.on('error', (err) => { log(`[Lepton-TCP] Socket error: ${err.message}`); cleanupLeptonTcpClient('socket error'); });
    socket.on('close', () => { log('[Lepton-TCP] PC disconnected'); cleanupLeptonTcpClient('socket close'); });
    socket.on('end', () => { log('[Lepton-TCP] PC sent FIN'); });
  });

  server.on('error', (err) => {
    log(`[Lepton-TCP] Server error: ${err.message}`);
    if (err.code === 'EADDRINUSE') log('[Lepton-TCP] Port in use, retrying in 3s...');
    setTimeout(startLeptonTCPServer, 3000);
  });

  server.listen(LEPTON_TCP_PORT, () => { log(`[Lepton-TCP] Server listening on port ${LEPTON_TCP_PORT}`); });
  leptonTcpServer = server;
}

function handleLeptonTCPCommand(line) {
  log(`[Lepton-TCP RX] ${line}`);

  if (line === '$PING') {
    sendToLeptonPC('$PONG\r\n');
    resetLeptonPingTimer();
    return;
  }

  if (line.startsWith('$CMD,')) {
    const cmd = line.substring(5);
    resetLeptonPingTimer();
    if (cmd === 'LIST_PORTS') {
      handleLeptonListPorts();
    } else if (cmd.startsWith('OPEN,')) {
      handleLeptonOpenPort(cmd.substring(5));
    }
    return;
  }

  if (line.startsWith('$')) {
    if (leptonSerialPort?.isOpen) {
      leptonSerialPort.write(Buffer.from(line + '\r\n', 'ascii'), (err) => {
        if (err) logLepton(`Serial TX Error: ${err.message}`);
      });
    }
    resetLeptonPingTimer();
  }
}

function sendToLeptonPC(data) {
  if (leptonTcpClient && !leptonTcpClient.destroyed && leptonTcpClient.writable) {
    leptonTcpClient.write(data, (err) => { if (err) log(`[Lepton-TCP TX] Error: ${err.message}`); });
  }
}

function resetLeptonPingTimer() {
  clearLeptonPingTimer();
  leptonPingTimer = setTimeout(() => {
    log('[Lepton-TCP] Ping timeout, disconnecting PC');
    cleanupLeptonTcpClient('ping timeout');
  }, LEPTON_PING_TIMEOUT);
}

function clearLeptonPingTimer() {
  if (leptonPingTimer) { clearTimeout(leptonPingTimer); leptonPingTimer = null; }
}

async function handleLeptonListPorts() {
  try {
    const ports = await SerialPort.list();
    const portPaths = ports.map(p => p.path);
    const resp = `$PORTS,${portPaths.join(',')}\r\n`;
    logLepton(`LIST_PORTS -> ${portPaths.length} ports: ${portPaths.join(', ')}`);
    sendToLeptonPC(resp);
  } catch (err) {
    logLepton(`LIST_PORTS error: ${err.message}`);
    sendToLeptonPC('$PORTS,\r\n');
  }
}

function handleLeptonOpenPort(portPath) {
  if (!portPath || portPath.trim() === '') {
    sendToLeptonPC('$CMD_ERR,OPEN,invalid port path\r\n');
    return;
  }
  portPath = portPath.trim();
  if (leptonSerialPort?.isOpen && currentLeptonSerialPath === portPath) {
    sendToLeptonPC('$CMD_OK,OPEN\r\n');
    return;
  }
  if (leptonSerialPort) {
    leptonSerialPort.removeAllListeners();
    if (leptonSerialPort.isOpen) {
      leptonSerialPort.close(() => {
        leptonSerialPort = null;
        openLeptonSerial(portPath);
      });
    } else {
      leptonSerialPort = null;
      openLeptonSerial(portPath);
    }
  } else {
    openLeptonSerial(portPath);
  }
  const checkTimer = setInterval(() => {
    if (leptonSerialPort?.isOpen && currentLeptonSerialPath === portPath) {
      clearInterval(checkTimer);
      sendToLeptonPC('$CMD_OK,OPEN\r\n');
    }
  }, 100);
  setTimeout(() => clearInterval(checkTimer), 2000);
}

// ============ 启动 ============

log(`[Relay] MCU: serial=${SERIAL_PORT} @ ${BAUD_RATE}, tcp=${TCP_PORT}`);
openMCUSerial();
startMCUTCPServer();

if (LEPTON_ENABLED) {
  log(`[Relay] Lepton: serial=${LEPTON_PORT} @ ${LEPTON_BAUD}, tcp=${LEPTON_TCP_PORT}`);
  openLeptonSerial();
  startLeptonTCPServer();
} else {
  log('[Relay] Lepton relay disabled (set lepton.enabled=true in config to enable)');
}
