# -*- coding: utf-8 -*-
"""
hat_extensions.py — 智能巡检平台扩展模块

与 hat_main.py 完全解耦, 提供:
    1. 双模式热像仪连接 (WiFi TCP / Lepton 串口 / 模拟器)
    2. MCU 串口通信 (传感器数据上报 + 报警指令下发)
    3. 超温报警控制器 (阈值检测 + 状态机 + 串口联动)
    4. 热区追踪器 (跨帧追踪 + 叠加绘制)

依赖:
    pip install numpy pyserial
    (OpenCV 仅用于叠加绘制, 没有时退化为纯数据模式)

用法:
    from hat_extensions import ThermalCameraManager, MCUSerialBridge, AlarmController, ThermalZoneTracker
    详见 docs/02-扩展模块接口.md
"""

import json
import math
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    serial = None


# ════════════════════════════════════════════════════════════════
# 常量
# ════════════════════════════════════════════════════════════════

THERMAL_MATRIX_W = 80
THERMAL_MATRIX_H = 60
HIGH_TEMP_LIMIT = 70.0

# Lepton 帧协议
LEPTON_FRAME_SIZE = 9606   # 4(header) + 9600(4800*2) + 2(CRC16)
LEPTON_WIDTH = 80
LEPTON_HEIGHT = 60
LEPTON_TEMP_COUNT = LEPTON_WIDTH * LEPTON_HEIGHT  # 4800

# 颜色常量 (与主文件保持一致)
COLOR_GREEN = "#20e986"
COLOR_YELLOW = "#ffdf4d"
COLOR_RED = "#ff3038"
COLOR_ORANGE = "#ff8b25"
COLOR_CYAN = "#25eaff"


# ════════════════════════════════════════════════════════════════
# 数据结构
# ════════════════════════════════════════════════════════════════

@dataclass
class HotRegion:
    """高温区域结构。"""
    name: str
    x: int
    y: int
    temp: float
    level: str
    region_id: int = -1  # 追踪 ID, -1 表示未分配

    def dist_sq(self, other_x: int, other_y: int) -> float:
        return (self.x - other_x) ** 2 + (self.y - other_y) ** 2


@dataclass
class ThermalFrame:
    """热像仪帧数据。"""
    temps: np.ndarray       # 温度矩阵 (H x W)
    rgb: np.ndarray         # 伪彩图 (H x W x 3) uint8
    width: int
    height: int
    max_temp: float
    min_temp: float
    avg_temp: float
    center_temp: float
    max_pos: Tuple[int, int]  # (y, x)


@dataclass
class AlarmState:
    """报警状态。"""
    active: bool = False
    level: str = "normal"           # "normal" / "alarm" / "suppressed"
    color: str = COLOR_GREEN
    text: str = "正常"
    max_temp: float = 0.0
    threshold: float = 70.0


@dataclass
class SensorData:
    """MCU 传感器数据。"""
    dht11_temp: float = 0.0
    dht11_humi: float = 0.0
    ultrasonic: int = 0
    smoke: int = 0
    light: int = 0
    valid: bool = False
    timestamp: float = 0.0


# ════════════════════════════════════════════════════════════════
# §1 伪彩映射 (独立函数, 无外部依赖)
# ════════════════════════════════════════════════════════════════

def temperature_to_rgb(temp: np.ndarray, t_min: float = None, t_max: float = None) -> np.ndarray:
    """温度矩阵 → 伪彩 RGB 图 (蓝→青→黄→红)。

    Args:
        temp: 温度矩阵 (H x W)
        t_min: 映射下限, None 则自适应
        t_max: 映射上限, None 则自适应

    Returns:
        RGB 图像 (H x W x 3), uint8
    """
    if t_min is None:
        t_min = float(np.min(temp))
    if t_max is None:
        t_max = float(np.max(temp))
    t_range = t_max - t_min
    if t_range < 0.1:
        t_range = 1.0
    norm = np.clip((temp - t_min) / t_range, 0, 1)
    r = np.clip(2.2 * norm - 0.45, 0, 1)
    g = np.clip(1.7 - np.abs(norm - 0.55) * 3.0, 0, 1)
    b = np.clip(1.25 - norm * 1.7, 0, 1)
    return (np.dstack([r, g, b]) * 255).astype(np.uint8)


def temperature_alarm_color(temp: float, thresholds: dict = None) -> Tuple[str, str]:
    """根据温度返回 (等级名称, 颜色)。"""
    if thresholds is None:
        thresholds = {35: COLOR_YELLOW, 45: COLOR_ORANGE, 65: COLOR_RED}
    if temp >= 65:
        return "三级报警", COLOR_RED
    if temp >= 45:
        return "二级预警", COLOR_ORANGE
    if temp >= 35:
        return "一级预警", COLOR_YELLOW
    return "正常", COLOR_GREEN


# ════════════════════════════════════════════════════════════════
# §2 CRC16 校验 (Lepton 帧验证)
# ════════════════════════════════════════════════════════════════

_CRC_TAB_H = [
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0,
    0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0,
    0x80, 0x41, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40,
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1,
    0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1,
    0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0,
    0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40,
    0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1,
    0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40,
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0,
    0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40,
    0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0,
    0x80, 0x41, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40,
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0,
    0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0,
    0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41,
    0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0,
    0x80, 0x41, 0x00, 0xC1, 0x81, 0x40, 0x00, 0xC1, 0x81, 0x40,
    0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0, 0x80, 0x41, 0x00, 0xC1,
    0x81, 0x40, 0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41,
    0x00, 0xC1, 0x81, 0x40, 0x01, 0xC0, 0x80, 0x41, 0x01, 0xC0,
    0x80, 0x41, 0x00, 0xC1, 0x81, 0x40
]
_CRC_TAB_L = [
    0x00, 0xC0, 0xC1, 0x01, 0xC3, 0x03, 0x02, 0xC2, 0xC6, 0x06,
    0x07, 0xC7, 0x05, 0xC5, 0xC4, 0x04, 0xCC, 0x0C, 0x0D, 0xCD,
    0x0F, 0xCF, 0xCE, 0x0E, 0x0A, 0xCA, 0xCB, 0x0B, 0xC9, 0x09,
    0x08, 0xC8, 0xD8, 0x18, 0x19, 0xD9, 0x1B, 0xDB, 0xDA, 0x1A,
    0x1E, 0xDE, 0xDF, 0x1F, 0xDD, 0x1D, 0x1C, 0xDC, 0x14, 0xD4,
    0xD5, 0x15, 0xD7, 0x17, 0x16, 0xD6, 0xD2, 0x12, 0x13, 0xD3,
    0x11, 0xD1, 0xD0, 0x10, 0xF0, 0x30, 0x31, 0xF1, 0x33, 0xF3,
    0xF2, 0x32, 0x36, 0xF6, 0xF7, 0x37, 0xF5, 0x35, 0x34, 0xF4,
    0x3C, 0xFC, 0xFD, 0x3D, 0xFF, 0x3F, 0x3E, 0xFE, 0xFA, 0x3A,
    0x3B, 0xFB, 0x39, 0xF9, 0xF8, 0x38, 0x28, 0xE8, 0xE9, 0x29,
    0xEB, 0x2B, 0x2A, 0xEA, 0xEE, 0x2E, 0x2F, 0xEF, 0x2D, 0xED,
    0xEC, 0x2C, 0xE4, 0x24, 0x25, 0xE5, 0x27, 0xE7, 0xE6, 0x26,
    0x22, 0xE2, 0xE3, 0x23, 0xE1, 0x21, 0x20, 0xE0, 0xA0, 0x60,
    0x61, 0xA1, 0x63, 0xA3, 0xA2, 0x62, 0x66, 0xA6, 0xA7, 0x67,
    0xA5, 0x65, 0x64, 0xA4, 0x6C, 0xAC, 0xAD, 0x6D, 0xAF, 0x6F,
    0x6E, 0xAE, 0xAA, 0x6A, 0x6B, 0xAB, 0x69, 0xA9, 0xA8, 0x68,
    0x78, 0xB8, 0xB9, 0x79, 0xBB, 0x7B, 0x7A, 0xBA, 0xBE, 0x7E,
    0x7F, 0xBF, 0x7D, 0xBD, 0xBC, 0x7C, 0xB4, 0x74, 0x75, 0xB5,
    0x77, 0xB7, 0xB6, 0x76, 0x72, 0xB2, 0xB3, 0x73, 0xB1, 0x71,
    0x70, 0xB0, 0x50, 0x90, 0x91, 0x51, 0x93, 0x53, 0x52, 0x92,
    0x96, 0x56, 0x57, 0x97, 0x55, 0x95, 0x94, 0x54, 0x9C, 0x5C,
    0x5D, 0x9D, 0x5F, 0x9F, 0x9E, 0x5E, 0x5A, 0x9A, 0x9B, 0x5B,
    0x99, 0x59, 0x58, 0x98, 0x88, 0x48, 0x49, 0x89, 0x4B, 0x8B,
    0x8A, 0x4A, 0x4E, 0x8E, 0x8F, 0x4F, 0x8D, 0x4D, 0x4C, 0x8C,
    0x44, 0x84, 0x85, 0x45, 0x87, 0x47, 0x46, 0x86, 0x82, 0x42,
    0x43, 0x83, 0x41, 0x81, 0x80, 0x40
]


def _crc16(data: bytes) -> int:
    """计算 CRC16 (与 Lepton 帧校验一致)。"""
    crc_h = 0xFF
    crc_l = 0xFF
    for byte in data:
        idx = crc_h ^ byte
        crc_h = crc_l ^ _CRC_TAB_H[idx]
        crc_l = _CRC_TAB_L[idx]
    return (crc_h << 8) | crc_l


# ════════════════════════════════════════════════════════════════
# §3 ThermalCameraManager — 双模式热像仪管理器
# ════════════════════════════════════════════════════════════════

class ThermalWiFiClient:
    """WiFi 热像仪 TCP 客户端 — 长连接 + 后台拉取。

    协议: \\x02{"cmd":"get_image"}\\x03
    返回: JSON { radiometric: base64, width, height, maxTemp, ... }
    """

    def __init__(self, host: str = "192.168.3.166", port: int = 5001,
                 connect_timeout: float = 3.0, read_timeout: float = 8.0):
        self.host = host
        self.port = port
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._sock: Optional[socket.socket] = None
        self._connected = False
        self._lock = threading.Lock()
        self._latest: Optional[dict] = None
        self._fetch_thread: Optional[threading.Thread] = None
        self._fetching = False

    def connect(self) -> bool:
        with self._lock:
            self._cleanup()
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self._connect_timeout)
                sock.connect((self.host, self.port))
                sock.settimeout(self._read_timeout)
                self._sock = sock
                self._connected = True
                print(f"[ThermalWiFi] 已连接 {self.host}:{self.port}")
            except Exception as e:
                print(f"[ThermalWiFi] 连接失败: {e}")
                self._connected = False
                return False

        self._fetching = True
        self._fetch_thread = threading.Thread(target=self._fetch_loop, daemon=True)
        self._fetch_thread.start()
        return True

    def _fetch_loop(self):
        while self._fetching and self._connected:
            frame = self.request_frame()
            time.sleep(0.15 if frame else 0.5)

    def _cleanup(self):
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        self._connected = False

    def disconnect(self):
        self._fetching = False
        with self._lock:
            self._cleanup()

    def is_connected(self) -> bool:
        return self._connected and self._sock is not None

    def request_frame(self) -> Optional[dict]:
        with self._lock:
            if not self._connected or not self._sock:
                return None
            try:
                cmd = b'\x02{"cmd":"get_image"}\x03'
                self._sock.sendall(cmd)

                data = b''
                while True:
                    chunk = self._sock.recv(65536)
                    if not chunk:
                        raise ConnectionError("连接断开")
                    data += chunk
                    if b'\x03' in data:
                        break
                    if len(data) > 2_000_000:
                        raise ValueError("数据过大")

                json_str = data.split(b'\x03')[0].replace(b'\x02', b'').decode('utf-8')
                result = json.loads(json_str)

                radiometric = result.get('radiometric', '')
                width = result.get('width', 160)
                height = result.get('height', 120)
                if not radiometric:
                    return None

                import base64
                raw_bytes = base64.b64decode(radiometric)
                temps = np.zeros(width * height, dtype=np.float32)
                for i in range(width * height):
                    low = raw_bytes[i * 2]
                    high = raw_bytes[i * 2 + 1]
                    kelvin = (high << 8) | low
                    temps[i] = kelvin / 100.0 - 273.15

                temps_2d = temps.reshape(height, width)
                max_temp = float(np.max(temps_2d))
                min_temp = float(np.min(temps_2d))
                avg_temp = float(np.mean(temps_2d))
                max_pos = np.unravel_index(np.argmax(temps_2d), temps_2d.shape)
                center_temp = float(temps_2d[height // 2, width // 2])
                rgb = temperature_to_rgb(temps_2d)

                self._latest = {
                    'temps': temps_2d, 'rgb': rgb,
                    'width': width, 'height': height,
                    'maxTemp': max_temp, 'minTemp': min_temp,
                    'avgTemp': avg_temp, 'centerTemp': center_temp,
                    'maxPos': {'y': int(max_pos[0]), 'x': int(max_pos[1])},
                }
                return self._latest

            except Exception as e:
                print(f"[ThermalWiFi] 请求失败: {e}")
                self._connected = False
                return None

    def get_latest(self) -> Optional[dict]:
        return self._latest


class ThermalLeptonReader:
    """FLIR Lepton 2.5 串口红外热像仪驱动。

    支持两种连接模式:
      1. 本地串口 (local): 直连 Lepton 串口
      2. 远程 TCP (remote): 通过工控机 relay 连接 Lepton

    协议: 9606 字节帧
      Byte 0:      0x5A (帧头)
      Byte 1:      0x08 (80 = 宽度高字节)
      Byte 2:      0x06 (60 = 高度高字节)
      Byte 3:      设备ID
      Byte 4-9603: 温度数据 (4800点 x 2字节, 低位在前)
      Byte 9604-5: CRC16

    温度解码: (high<<8 | low) / 100 - 40 ℃
    """

    # 心跳间隔 (秒)
    PING_INTERVAL = 10
    # 远程重连间隔 (秒)
    REMOTE_RECONNECT_INTERVAL = 3

    def __init__(self, port: str = None, baudrate: int = 921600):
        self.port_name = port
        self.baudrate = baudrate
        self._port = None
        self._connected = False
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._buffer = bytearray()
        self._latest: Optional[dict] = None

        # 远程 TCP 模式
        self._mode: Optional[str] = None          # 'local' / 'remote' / None
        self._tcp_socket: Optional[socket.socket] = None
        self._remote_host: Optional[str] = None
        self._remote_port: int = 6002
        self._ping_timer: Optional[threading.Timer] = None
        self._reconnect_timer: Optional[threading.Timer] = None
        self._remote_ports: List[str] = []
        self._remote_cmd_response: Optional[str] = None
        self._remote_port_opened: bool = False    # 串口是否已打开（远程模式下）

    # ════════════════════════════════════════════════════════
    # 连接模式: 本地串口
    # ════════════════════════════════════════════════════════

    def connect(self, port: str = None) -> bool:
        """连接本地串口。"""
        if port:
            self.port_name = port
        if not self.port_name:
            print("[Lepton] 未指定串口")
            return False
        if serial is None:
            print("[Lepton] pyserial 未安装")
            return False

        # 先清理远程连接
        self._force_cleanup_tcp()

        with self._lock:
            self._disconnect_internal()
            try:
                self._port = serial.Serial(
                    port=self.port_name,
                    baudrate=self.baudrate,
                    bytesize=8, parity='N', stopbits=1,
                    timeout=0.1,
                )
                self._connected = True
                self._running = True
                self._mode = 'local'
                self._buffer = bytearray()
                print(f"[Lepton] 已打开本地串口 {self.port_name} @ {self.baudrate}")

                # 发送自动输出命令
                self._port.write(bytes([0x5A, 0x01, 0x01]))

                self._thread = threading.Thread(target=self._read_loop, daemon=True)
                self._thread.start()
                return True
            except Exception as e:
                print(f"[Lepton] 连接失败: {e}")
                self._connected = False
                return False

    def _disconnect_internal(self):
        self._running = False
        if self._port and self._port.is_open:
            try:
                self._port.close()
            except Exception:
                pass
        self._port = None
        if self._mode == 'local':
            self._connected = False
            self._mode = None

    # ════════════════════════════════════════════════════════
    # 连接模式: 远程 TCP
    # ════════════════════════════════════════════════════════

    def open_remote(self, host: str, port: int = 6002) -> bool:
        """通过 TCP 连接工控机上的 Lepton 中继服务。"""
        # 先清理旧连接
        self.disconnect()

        self._remote_host = host
        self._remote_port = port

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(5.0)
            sock.connect((host, port))
            sock.settimeout(None)
        except Exception as e:
            print(f"[Lepton] 远程连接失败: {e}")
            return False

        self._tcp_socket = sock
        self._connected = True
        self._running = True
        self._mode = 'remote'
        self._buffer = bytearray()
        self._remote_port_opened = False
        print(f"[Lepton] 已连接远程工控机 {host}:{port}")

        self._thread = threading.Thread(target=self._remote_read_loop, daemon=True)
        self._thread.start()
        self._reset_ping_timer()

        return True

    def _remote_read_loop(self):
        """远程 TCP 读取线程。"""
        while self._running and self._tcp_socket:
            try:
                data = self._tcp_socket.recv(4096)
                if not data:
                    print("[Lepton] 远程连接断开 (收到空数据)")
                    break
                self._buffer.extend(data)
                # 从缓冲区提取 ASCII 控制帧
                self._process_control_frames()
                # 处理二进制 Lepton 帧
                self._process_buffer()
            except OSError as e:
                if self._running:
                    print(f"[Lepton] 远程读取错误: {e}")
                break
            except Exception as e:
                if self._running:
                    print(f"[Lepton] 远程读取异常: {e}")
                break

        # 连接断开
        was_running = self._running
        self._connected = False
        self._mode = None
        self._clear_ping_timer()
        if self._tcp_socket:
            try:
                self._tcp_socket.close()
            except Exception:
                pass
            self._tcp_socket = None

        # 自动重连 (非主动断开时)
        if was_running and self._remote_host:
            print(f"[Lepton] 将在 {self.REMOTE_RECONNECT_INTERVAL} 秒后重连...")
            self._reconnect_timer = threading.Timer(
                self.REMOTE_RECONNECT_INTERVAL,
                self._auto_reconnect_remote
            )
            self._reconnect_timer.daemon = True
            self._reconnect_timer.start()

    def _auto_reconnect_remote(self):
        """自动重连远程工控机。"""
        if self._remote_host and not self._connected:
            print(f"[Lepton] 自动重连 {self._remote_host}:{self._remote_port}...")
            self.open_remote(self._remote_host, self._remote_port)

    def _force_cleanup_tcp(self):
        """强制清理远程 TCP 连接。"""
        self._clear_ping_timer()
        self._clear_reconnect_timer()
        if self._tcp_socket:
            try:
                self._tcp_socket.close()
            except Exception:
                pass
            self._tcp_socket = None
        if self._mode == 'remote':
            self._connected = False
            self._mode = None

    def _reset_ping_timer(self):
        """重置心跳定时器。"""
        self._clear_ping_timer()
        self._ping_timer = threading.Timer(self.PING_INTERVAL, self._send_ping)
        self._ping_timer.daemon = True
        self._ping_timer.start()

    def _clear_ping_timer(self):
        if self._ping_timer:
            self._ping_timer.cancel()
            self._ping_timer = None

    def _clear_reconnect_timer(self):
        if self._reconnect_timer:
            self._reconnect_timer.cancel()
            self._reconnect_timer = None

    def _send_ping(self):
        """发送心跳包。"""
        if self._mode == 'remote' and self._connected:
            self._write_raw(b'$PING\r\n')
            self._reset_ping_timer()

    # ════════════════════════════════════════════════════════
    # 远程控制命令
    # ════════════════════════════════════════════════════════

    def list_remote_ports(self) -> List[str]:
        """查询工控机上的串口列表。"""
        if self._mode != 'remote' or not self._connected:
            return []
        self._remote_cmd_response = None
        self._remote_ports = []
        self._write_raw(b'$CMD,LIST_PORTS\r\n')
        # 等待响应 (最多 2 秒)
        for _ in range(20):
            time.sleep(0.1)
            if self._remote_cmd_response is not None:
                break
        return list(self._remote_ports)

    def open_remote_port(self, port_name: str) -> bool:
        """请求工控机切换串口。"""
        if self._mode != 'remote' or not self._connected:
            return False
        self._remote_cmd_response = None
        self._write_raw(f'$CMD,OPEN,{port_name}\r\n'.encode('ascii'))
        # 等待响应 (最多 3 秒)
        for _ in range(30):
            time.sleep(0.1)
            if self._remote_cmd_response is not None:
                return self._remote_cmd_response == 'CMD_OK'
        return False

    # ════════════════════════════════════════════════════════
    # 通用断开/状态
    # ════════════════════════════════════════════════════════

    def disconnect(self):
        """断开所有连接 (本地串口和远程 TCP)。"""
        self._force_cleanup_tcp()
        with self._lock:
            self._disconnect_internal()

    def is_connected(self) -> bool:
        if self._mode == 'local':
            return self._connected and self._port is not None and self._port.is_open
        if self._mode == 'remote':
            return self._connected and self._tcp_socket is not None
        return False

    def get_mode(self) -> Optional[str]:
        """返回当前连接模式: 'local' / 'remote' / None。"""
        return self._mode

    def get_connection_info(self) -> dict:
        """返回连接信息字典。"""
        if self._mode == 'local':
            return {'mode': 'local', 'port': self.port_name, 'baudrate': self.baudrate}
        elif self._mode == 'remote':
            return {'mode': 'remote', 'host': self._remote_host, 'port': self._remote_port}
        return {'mode': None}

    # ════════════════════════════════════════════════════════
    # 数据读取 (本地串口)
    # ════════════════════════════════════════════════════════

    def _read_loop(self):
        """本地串口读取线程。"""
        while self._running:
            try:
                if not self._port or not self._port.is_open:
                    break
                data = self._port.read(4096)
                if data:
                    self._buffer.extend(data)
                    self._process_buffer()
            except Exception as e:
                if self._running:
                    print(f"[Lepton] 读取错误: {e}")
                break
        self._connected = False
        self._mode = None

    # ════════════════════════════════════════════════════════
    # 协议解析 (本地和远程共用)
    # ════════════════════════════════════════════════════════

    def _process_control_frames(self):
        """从缓冲区提取并处理 ASCII 控制帧（远程模式）。"""
        while b'\r\n' in self._buffer:
            line_end = self._buffer.index(b'\r\n')
            line = bytes(self._buffer[:line_end])
            self._buffer = self._buffer[line_end + 2:]
            try:
                line_str = line.decode('ascii', errors='ignore')
                if line_str:
                    self._process_control_line(line_str)
            except Exception:
                pass

    def _process_control_line(self, line: str):
        """解析中继控制帧。"""
        if not line:
            return

        if line == '$PONG':
            return

        if line.startswith('$PORTS,'):
            ports_str = line[7:]
            self._remote_ports = [p.strip() for p in ports_str.split(',') if p.strip()]
            self._remote_cmd_response = 'PORTS'
            print(f"[Lepton] 工控机串口列表: {self._remote_ports}")
            return

        if line == '$CMD_OK,OPEN':
            self._remote_cmd_response = 'CMD_OK'
            print("[Lepton] 工控机串口切换成功")
            return

        if line.startswith('$CMD_ERR,OPEN'):
            self._remote_cmd_response = 'CMD_ERR'
            reason = line[14:] if len(line) > 14 else 'unknown'
            print(f"[Lepton] 工控机串口切换失败: {reason}")
            return

        if line.startswith('$STATUS,'):
            status = line[8:]
            if status == 'CONNECTED':
                print("[Lepton] 工控机报告: Lepton 串口已连接")
            elif status == 'DISCONNECTED':
                print("[Lepton] 工控机报告: Lepton 串口已断开")
            return

    def _write_raw(self, data: bytes) -> bool:
        """底层写入, 根据模式选择输出到串口或 TCP。"""
        if self._mode == 'local':
            with self._lock:
                if not self._port or not self._port.is_open:
                    return False
                try:
                    self._port.write(data)
                    return True
                except Exception as e:
                    print(f"[Lepton] 本地发送失败: {e}")
                    return False
        elif self._mode == 'remote':
            if not self._tcp_socket:
                return False
            try:
                self._tcp_socket.sendall(data)
                return True
            except Exception as e:
                print(f"[Lepton] 远程发送失败: {e}")
                return False
        return False

    def _process_buffer(self):
        while len(self._buffer) >= LEPTON_FRAME_SIZE:
            header_idx = -1
            for i in range(len(self._buffer)):
                if self._buffer[i] == 0x5A:
                    header_idx = i
                    break

            if header_idx < 0:
                self._buffer.clear()
                return
            if header_idx > 0:
                self._buffer = self._buffer[header_idx:]
            if len(self._buffer) < LEPTON_FRAME_SIZE:
                return

            frame = bytes(self._buffer[:LEPTON_FRAME_SIZE])
            self._buffer = self._buffer[LEPTON_FRAME_SIZE:]

            if frame[0] != 0x5A:
                continue
            if frame[1] != 0x08 or frame[2] != 0x06:
                continue

            result = self._parse_frame(frame)
            if result:
                self._latest = result

    def _parse_frame(self, frame: bytes) -> Optional[dict]:
        device_id = frame[3]
        temps = np.zeros(LEPTON_TEMP_COUNT, dtype=np.float32)

        for i in range(LEPTON_TEMP_COUNT):
            byte_idx = 4 + i * 2
            low = frame[byte_idx]
            high = frame[byte_idx + 1]
            raw = (high << 8) | low
            temps[i] = raw / 100.0 - 40.0

        temps_2d = temps.reshape(LEPTON_HEIGHT, LEPTON_WIDTH)
        max_temp = float(np.max(temps_2d))
        min_temp = float(np.min(temps_2d))
        avg_temp = float(np.mean(temps_2d))
        max_pos = np.unravel_index(np.argmax(temps_2d), temps_2d.shape)
        center_temp = float(temps_2d[LEPTON_HEIGHT // 2, LEPTON_WIDTH // 2])
        rgb = temperature_to_rgb(temps_2d)

        return {
            'temps': temps_2d, 'rgb': rgb,
            'width': LEPTON_WIDTH, 'height': LEPTON_HEIGHT,
            'maxTemp': max_temp, 'minTemp': min_temp,
            'avgTemp': avg_temp, 'centerTemp': center_temp,
            'maxPos': {'y': int(max_pos[0]), 'x': int(max_pos[1])},
        }

    def get_latest(self) -> Optional[dict]:
        return self._latest


class ThermalSimulator:
    """80x60 热成像模拟器 (与主文件中的版本兼容)。"""

    def __init__(self, width: int = THERMAL_MATRIX_W, height: int = THERMAL_MATRIX_H):
        self.width = width
        self.height = height
        self.tick = 0
        y, x = np.mgrid[0:height, 0:width]
        self.x = x
        self.y = y

    def next_frame(self) -> Tuple[np.ndarray, np.ndarray, list]:
        self.tick += 1
        t = self.tick / 8.0

        temp = 24.0 + 1.8 * np.sin(self.x / 7.5 + t)
        temp += 1.5 * np.cos(self.y / 6.0 + t * 0.8)
        temp += np.random.normal(0, 0.18, (self.height, self.width))

        hot_sources = [
            (22 + 8 * math.sin(t * 0.8), 18 + 4 * math.cos(t * 0.7), 84),
            (53 + 6 * math.cos(t * 0.6), 32 + 5 * math.sin(t * 0.9), 76),
            (40 + 4 * math.sin(t * 0.5), 47 + 3 * math.cos(t), 64),
        ]
        for cx, cy, peak in hot_sources:
            sigma = 4.2
            g = np.exp(-(((self.x - cx) ** 2 + (self.y - cy) ** 2) / (2 * sigma ** 2)))
            temp += g * (peak - 24.0)

        rgb = temperature_to_rgb(temp)
        return temp, rgb, []


class ThermalCameraManager:
    """统一热像仪管理器 — 封装 WiFi / Lepton / Lepton远程 / 模拟器四种模式。

    用法:
        mgr = ThermalCameraManager(config)
        mgr.connect_wifi()           # 或 connect_lepton("COM4") 或 connect_lepton_remote(host, port)
        frame = mgr.get_frame()      # ThermalFrame 或 None
        mgr.disconnect_all()
    """

    def __init__(self, config: dict = None):
        cfg = config or {}
        self._mode = cfg.get("default_device", "simulator")

        self._wifi = ThermalWiFiClient(
            host=cfg.get("wifi_host", "192.168.3.166"),
            port=cfg.get("wifi_port", 5001),
        )
        self._lepton = ThermalLeptonReader(
            baudrate=cfg.get("lepton_baud", 921600),
        )
        self._simulator = ThermalSimulator()

        self._last_frame: Optional[ThermalFrame] = None

    # ---------- 连接 ----------

    def connect_wifi(self) -> bool:
        self._mode = "wifi"
        return self._wifi.connect()

    def connect_lepton(self, port: str) -> bool:
        self._mode = "lepton"
        return self._lepton.connect(port)

    def connect_lepton_remote(self, host: str, port: int = 6002) -> bool:
        """通过远程中继连接 Lepton 热像仪。"""
        self._mode = "lepton_remote"
        return self._lepton.open_remote(host, port)

    def set_mode(self, mode: str):
        """切换模式: "wifi" / "lepton" / "lepton_remote" / "simulator"。"""
        self._mode = mode

    def disconnect_all(self):
        self._wifi.disconnect()
        self._lepton.disconnect()

    # ---------- 状态 ----------

    def is_connected(self) -> bool:
        if self._mode == "wifi":
            return self._wifi.is_connected()
        if self._mode in ("lepton", "lepton_remote"):
            return self._lepton.is_connected()
        return True  # simulator always "connected"

    def get_mode(self) -> str:
        return self._mode

    def get_state_text(self) -> str:
        if self._mode == "wifi":
            return "WiFi 已连接" if self._wifi.is_connected() else "WiFi 未连接"
        if self._mode == "lepton":
            return "Lepton 已连接" if self._lepton.is_connected() else "Lepton 未连接"
        if self._mode == "lepton_remote":
            if self._lepton.is_connected():
                info = self._lepton.get_connection_info()
                return f"远程 Lepton {info.get('host', '?')}:{info.get('port', '?')}"
            return "远程 Lepton 未连接"
        return "模拟数据"

    # ---------- 帧获取 ----------

    def get_frame(self) -> Optional[ThermalFrame]:
        """获取最新一帧热成像数据。无数据时返回 None。"""
        raw = None

        if self._mode == "wifi":
            raw = self._wifi.get_latest()
        elif self._mode in ("lepton", "lepton_remote"):
            raw = self._lepton.get_latest()
        else:
            # simulator
            temps, rgb, _ = self._simulator.next_frame()
            raw = {
                'temps': temps, 'rgb': rgb,
                'width': self._simulator.width, 'height': self._simulator.height,
                'maxTemp': float(np.max(temps)), 'minTemp': float(np.min(temps)),
                'avgTemp': float(np.mean(temps)),
                'centerTemp': float(temps[self._simulator.height // 2, self._simulator.width // 2]),
                'maxPos': np.unravel_index(np.argmax(temps), temps.shape),
            }

        if raw is None:
            return None

        max_pos_raw = raw.get('maxPos', (0, 0))
        if isinstance(max_pos_raw, dict):
            max_pos = (max_pos_raw['y'], max_pos_raw['x'])
        else:
            max_pos = tuple(max_pos_raw)

        frame = ThermalFrame(
            temps=raw['temps'],
            rgb=raw['rgb'],
            width=raw['width'],
            height=raw['height'],
            max_temp=raw['maxTemp'],
            min_temp=raw['minTemp'],
            avg_temp=raw['avgTemp'],
            center_temp=raw['centerTemp'],
            max_pos=max_pos,
        )
        self._last_frame = frame
        return frame

    def get_last_frame(self) -> Optional[ThermalFrame]:
        return self._last_frame


# ════════════════════════════════════════════════════════════════
# §4 MCUSerialBridge — MCU 串口桥接
# ════════════════════════════════════════════════════════════════

class MCUSerialBridge:
    """MCU 串口管理器 — 传感器数据接收 + 报警指令下发。

    支持两种连接模式:
        1. 本地串口 (local): 直连 MCU
        2. 远程 TCP (remote): 通过工控机中继服务连接 MCU

    协议:
        MCU → 上位机: $DATA,<temp>,<humi>,<ultrasonic>,<smoke>,<light>\\r\\n
        上位机 → MCU: $ALARM,<temp>\\r\\n / $NORMAL,<temp>\\r\\n
        MCU 应答:     $ACK,<STATE>\\r\\n

    远程中继控制帧:
        PC → 工控机:   $PING\\r\\n / $CMD,LIST_PORTS\\r\\n / $CMD,OPEN,<port>\\r\\n
        工控机 → PC:   $PONG\\r\\n / $PORTS,...\\r\\n / $CMD_OK,...\\r\\n / $CMD_ERR,...\\r\\n / $STATUS,...\\r\\n
    """

    # 心跳间隔 (秒)
    PING_INTERVAL = 10
    # 远程重连间隔 (秒)
    REMOTE_RECONNECT_INTERVAL = 3

    def __init__(self, port: str = None, baudrate: int = 115200):
        self.port_name = port
        self.baudrate = baudrate
        self._port = None
        self._connected = False
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._line_buffer = ""
        self._sensor_data = SensorData()

        # 远程 TCP 模式
        self._mode: Optional[str] = None          # 'local' / 'remote' / None
        self._tcp_socket: Optional[socket.socket] = None
        self._remote_host: Optional[str] = None
        self._remote_port: int = 6001
        self._ping_timer: Optional[threading.Timer] = None
        self._reconnect_timer: Optional[threading.Timer] = None
        self._remote_ports: List[str] = []
        self._remote_cmd_response: Optional[str] = None

    # ════════════════════════════════════════════════════════
    # 连接模式: 本地串口
    # ════════════════════════════════════════════════════════

    def connect(self, port: str = None) -> bool:
        """连接本地串口。"""
        if port:
            self.port_name = port
        if not self.port_name:
            print("[MCU] 未指定串口")
            return False
        if serial is None:
            print("[MCU] pyserial 未安装")
            return False

        # 先清理远程连接
        self._force_cleanup_tcp()

        with self._lock:
            self._disconnect_internal()
            try:
                self._port = serial.Serial(
                    port=self.port_name,
                    baudrate=self.baudrate,
                    bytesize=8, parity='N', stopbits=1,
                    timeout=0.1,
                )
                self._connected = True
                self._running = True
                self._mode = 'local'
                self._line_buffer = ""
                print(f"[MCU] 已打开本地串口 {self.port_name} @ {self.baudrate}")

                self._thread = threading.Thread(target=self._read_loop, daemon=True)
                self._thread.start()
                return True
            except Exception as e:
                print(f"[MCU] 连接失败: {e}")
                self._connected = False
                return False

    def _disconnect_internal(self):
        self._running = False
        if self._port and self._port.is_open:
            try:
                self._port.close()
            except Exception:
                pass
        self._port = None
        if self._mode == 'local':
            self._connected = False
            self._mode = None

    # ════════════════════════════════════════════════════════
    # 连接模式: 远程 TCP
    # ════════════════════════════════════════════════════════

    def open_remote(self, host: str, port: int = 6001) -> bool:
        """通过 TCP 连接工控机中继服务。

        Args:
            host: 工控机 IP 地址
            port: TCP 端口号 (默认 6001)

        Returns:
            是否连接成功
        """
        # 先清理旧连接 (本地串口或旧远程)
        self.disconnect()

        self._remote_host = host
        self._remote_port = port

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(5.0)
            sock.connect((host, port))
            sock.settimeout(None)  # 连接成功后取消超时, 由读线程控制
        except Exception as e:
            print(f"[MCU] 远程连接失败: {e}")
            return False

        self._tcp_socket = sock
        self._connected = True
        self._running = True
        self._mode = 'remote'
        self._line_buffer = ""
        print(f"[MCU] 已连接远程工控机 {host}:{port}")

        # 启动读取线程
        self._thread = threading.Thread(target=self._remote_read_loop, daemon=True)
        self._thread.start()

        # 启动心跳
        self._reset_ping_timer()

        return True

    def _remote_read_loop(self):
        """远程 TCP 读取线程。"""
        while self._running and self._tcp_socket:
            try:
                data = self._tcp_socket.recv(4096)
                if not data:
                    print("[MCU] 远程连接断开 (收到空数据)")
                    break
                self._line_buffer += data.decode('ascii', errors='ignore')
                while '\r\n' in self._line_buffer:
                    line, self._line_buffer = self._line_buffer.split('\r\n', 1)
                    self._process_line(line.strip())
            except OSError as e:
                if self._running:
                    print(f"[MCU] 远程读取错误: {e}")
                break
            except Exception as e:
                if self._running:
                    print(f"[MCU] 远程读取异常: {e}")
                break

        # 连接断开
        was_running = self._running
        self._connected = False
        self._mode = None
        self._clear_ping_timer()
        if self._tcp_socket:
            try:
                self._tcp_socket.close()
            except Exception:
                pass
            self._tcp_socket = None

        # 自动重连 (非主动断开时)
        if was_running and self._remote_host:
            print(f"[MCU] 将在 {self.REMOTE_RECONNECT_INTERVAL} 秒后重连...")
            self._reconnect_timer = threading.Timer(
                self.REMOTE_RECONNECT_INTERVAL,
                self._auto_reconnect_remote
            )
            self._reconnect_timer.daemon = True
            self._reconnect_timer.start()

    def _auto_reconnect_remote(self):
        """自动重连远程工控机。"""
        if self._remote_host and not self._connected:
            print(f"[MCU] 自动重连 {self._remote_host}:{self._remote_port}...")
            self.open_remote(self._remote_host, self._remote_port)

    def _force_cleanup_tcp(self):
        """强制清理远程 TCP 连接。"""
        self._clear_ping_timer()
        self._clear_reconnect_timer()
        if self._tcp_socket:
            try:
                self._tcp_socket.close()
            except Exception:
                pass
            self._tcp_socket = None
        if self._mode == 'remote':
            self._connected = False
            self._mode = None

    def _reset_ping_timer(self):
        """重置心跳定时器。"""
        self._clear_ping_timer()
        self._ping_timer = threading.Timer(self.PING_INTERVAL, self._send_ping)
        self._ping_timer.daemon = True
        self._ping_timer.start()

    def _clear_ping_timer(self):
        if self._ping_timer:
            self._ping_timer.cancel()
            self._ping_timer = None

    def _clear_reconnect_timer(self):
        if self._reconnect_timer:
            self._reconnect_timer.cancel()
            self._reconnect_timer = None

    def _send_ping(self):
        """发送心跳包。"""
        if self._mode == 'remote' and self._connected:
            self._write_raw('$PING\r\n')
            self._reset_ping_timer()

    # ════════════════════════════════════════════════════════
    # 远程控制命令
    # ════════════════════════════════════════════════════════

    def list_remote_ports(self) -> List[str]:
        """查询工控机上的串口列表。

        Returns:
            串口名列表, 如 ['COM1', 'COM3']
        """
        if self._mode != 'remote' or not self._connected:
            return []
        self._remote_cmd_response = None
        self._remote_ports = []
        self._write_raw('$CMD,LIST_PORTS\r\n')
        # 等待响应 (最多 2 秒)
        for _ in range(20):
            time.sleep(0.1)
            if self._remote_cmd_response is not None:
                break
        return list(self._remote_ports)

    def open_remote_port(self, port_name: str) -> bool:
        """请求工控机切换串口。

        Args:
            port_name: 串口名, 如 'COM3'

        Returns:
            是否切换成功
        """
        if self._mode != 'remote' or not self._connected:
            return False
        self._remote_cmd_response = None
        self._write_raw(f'$CMD,OPEN,{port_name}\r\n')
        # 等待响应 (最多 3 秒)
        for _ in range(30):
            time.sleep(0.1)
            if self._remote_cmd_response is not None:
                return self._remote_cmd_response == 'CMD_OK'
        return False

    # ════════════════════════════════════════════════════════
    # 通用断开/状态
    # ════════════════════════════════════════════════════════

    def disconnect(self):
        """断开所有连接 (本地串口和远程 TCP)。"""
        self._force_cleanup_tcp()
        with self._lock:
            self._disconnect_internal()

    def is_connected(self) -> bool:
        if self._mode == 'local':
            return self._connected and self._port is not None and self._port.is_open
        if self._mode == 'remote':
            return self._connected and self._tcp_socket is not None
        return False

    def get_mode(self) -> Optional[str]:
        """返回当前连接模式: 'local' / 'remote' / None。"""
        return self._mode

    def get_connection_info(self) -> dict:
        """返回连接信息字典。"""
        if self._mode == 'local':
            return {'mode': 'local', 'port': self.port_name, 'baudrate': self.baudrate}
        elif self._mode == 'remote':
            return {'mode': 'remote', 'host': self._remote_host, 'port': self._remote_port}
        return {'mode': None}

    # ════════════════════════════════════════════════════════
    # 数据读取 (本地串口)
    # ════════════════════════════════════════════════════════

    def _read_loop(self):
        """本地串口读取线程。"""
        while self._running:
            try:
                if not self._port or not self._port.is_open:
                    break
                data = self._port.read(512)
                if data:
                    self._line_buffer += data.decode('ascii', errors='ignore')
                    while '\r\n' in self._line_buffer:
                        line, self._line_buffer = self._line_buffer.split('\r\n', 1)
                        self._process_line(line.strip())
            except Exception as e:
                if self._running:
                    print(f"[MCU] 读取错误: {e}")
                break
        self._connected = False
        self._mode = None

    # ════════════════════════════════════════════════════════
    # 协议解析 (本地和远程共用)
    # ════════════════════════════════════════════════════════

    def _process_line(self, line: str):
        """解析 MCU 数据帧和中继控制帧。"""
        if not line:
            return

        # --- 中继控制帧 ---
        if line == '$PONG':
            # 心跳回复, 无需处理
            return

        if line.startswith('$PORTS,'):
            # 串口列表响应
            ports_str = line[7:]
            self._remote_ports = [p.strip() for p in ports_str.split(',') if p.strip()]
            self._remote_cmd_response = 'PORTS'
            print(f"[MCU] 工控机串口列表: {self._remote_ports}")
            return

        if line == '$CMD_OK,OPEN':
            self._remote_cmd_response = 'CMD_OK'
            print("[MCU] 工控机串口切换成功")
            return

        if line.startswith('$CMD_ERR,OPEN'):
            self._remote_cmd_response = 'CMD_ERR'
            reason = line[14:] if len(line) > 14 else 'unknown'
            print(f"[MCU] 工控机串口切换失败: {reason}")
            return

        if line.startswith('$STATUS,'):
            status = line[8:]
            if status == 'CONNECTED':
                print("[MCU] 工控机报告: MCU 串口已连接")
            elif status == 'DISCONNECTED':
                print("[MCU] 工控机报告: MCU 串口已断开")
            return

        # --- MCU 数据帧 ---
        if line.startswith('$ACK,'):
            ack_type = line[5:]
            print(f"[MCU] ACK: {ack_type}")
        elif line.startswith('$DATA,'):
            try:
                parts = line.split(',')
                if len(parts) == 6:
                    def _sf(v, d=0.0):
                        try:
                            return float(v)
                        except (ValueError, TypeError):
                            return d

                    def _si(v, d=0):
                        try:
                            return int(float(v))
                        except (ValueError, TypeError):
                            return d

                    self._sensor_data = SensorData(
                        dht11_temp=_sf(parts[1]),
                        dht11_humi=_sf(parts[2]),
                        ultrasonic=_si(parts[3]),
                        smoke=_si(parts[4]),
                        light=_si(parts[5]),
                        valid=True,
                        timestamp=time.time(),
                    )
            except Exception as e:
                print(f"[MCU] 解析传感器数据失败: {e}")

    # ════════════════════════════════════════════════════════
    # 发送
    # ════════════════════════════════════════════════════════

    def send_alarm(self, temp: float) -> bool:
        """发送超温报警指令。"""
        return self._send_line(f"$ALARM,{temp:.1f}")

    def send_normal(self, temp: float) -> bool:
        """发送恢复正常指令。"""
        return self._send_line(f"$NORMAL,{temp:.1f}")

    def _send_line(self, line: str) -> bool:
        """发送一行数据, 根据模式选择输出。"""
        return self._write_raw(line + '\r\n')

    def _write_raw(self, data: str) -> bool:
        """底层写入, 根据模式选择输出到串口或 TCP。"""
        encoded = data.encode('ascii')
        if self._mode == 'local':
            with self._lock:
                if not self._port or not self._port.is_open:
                    return False
                try:
                    self._port.write(encoded)
                    return True
                except Exception as e:
                    print(f"[MCU] 本地发送失败: {e}")
                    return False
        elif self._mode == 'remote':
            if not self._tcp_socket:
                return False
            try:
                self._tcp_socket.sendall(encoded)
                return True
            except Exception as e:
                print(f"[MCU] 远程发送失败: {e}")
                return False
        return False

    # ---------- 读取 ----------

    def get_sensor_data(self) -> SensorData:
        return self._sensor_data


def list_serial_ports() -> List[Tuple[str, str]]:
    """列出系统可用串口。"""
    if serial is None:
        return []
    return [(p.device, p.description) for p in serial.tools.list_ports.comports()]


# ════════════════════════════════════════════════════════════════
# §5 AlarmController — 报警控制器
# ════════════════════════════════════════════════════════════════

class AlarmController:
    """超温报警状态机。

    状态:
        NORMAL     → max > 阈值 → ALARM
        ALARM      → max <= 阈值 → NORMAL
        ALARM      → 手动取消 → SUPPRESSED (3秒)
        SUPPRESSED → 超时 → NORMAL

    联动:
        进入 ALARM 时自动发送 $ALARM
        离开 ALARM 时自动发送 $NORMAL
    """

    def __init__(self, threshold: float = 70.0, mcu_bridge: MCUSerialBridge = None):
        self._threshold = threshold
        self._mcu = mcu_bridge
        self._active = False
        self._suppressed = False
        self._suppress_timer: Optional[threading.Timer] = None
        self._state = AlarmState(threshold=threshold)

    def set_threshold(self, threshold: float):
        self._threshold = threshold
        self._state.threshold = threshold

    def check(self, max_temp: float) -> AlarmState:
        """每帧调用, 传入最高温度, 返回报警状态。"""
        if self._suppressed:
            self._state.max_temp = max_temp
            return self._state

        is_alarm = max_temp > self._threshold

        if is_alarm and not self._active:
            self._active = True
            self._state.active = True
            self._state.level = "alarm"
            self._state.color = COLOR_RED
            self._state.text = f"!! 报警 {max_temp:.1f}℃"
            self._state.max_temp = max_temp
            if self._mcu and self._mcu.is_connected():
                self._mcu.send_alarm(max_temp)
                print(f"[Alarm] 触发报警: {max_temp:.1f}℃ > {self._threshold}℃")

        elif not is_alarm and self._active:
            self._active = False
            self._state.active = False
            self._state.level = "normal"
            self._state.color = COLOR_GREEN
            self._state.text = "正常"
            self._state.max_temp = max_temp
            if self._mcu and self._mcu.is_connected():
                self._mcu.send_normal(max_temp)
                print(f"[Alarm] 恢复正常: {max_temp:.1f}℃")

        elif is_alarm and self._active:
            self._state.max_temp = max_temp
            self._state.text = f"!! 报警 {max_temp:.1f}℃"

        else:
            self._state.max_temp = max_temp

        return self._state

    def cancel(self):
        """手动取消报警, 抑制 3 秒。"""
        self._active = False
        self._suppressed = True
        self._state.active = False
        self._state.level = "suppressed"
        self._state.color = COLOR_YELLOW
        self._state.text = "已取消"

        if self._mcu and self._mcu.is_connected():
            self._mcu.send_normal(self._threshold)
            print("[Alarm] 手动取消报警, 下发 $NORMAL")

        if self._suppress_timer:
            self._suppress_timer.cancel()
        self._suppress_timer = threading.Timer(3.0, self._lift_suppression)
        self._suppress_timer.start()

    def _lift_suppression(self):
        self._suppressed = False
        self._state.level = "normal"
        self._state.color = COLOR_GREEN
        self._state.text = "正常"

    def is_active(self) -> bool:
        return self._active

    def get_state(self) -> AlarmState:
        return self._state

    def get_threshold(self) -> float:
        return self._threshold


# ════════════════════════════════════════════════════════════════
# §6 ThermalZoneTracker — 热区追踪器
# ════════════════════════════════════════════════════════════════

class ThermalZoneTracker:
    """高温区域追踪器。

    功能:
        1. 每帧分析温度矩阵, 提取高温区域
        2. 跨帧追踪: 基于距离匹配, 分配稳定 region_id
        3. 轨迹记录: 记录每个热区的移动历史
        4. 叠加绘制: 在伪彩图上绘制热区标记框 + 十字准星 + 温度标注
    """

    def __init__(self, threshold: float = HIGH_TEMP_LIMIT,
                 max_regions: int = 6, merge_distance: int = 7,
                 track_history: int = 30):
        self._threshold = threshold
        self._max_regions = max_regions
        self._merge_distance = merge_distance
        self._track_history = track_history

        self._regions: List[HotRegion] = []
        self._tracks: Dict[int, List[Tuple[int, int]]] = {}  # region_id → [(x,y), ...]
        self._next_id = 0
        self._table_data: List[dict] = []
        self._matrix_w = THERMAL_MATRIX_W  # 由 update() 每帧更新
        self._matrix_h = THERMAL_MATRIX_H

    # ---------- 参数设置 (右键菜单调用) ----------

    def set_threshold(self, threshold: float):
        """设置高温检测阈值 (℃)。"""
        self._threshold = threshold

    def set_max_regions(self, max_regions: int):
        """设置最大追踪区域数。"""
        self._max_regions = max(1, min(20, max_regions))

    def set_merge_distance(self, merge_distance: int):
        """设置区域合并距离 (像素)。"""
        self._merge_distance = max(3, min(30, merge_distance))

    def set_track_history(self, track_history: int):
        """设置轨迹历史长度。"""
        self._track_history = max(5, min(100, track_history))

    def get_params(self) -> dict:
        """返回当前参数字典。"""
        return {
            'threshold': self._threshold,
            'max_regions': self._max_regions,
            'merge_distance': self._merge_distance,
            'track_history': self._track_history,
        }

    def update(self, temps: np.ndarray) -> List[HotRegion]:
        """每帧调用, 分析温度矩阵, 返回高温区域列表。"""
        # 记录实际矩阵尺寸 (用于 draw_overlay 坐标映射)
        self._matrix_h, self._matrix_w = temps.shape[:2]
        new_regions = self._extract_regions(temps)
        self._match_and_track(new_regions)
        self._update_table()
        return list(self._regions)

    def _extract_regions(self, temps: np.ndarray) -> List[HotRegion]:
        mask = temps >= self._threshold
        if not np.any(mask):
            return []

        ys, xs = np.where(mask)
        values = temps[ys, xs]
        order = np.argsort(values)[::-1]
        regions = []
        for idx in order:
            x = int(xs[idx])
            y = int(ys[idx])
            value = float(values[idx])
            # 合并临近点
            if any(r.dist_sq(x, y) < self._merge_distance ** 2 for r in regions):
                continue
            level = "严重高温" if value >= 85 else "高温"
            regions.append(HotRegion(f"高温区域{len(regions) + 1}", x, y, value, level))
            if len(regions) >= self._max_regions:
                break
        return regions

    def _match_and_track(self, new_regions: List[HotRegion]):
        """跨帧匹配: 基于最近距离分配 region_id。"""
        if not new_regions:
            # 所有热区消失, 清除轨迹
            if not self._regions:
                return
            self._regions = []
            return

        if not self._regions:
            # 首次出现, 全部分配新 ID
            for r in new_regions:
                r.region_id = self._next_id
                self._next_id += 1
                self._tracks[r.region_id] = [(r.x, r.y)]
            self._regions = new_regions
            return

        # 贪心匹配: 对每个新区域找最近的旧区域
        used_old = set()
        matched = []
        for nr in new_regions:
            best_dist = float('inf')
            best_old = None
            for i, old in enumerate(self._regions):
                if i in used_old:
                    continue
                d = nr.dist_sq(old.x, old.y)
                if d < best_dist:
                    best_dist = d
                    best_old = i
            if best_old is not None and best_dist < (self._merge_distance * 3) ** 2:
                # 匹配成功, 继承 ID
                nr.region_id = self._regions[best_old].region_id
                used_old.add(best_old)
            else:
                # 新区域, 分配新 ID
                nr.region_id = self._next_id
                self._next_id += 1

            # 更新轨迹
            if nr.region_id not in self._tracks:
                self._tracks[nr.region_id] = []
            track = self._tracks[nr.region_id]
            track.append((nr.x, nr.y))
            if len(track) > self._track_history:
                track.pop(0)

            matched.append(nr)

        # 清除消失区域的轨迹 (防止内存泄漏)
        current_ids = {r.region_id for r in matched}
        disappeared = set(self._tracks.keys()) - current_ids
        for rid in disappeared:
            del self._tracks[rid]

        self._regions = matched

    def _update_table(self):
        now_str = time.strftime("%H:%M:%S")
        self._table_data = []
        for r in self._regions:
            self._table_data.append({
                'point': f"P{r.region_id + 1:02d}",
                'time': now_str,
                'result': f"{r.level} {r.temp:.1f}℃",
                'action': '查看',
            })

    # ---------- 查询接口 ----------

    def get_regions(self) -> List[HotRegion]:
        return list(self._regions)

    def get_tracks(self) -> Dict[int, List[Tuple[int, int]]]:
        return dict(self._tracks)

    def get_table_data(self) -> List[dict]:
        return list(self._table_data)

    def get_region_count(self) -> int:
        return len(self._regions)

    def get_hot_info_text(self) -> str:
        """生成高温信息文本 (用于 hot_info 控件)。"""
        if not self._regions:
            return "高温区域数量: 0\n高温点位位置: 无\n过高区域: 无"
        positions = ", ".join(f"({r.x},{r.y})" for r in self._regions[:3])
        levels = ", ".join(f"{r.level} {r.temp:.1f}℃" for r in self._regions[:3])
        return f"高温区域: {len(self._regions)}\n位置: {positions}\n{levels}"

    # ---------- 叠加绘制 ----------

    def draw_overlay(self, rgb_image: np.ndarray) -> np.ndarray:
        """在伪彩图上叠加热区标记框 + 十字准星 + 温度标注。

        如果 OpenCV 不可用, 返回原图。
        """
        if cv2 is None or not self._regions:
            return rgb_image

        annotated = rgb_image.copy()
        h, w = annotated.shape[:2]

        # 使用实际矩阵尺寸做坐标映射 (兼容 80x60 / 160x120 等不同分辨率)
        mat_w = getattr(self, '_matrix_w', THERMAL_MATRIX_W)
        mat_h = getattr(self, '_matrix_h', THERMAL_MATRIX_H)

        for r in self._regions:
            # 将热像仪坐标映射到显示坐标
            sx = int(r.x * w / mat_w)
            sy = int(r.y * h / mat_h)

            # 颜色 (BGR 格式, OpenCV 绘图用)
            if r.temp >= 85:
                color = (0, 0, 255)     # 红色
            elif r.temp >= 70:
                color = (0, 100, 255)   # 橙色
            else:
                color = (0, 255, 255)   # 黄色

            # 标记框
            box_size = max(20, min(w, h) // 8)
            x1 = max(0, sx - box_size)
            y1 = max(0, sy - box_size)
            x2 = min(w - 1, sx + box_size)
            y2 = min(h - 1, sy + box_size)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # 十字准星
            cross_len = box_size // 2
            cv2.line(annotated, (sx - cross_len, sy), (sx + cross_len, sy), color, 1)
            cv2.line(annotated, (sx, sy - cross_len), (sx, sy + cross_len), color, 1)

            # 温度标注
            label = f"{r.temp:.1f}C"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            label_x = min(w - tw - 4, max(4, sx + box_size + 4))
            label_y = max(th + baseline + 4, sy - box_size)
            cv2.rectangle(annotated, (label_x - 2, label_y - th - baseline - 2),
                          (label_x + tw + 2, label_y + 2), color, -1)
            cv2.putText(annotated, label, (label_x, label_y - 2),
                        font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

            # 轨迹线
            track = self._tracks.get(r.region_id, [])
            if len(track) >= 2:
                for i in range(1, len(track)):
                    px0 = int(track[i - 1][0] * w / mat_w)
                    py0 = int(track[i - 1][1] * h / mat_h)
                    px1 = int(track[i][0] * w / mat_w)
                    py1 = int(track[i][1] * h / mat_h)
                    alpha = int(80 + 175 * i / len(track))
                    cv2.line(annotated, (px0, py0), (px1, py1), color, 1)

        return annotated


# ════════════════════════════════════════════════════════════════
# §7 串口工具
# ════════════════════════════════════════════════════════════════

class Stm32SerialController:
    """STM32 机器人控制串口。

    指令示例:
        前进: F:1,S:0
        后退: F:0,S:1
        停止: F:0,S:0
    """

    def __init__(self, port: str = "COM3", baudrate: int = 115200):
        self.port_name = port
        self.baudrate = baudrate
        self._port = None

    def connect(self, port: str = None) -> bool:
        if port:
            self.port_name = port
        if serial is None:
            print("[STM32] pyserial 未安装")
            return False
        try:
            self._port = serial.Serial(
                port=self.port_name,
                baudrate=self.baudrate,
                bytesize=8, parity='N', stopbits=1,
                timeout=0.1,
            )
            print(f"[STM32] 已打开 {self.port_name}")
            return True
        except Exception as e:
            print(f"[STM32] 连接失败: {e}")
            return False

    def disconnect(self):
        if self._port and self._port.is_open:
            try:
                self._port.close()
            except Exception:
                pass
        self._port = None

    def is_connected(self) -> bool:
        return self._port is not None and self._port.is_open

    def send_command(self, command: str):
        print(f"[STM32] 指令: {command}")
        if self._port is not None and self._port.is_open:
            try:
                self._port.write((command + "\r\n").encode("ascii"))
            except Exception as e:
                print(f"[STM32] 发送失败: {e}")


# ════════════════════════════════════════════════════════════════
# 模块自测
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== hat_extensions.py 自测 ===")

    # 测试模拟器
    sim = ThermalSimulator()
    temps, rgb, _ = sim.next_frame()
    print(f"模拟器: temps={temps.shape}, rgb={rgb.shape}, "
          f"max={np.max(temps):.1f}, min={np.min(temps):.1f}")

    # 测试热区追踪
    tracker = ThermalZoneTracker(threshold=70.0)
    regions = tracker.update(temps)
    print(f"热区追踪: {len(regions)} 个高温区域")
    for r in regions:
        print(f"  {r.name} ({r.x},{r.y}) {r.temp:.1f}℃ [{r.level}] id={r.region_id}")

    # 测试报警控制器
    alarm = AlarmController(threshold=70.0)
    state = alarm.check(float(np.max(temps)))
    print(f"报警状态: active={state.active}, level={state.level}, text={state.text}")

    # 测试 ThermalCameraManager 模拟模式
    mgr = ThermalCameraManager({"default_device": "simulator"})
    frame = mgr.get_frame()
    print(f"CameraManager: mode={mgr.get_mode()}, state={mgr.get_state_text()}")
    if frame:
        print(f"  frame: {frame.width}x{frame.height}, max={frame.max_temp:.1f}")

    # 测试叠加绘制
    if cv2 is not None and regions:
        overlay = tracker.draw_overlay(rgb)
        print(f"叠加绘制: {overlay.shape}")
    else:
        print("叠加绘制: 跳过 (OpenCV 未安装或无高温区域)")

    print("=== 自测完成 ===")
