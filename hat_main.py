# -*- coding: utf-8 -*-
"""
智能巡检-综合管理平台

安装依赖:
    pip install pyside6 opencv-python pyqtgraph pyserial

如果使用 PyQt5:
    pip install pyqt5 opencv-python pyqtgraph pyserial

运行:
    python main.py

说明:
    1. 本文件为单文件 Qt 上位机界面, 不依赖 Qt Designer。
    2. 优先使用 PySide6, 如果没有 PySide6 会自动尝试 PyQt5。
    3. 可见光相机使用 OpenCV 读取 Windows 系统摄像头/USB 摄像头。
"""

import math
import queue
import random
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ════════════════════════════════════════════════════════════════
# [扩展接口] 导入扩展模块 (热像仪/MCU/报警/热区追踪)
# ════════════════════════════════════════════════════════════════
try:
    from hat_extensions import (
        ThermalCameraManager,
        MCUSerialBridge,
        AlarmController,
        ThermalZoneTracker,
        Stm32SerialController as ExtStm32Controller,
    )
    EXTENSIONS_AVAILABLE = True
    print("[Extensions] hat_extensions.py 已加载")
except ImportError as _ext_err:
    EXTENSIONS_AVAILABLE = False
    print(f"[Extensions] hat_extensions.py 未找到, 使用内置模拟: {_ext_err}")




# ----------------------------- Qt 兼容导入 -----------------------------
try:
    from PySide6.QtCore import QPointF, QRectF, Qt, QThread, QTimer, QUrl, Signal
    from PySide6.QtGui import QColor, QFont, QImage, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap
    try:
        from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
    except ImportError:
        QAudioOutput = None
        QMediaPlayer = None
    from PySide6.QtWidgets import (
        QApplication,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QDoubleSpinBox,
        QFormLayout,
        QFrame,
        QGridLayout,
        QHBoxLayout,
        QHeaderView,
        QInputDialog,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMenu,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

    QT_LIB = "PySide6"
except ImportError:
    try:
        from PyQt5.QtCore import QPointF, QRectF, Qt, QThread, QTimer, QUrl, pyqtSignal as Signal
        from PyQt5.QtGui import QColor, QFont, QImage, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap
        try:
            from PyQt5.QtMultimedia import QAudioOutput, QMediaPlayer, QMediaContent
        except ImportError:
            QAudioOutput = None
            QMediaPlayer = None
            QMediaContent = None
        from PyQt5.QtWidgets import (
            QApplication,
            QComboBox,
            QDialog,
            QDialogButtonBox,
            QDoubleSpinBox,
            QFormLayout,
            QFrame,
            QGridLayout,
            QHBoxLayout,
            QHeaderView,
            QInputDialog,
            QLabel,
            QLineEdit,
            QMainWindow,
            QMenu,
            QMessageBox,
            QPushButton,
            QSpinBox,
            QTableWidget,
            QTableWidgetItem,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )

        QT_LIB = "PyQt5"
    except ImportError:
        print("未安装 PySide6 或 PyQt5, 请先执行:")
        print("pip install pyside6 opencv-python pyqtgraph pyserial")
        raise


# OpenCV 用于读取 USB 摄像头。没有安装时, 程序仍可运行。
try:
    import cv2
except ImportError:
    cv2 = None


# PySerial 预留给 STM32 和热成像串口。没有安装时, 只是不启用串口。
try:
    import serial  # noqa: F401
    import serial.tools.list_ports
except Exception:
    serial = None


def _list_serial_ports():
    """列出系统可用串口 (不依赖扩展模块)。"""
    if serial is None:
        return []
    return [(p.device, p.description) for p in serial.tools.list_ports.comports()]


# ----------------------------- 基础配置 -----------------------------
# 从 config.json 读取配置
_CONFIG_LOADED = {}
try:
    _config_path = Path(__file__).parent / "config.json"
    if _config_path.exists():
        import json as _json
        with open(_config_path, 'r', encoding='utf-8') as _f:
            _CONFIG_LOADED = _json.load(_f)
except Exception:
    pass

def _get_config(network_key, default):
    """从 config.json 读取配置"""
    keys = network_key.split('.')
    val = _CONFIG_LOADED
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k, None)
        else:
            return default
    return val if val is not None else default

NETWORK_CAMERA_URL = _get_config('camera.network_url', "http://192.168.124.73:5000/video")
LOCAL_CAMERA_INDEX = 0
VISIBLE_CAMERA_INDEX = NETWORK_CAMERA_URL
VISIBLE_CAMERA_WIDTH = 640
VISIBLE_CAMERA_HEIGHT = 480
CAMERA_REFRESH_MS = 25  #刷新等待时间
AUX_REFRESH_MS = 1000
ENABLE_YOLO_DETECTION = True
YOLO_DETECT_INTERVAL = 1
NETWORK_FALLBACK_FAILURES = 3
NETWORK_OPEN_TIMEOUT_MS = 1500
NETWORK_READ_TIMEOUT_MS = 1500
YOLO_MODEL_PATH = Path(__file__).resolve().parent / "best2.pt"
YOLO_CONFIDENCE = 0.55
YOLO_IMAGE_SIZE = 640
YOLO_DEVICE = "auto"
YOLO_HALF = True
YOLO_BOX_HOLD_FRAMES = 8
NO_HAT_ALERT_SECONDS = 2.0
NO_HAT_LABEL_KEYWORDS = ("no_hat", "nohat", "no-helmet", "no_helmet", "without_hat", "head", "未戴")
HAT_LABEL_KEYWORDS = ("hat", "helmet", "hardhat", "safe_hat", "safety_hat", "戴帽")
PERSON_LABEL_KEYWORDS = ("person", "people", "worker", "human", "人")
NO_HAT_LOST_GRACE_SECONDS = 0.5
THERMAL_MATRIX_W = 80
THERMAL_MATRIX_H = 60
HIGH_TEMP_LIMIT = 70.0
TEMP_WARN_LEVEL_1 = 40.0   # 低于此值: 绿色(正常)
TEMP_WARN_LEVEL_2 = 70.0   # 40~70: 黄色闪烁(二级预警); >=70: 红色闪烁(三级报警)

BG = "#020817"
PANEL_BG = "#061a32"
PANEL_BG_2 = "#082541"
TITLE_BLUE = "#0b76bd"
TITLE_DARK = "#06375f"
LINE = "#14a9ff"
TEXT = "#dff7ff"
MUTED = "#82b8d8"
CYAN = "#25eaff"
BLUE = "#2b91ff"
GREEN = "#20e986"
YELLOW = "#ffdf4d"
RED = "#ff3038"
ORANGE = "#ff8b25"


@dataclass
class HotRegion:
    """热成像高温区域结构。真实热像仪接入后也可复用。"""

    name: str
    x: int
    y: int
    temp: float
    level: str


def align_center():
    return Qt.AlignCenter


def qimage_rgb888():
    try:
        return QImage.Format_RGB888
    except AttributeError:
        return QImage.Format.Format_RGB888


def run_qt_app(app):
    """兼容 PySide6 的 exec() 与 PyQt5 的 exec_()。"""

    if hasattr(app, "exec"):
        return app.exec()
    return app.exec_()


def camera_display_name(index):
    if isinstance(index, str):
        if index == NETWORK_CAMERA_URL:
            return "网络摄像头"
        return "网络摄像头"
    return f"本地摄像头 {index}"


# ----------------------------- 后续扩展接口 -----------------------------
def local_camera_backends():
    if cv2 is None:
        return []
    backends = []
    for backend_name in ("CAP_DSHOW", "CAP_MSMF", "CAP_ANY"):
        if hasattr(cv2, backend_name):
            backends.append(getattr(cv2, backend_name))
    return backends


def open_local_camera_capture(index):
    if cv2 is None:
        return None
    for backend in local_camera_backends():
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(index)


def discover_local_camera_indices(limit=6):
    if cv2 is None:
        return []
    indices = []
    for index in range(limit):
        cap = open_local_camera_capture(index)
        if cap is None:
            continue
        ok = cap.isOpened()
        frame_ok = False
        if ok:
            ret, frame = cap.read()
            frame_ok = bool(ret and frame is not None)
        cap.release()
        if ok and frame_ok:
            indices.append(index)
    return indices


def resolve_yolo_device():
    if YOLO_DEVICE != "auto":
        return YOLO_DEVICE
    try:
        import torch

        if torch.cuda.is_available():
            return 0
    except Exception:
        pass
    return "cpu"


def detection_class_name(detection):
    if len(detection) >= 6:
        return str(detection[5]).lower()
    if len(detection) >= 5:
        label = str(detection[4])
        return label.rsplit(" ", 1)[0].lower()
    return ""


def has_label_keyword(detection, keywords):
    name = detection_class_name(detection)
    return any(keyword.lower() in name for keyword in keywords)


def is_no_hat_detection(detection):
    name = detection_class_name(detection)
    if any(keyword.lower() in name for keyword in NO_HAT_LABEL_KEYWORDS):
        return True
    has_hat_word = any(keyword.lower() in name for keyword in HAT_LABEL_KEYWORDS)
    negative_word = any(keyword in name for keyword in ("no", "none", "without", "not", "未", "无", "沒", "没"))
    return has_hat_word and negative_word


def temperature_alarm_state(temp, warn_threshold=None, alarm_threshold=None):
    warn = warn_threshold if warn_threshold is not None else TEMP_WARN_LEVEL_1
    alarm = alarm_threshold if alarm_threshold is not None else TEMP_WARN_LEVEL_2
    if temp >= alarm:
        return "三级报警", RED
    if temp >= warn:
        return "二级预警", YELLOW
    return "正常", GREEN


class YoloDetector:
    """YOLOv8 检测接口预留。

    后续可以在这里加载 YOLOv8 模型, 完成:
        - 火焰检测
        - 烟雾检测
        - 安全帽/静电帽检测
    """

    def __init__(self, model_path=YOLO_MODEL_PATH, confidence=YOLO_CONFIDENCE, imgsz=YOLO_IMAGE_SIZE):
        self.enabled = False
        self.model = None
        self.model_path = Path(model_path)
        self.confidence = confidence
        self.imgsz = imgsz
        self.device = resolve_yolo_device()
        self.half = bool(YOLO_HALF and self.device != "cpu")
        self.error = ""
        self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            self.error = f"YOLO model not found: {self.model_path}"
            print(self.error)
            return
        try:
            from ultralytics import YOLO

            self.model = YOLO(str(self.model_path))
            self.enabled = True
            self.error = ""
            print(f"YOLOv8 model loaded: {self.model_path}, device={self.device}, half={self.half}")
        except Exception as exc:
            self.enabled = False
            self.model = None
            self.error = f"YOLO load failed: {exc}"
            print(self.error)

    def detect(self, rgb_frame):
        """返回绘制检测框后的 RGB 图像和检测数量。"""

        if rgb_frame is None or not self.enabled or self.model is None or cv2 is None:
            return []

        try:
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            results = self.model.predict(
                source=bgr_frame,
                conf=self.confidence,
                imgsz=self.imgsz,
                device=self.device,
                half=self.half,
                verbose=False,
            )
        except Exception as exc:
            self.error = f"YOLO predict failed: {exc}"
            print(self.error)
            return []

        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        names = getattr(result, "names", {}) or {}
        if boxes is None:
            return []

        detections = []
        for box in boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy.tolist()
            conf = float(box.conf[0].detach().cpu())
            cls_id = int(box.cls[0].detach().cpu())
            class_name = str(names.get(cls_id, cls_id))
            label = f"{class_name} {conf:.2f}"
            detections.append((x1, y1, x2, y2, label, class_name, conf))

        return detections

    @staticmethod
    def draw_detections(frame, detections):
        if frame is None or not detections or cv2 is None:
            return frame
        annotated = frame.copy()
        for x1, y1, x2, y2, label, *_ in detections:
            YoloDetector._draw_box(annotated, x1, y1, x2, y2, label)
        return annotated

    @staticmethod
    def _draw_box(frame, x1, y1, x2, y2, label):
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        color = (37, 234, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 2)
        label_y = max(0, y1 - th - baseline - 6)
        cv2.rectangle(frame, (x1, label_y), (min(w - 1, x1 + tw + 8), label_y + th + baseline + 6), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 4, label_y + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (2, 8, 23),
            2,
            cv2.LINE_AA,
        )


class Stm32SerialController:
    """STM32 串口发送接口预留。

    指令示例:
        前进: F:1,S:0
        后退: F:0,S:1
        停止: F:0,S:0
    """

    def __init__(self, port="COM3", baudrate=115200):
        self.port_name = port
        self.baudrate = baudrate
        self.port = None

    def open(self):
        if serial is None:
            return False
        # 示例不主动打开串口, 避免没有硬件时报错。
        return False

    def send_command(self, command):
        print("STM32 指令预留:", command)
        if self.port is not None:
            self.port.write((command + "\r\n").encode("ascii"))


class ThermalSerialReader:
    """热成像串口接收接口预留。

    真实热像仪可在这里读取 80x60 温度数组, 返回 NumPy 矩阵。
    """

    def __init__(self, port="COM4", baudrate=921600):
        self.port_name = port
        self.baudrate = baudrate
        self.port = None

    def read_temperature_matrix(self):
        """当前返回 None, 表示继续使用模拟热成像数据。"""

        return None


# ----------------------------- 摄像头与热成像数据 -----------------------------
class VisibleCamera:
    """Windows 系统摄像头/USB 摄像头读取封装。"""

    def __init__(self, index=0, fallback_to_local=True):
        self.index = index
        self.cap = None
        self.fallback_to_local = fallback_to_local
        self.error = "OpenCV 未安装" if cv2 is None else ""
        self.open(index)

    def open(self, index=None):
        if index is not None:
            self.index = index
        self.release()
        if cv2 is None:
            self.error = "OpenCV 未安装, 无法读取摄像头"
            return False

        requested_index = self.index
        self.cap = self._open_capture(requested_index)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None

            if isinstance(requested_index, str) and self.fallback_to_local:
                fallback_index = LOCAL_CAMERA_INDEX
                self.cap = self._open_capture(fallback_index)
                if self.cap.isOpened():
                    self.index = fallback_index
                    self.error = f"网络流打开失败, 已切换到本地摄像头 {fallback_index}: {requested_index}"
                    print(self.error)
                    return self._finish_open()
                self.cap.release()
                self.cap = None
                self.error = f"网络流和本地摄像头均打开失败: {requested_index}, {fallback_index}"
                return False

            self.error = f"摄像头未连接、被占用或索引错误: {requested_index}"
            return False

        self.index = requested_index
        self.error = ""
        return self._finish_open()

    def _open_capture(self, index):
        # CAP_DSHOW 用于 Windows 系统摄像头, 对 USB 摄像头更稳定。
        if isinstance(index, str):
            params = []
            if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                params.extend([cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, NETWORK_OPEN_TIMEOUT_MS])
            if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                params.extend([cv2.CAP_PROP_READ_TIMEOUT_MSEC, NETWORK_READ_TIMEOUT_MS])
            if params and hasattr(cv2, "CAP_FFMPEG"):
                cap = cv2.VideoCapture(index, cv2.CAP_FFMPEG, params)
                if cap.isOpened():
                    return cap
                cap.release()
            return cv2.VideoCapture(index)
        return open_local_camera_capture(index)

    def _finish_open(self):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, VISIBLE_CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VISIBLE_CAMERA_HEIGHT)
        return True

    def read_rgb(self):
        if self.cap is None:
            return None
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.error = "摄像头读取失败"
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class YoloWorker(QThread):
    """Runs YOLO inference on the latest submitted frame without blocking camera reads."""

    def __init__(self, result_queue, parent=None):
        super().__init__(parent)
        self.result_queue = result_queue
        self.detector = None
        self._running = True
        self._latest_frame = None
        self._latest_frame_id = 0
        self._lock = threading.Lock()

    def submit_frame(self, frame, frame_id):
        if frame is None:
            return
        with self._lock:
            self._latest_frame = frame.copy()
            self._latest_frame_id = frame_id

    def stop(self):
        self._running = False
        if not self.wait(3000):
            self.terminate()
            self.wait(1000)

    def run(self):
        self.detector = YoloDetector()
        if not self.detector.enabled:
            self.result_queue.put(("status", self.detector.error))

        while self._running:
            frame, frame_id = self._take_latest_frame()
            if frame is None:
                self.msleep(5)
                continue

            if not self.detector.enabled:
                self.msleep(50)
                continue

            detections = self.detector.detect(frame)
            tooltip = str(self.detector.model_path) if not self.detector.error else self.detector.error
            self.result_queue.put(("detections", frame_id, detections, tooltip))

    def _take_latest_frame(self):
        with self._lock:
            frame = self._latest_frame
            frame_id = self._latest_frame_id
            self._latest_frame = None
            return frame, frame_id


class CameraWorker(QThread):
    """后台读取摄像头并执行 YOLO 检测, 避免阻塞 Qt 主线程。"""

    frame_ready = Signal(object, object, int, str, bool, object)
    error_ready = Signal(object, str)

    def __init__(self, index=VISIBLE_CAMERA_INDEX, parent=None):
        super().__init__(parent)
        self.index = index
        self.camera = None
        self.yolo_worker = None
        self.yolo_status = "YOLO 检测未启用"
        self._running = True
        self._pending_open = None
        self._open_generation = 0
        self._open_results = queue.Queue()
        self._yolo_results = queue.Queue()
        self._read_failures = 0
        self._frame_count = 0
        self._camera_generation = 0
        self._last_detections = []
        self._box_hold_frames = 0
        self._has_detection_result = False
        self._lock = threading.Lock()
        self._yolo_enabled = ENABLE_YOLO_DETECTION

    def request_open(self, index):
        with self._lock:
            self._open_generation += 1
            self._pending_open = (self._open_generation, index)

    def set_yolo_enabled(self, enabled):
        with self._lock:
            self._yolo_enabled = bool(enabled)
            if not self._yolo_enabled:
                self._last_detections = []
                self._box_hold_frames = 0
                self._has_detection_result = False
                self.yolo_status = "YOLO 检测已关闭"

    def is_yolo_enabled(self):
        with self._lock:
            return self._yolo_enabled

    def stop(self):
        self._running = False
        if self.yolo_worker is not None:
            self.yolo_worker.stop()
        if not self.wait(3000):
            self.terminate()
            self.wait(1000)

    def run(self):
        requested_start_index = self.index
        startup_index = LOCAL_CAMERA_INDEX if isinstance(requested_start_index, str) else requested_start_index
        self.camera = VisibleCamera(startup_index)
        if requested_start_index != startup_index:
            self._request_async_open(requested_start_index)
        if ENABLE_YOLO_DETECTION:
            self.yolo_status = "YOLO 模型加载中"
            self.yolo_worker = YoloWorker(self._yolo_results)
            self.yolo_worker.start()

        while self._running:
            self._apply_open_results()
            self._apply_yolo_results()
            pending_open = self._take_pending_open()
            if pending_open is not None:
                generation, pending_index = pending_open
                self._open_requested_camera(pending_index, generation)
                self._read_failures = 0

            frame = self.camera.read_rgb()
            if frame is None:
                self._read_failures += 1
                if self.camera.index != LOCAL_CAMERA_INDEX and self._read_failures >= NETWORK_FALLBACK_FAILURES:
                    failed_index = self.camera.index
                    if self.camera.open(LOCAL_CAMERA_INDEX):
                        self._read_failures = 0
                        message = f"摄像头无画面, 已自动切换到本地摄像头 {LOCAL_CAMERA_INDEX}: {failed_index}"
                        self.error_ready.emit(self.camera.index, message)
                    else:
                        self.error_ready.emit(self.camera.index, self.camera.error)
                    self.msleep(CAMERA_REFRESH_MS)
                    continue
                self.error_ready.emit(self.camera.index, self.camera.error)
                self.msleep(CAMERA_REFRESH_MS)
                continue

            self._read_failures = 0
            self._frame_count += 1
            tooltip = self.camera.error
            detect_interval = max(1, int(YOLO_DETECT_INTERVAL))
            should_detect = self._frame_count % detect_interval == 0

            yolo_enabled = self.is_yolo_enabled()
            if yolo_enabled and should_detect and self.yolo_worker is not None:
                self.yolo_worker.submit_frame(frame, (self._camera_generation, self._frame_count))
            if yolo_enabled:
                tooltip = self.yolo_status
            elif not tooltip:
                tooltip = "YOLO 检测已关闭"

            display_frame = frame
            if yolo_enabled:
                display_frame = YoloDetector.draw_detections(frame, self._last_detections)
            detection_count = len(self._last_detections)
            self.frame_ready.emit(
                display_frame,
                self.camera.index,
                detection_count,
                tooltip,
                self._has_detection_result,
                list(self._last_detections),
            )
            self.msleep(CAMERA_REFRESH_MS)

        if self.yolo_worker is not None:
            self.yolo_worker.stop()
        if self.camera is not None:
            self.camera.release()

    def _request_async_open(self, index):
        with self._lock:
            self._open_generation += 1
            generation = self._open_generation
        self._start_open_thread(index, generation)

    def _open_requested_camera(self, index, generation):
        if isinstance(index, str):
            self._start_open_thread(index, generation)
            return

        if self.camera is not None:
            self.camera.release()
        self.camera = VisibleCamera(index)
        self.index = self.camera.index
        self._read_failures = 0
        self._camera_generation += 1
        self._last_detections = []
        self._box_hold_frames = 0
        self._has_detection_result = False
        if self.camera.cap is None or not self.camera.cap.isOpened():
            self.error_ready.emit(index, self.camera.error)

    def _start_open_thread(self, index, generation):
        thread = threading.Thread(target=self._open_camera_thread, args=(index, generation), daemon=True)
        thread.start()

    def _open_camera_thread(self, index, generation):
        candidate = VisibleCamera(index, fallback_to_local=False)
        self._open_results.put((generation, index, candidate))

    def _apply_open_results(self):
        while True:
            try:
                generation, requested_index, candidate = self._open_results.get_nowait()
            except queue.Empty:
                return
            self._finish_async_open(generation, requested_index, candidate)

    def _finish_async_open(self, generation, requested_index, candidate):
        if generation != self._current_open_generation():
            candidate.release()
            return

        if candidate.cap is None or not candidate.cap.isOpened():
            error = candidate.error
            candidate.release()
            self.error_ready.emit(requested_index, error)
            return

        old_camera = self.camera
        self.camera = candidate
        self.index = candidate.index
        self._read_failures = 0
        self._camera_generation += 1
        self._last_detections = []
        self._box_hold_frames = 0
        self._has_detection_result = False
        if old_camera is not None and old_camera is not candidate:
            old_camera.release()
        if candidate.error:
            self.error_ready.emit(candidate.index, candidate.error)

    def _current_open_generation(self):
        with self._lock:
            return self._open_generation

    def _apply_yolo_results(self):
        while True:
            try:
                result = self._yolo_results.get_nowait()
            except queue.Empty:
                return

            if result[0] == "status":
                self.yolo_status = result[1]
                continue

            _, frame_key, detections, tooltip = result
            if isinstance(frame_key, tuple) and frame_key[0] != self._camera_generation:
                continue
            if detections:
                self._last_detections = detections
                self._box_hold_frames = YOLO_BOX_HOLD_FRAMES
            elif self._box_hold_frames > 0:
                self._box_hold_frames -= 1
            else:
                self._last_detections = []
            self._has_detection_result = True
            self.yolo_status = tooltip

    def _take_pending_open(self):
        with self._lock:
            pending_open = self._pending_open
            self._pending_open = None
            return pending_open


class ThermalSimulator:
    """80x60 热成像模拟器。"""

    def __init__(self, width=THERMAL_MATRIX_W, height=THERMAL_MATRIX_H):
        self.width = width
        self.height = height
        self.tick = 0
        y, x = np.mgrid[0:height, 0:width]
        self.x = x
        self.y = y

    def next_frame(self):
        """生成温度矩阵、伪彩图和高温点。"""

        self.tick += 1
        t = self.tick / 8.0

        temp = 24.0 + 1.8 * np.sin(self.x / 7.5 + t)
        temp += 1.5 * np.cos(self.y / 6.0 + t * 0.8)
        temp += np.random.normal(0, 0.18, (self.height, self.width))

        # 三个移动热源, 模拟轴承、电机、配电柜等设备发热。
        hot_sources = [
            (22 + 8 * math.sin(t * 0.8), 18 + 4 * math.cos(t * 0.7), 84),
            (53 + 6 * math.cos(t * 0.6), 32 + 5 * math.sin(t * 0.9), 76),
            (40 + 4 * math.sin(t * 0.5), 47 + 3 * math.cos(t), 64),
        ]
        for cx, cy, peak in hot_sources:
            sigma = 4.2
            g = np.exp(-(((self.x - cx) ** 2 + (self.y - cy) ** 2) / (2 * sigma ** 2)))
            temp += g * (peak - 24.0)

        hot_regions = self.analyze_hot_regions(temp)
        rgb = self.temperature_to_rgb(temp)
        return temp, rgb, hot_regions

    def analyze_hot_regions(self, temp):
        mask = temp >= HIGH_TEMP_LIMIT
        if not np.any(mask):
            return []

        ys, xs = np.where(mask)
        values = temp[ys, xs]
        order = np.argsort(values)[::-1]
        regions = []
        for idx in order:
            x = int(xs[idx])
            y = int(ys[idx])
            value = float(values[idx])
            if any((r.x - x) ** 2 + (r.y - y) ** 2 < 7 ** 2 for r in regions):
                continue
            level = "严重高温" if value >= 85 else "高温"
            regions.append(HotRegion(f"高温区域{len(regions) + 1}", x, y, value, level))
            if len(regions) >= 6:
                break
        return regions

    def temperature_to_rgb(self, temp):
        """蓝 -> 青 -> 黄 -> 红 的伪彩映射。"""

        norm = np.clip((temp - 20.0) / 80.0, 0, 1)
        r = np.clip(2.2 * norm - 0.45, 0, 1)
        g = np.clip(1.7 - np.abs(norm - 0.55) * 3.0, 0, 1)
        b = np.clip(1.25 - norm * 1.7, 0, 1)
        return (np.dstack([r, g, b]) * 255).astype(np.uint8)


# ----------------------------- 自定义科技控件 -----------------------------
class StaticPanel(QWidget):
    """工业面板。保留静态边框, 移除动态流光以降低刷新开销。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.radius = 0
        self.chamfer = 10
        self.setAttribute(Qt.WA_TranslucentBackground, True)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(2, 2, self.width() - 4, self.height() - 4)
        if rect.width() <= 8 or rect.height() <= 8:
            return

        path = self._panel_path(rect)

        # 半透明深蓝玻璃背景。
        bg = QLinearGradient(rect.topLeft(), rect.bottomRight())
        bg.setColorAt(0.0, QColor(6, 30, 58, 112))
        bg.setColorAt(0.45, QColor(3, 14, 29, 138))
        bg.setColorAt(1.0, QColor(5, 24, 48, 112))
        painter.fillPath(path, bg)

        # 静态边框。
        painter.setPen(QPen(QColor(0, 180, 255, 80), 1))
        painter.drawPath(path)

        # 角落高亮描边。
        self._draw_corners(painter, rect)

    def _panel_path(self, rect):
        cut = min(self.chamfer, rect.width() * 0.08, rect.height() * 0.20)
        path = QPainterPath()
        path.moveTo(rect.left() + cut, rect.top())
        path.lineTo(rect.right() - cut * 0.4, rect.top())
        path.lineTo(rect.right(), rect.top() + cut * 0.4)
        path.lineTo(rect.right(), rect.bottom() - cut)
        path.lineTo(rect.right() - cut, rect.bottom())
        path.lineTo(rect.left() + cut * 0.4, rect.bottom())
        path.lineTo(rect.left(), rect.bottom() - cut * 0.4)
        path.lineTo(rect.left(), rect.top() + cut)
        path.closeSubpath()
        return path

    def _draw_corners(self, painter, rect):
        corner = min(28, rect.width() * 0.12, rect.height() * 0.25)
        painter.setPen(QPen(QColor(0, 220, 255, 145), 1.5))
        painter.drawLine(rect.left(), rect.top(), rect.left() + corner, rect.top())
        painter.drawLine(rect.left(), rect.top(), rect.left(), rect.top() + corner)
        painter.drawLine(rect.right() - corner, rect.top(), rect.right(), rect.top())
        painter.drawLine(rect.right(), rect.top(), rect.right(), rect.top() + corner)
        painter.drawLine(rect.left(), rect.bottom() - corner, rect.left(), rect.bottom())
        painter.drawLine(rect.left(), rect.bottom(), rect.left() + corner, rect.bottom())
        painter.drawLine(rect.right() - corner, rect.bottom(), rect.right(), rect.bottom())
        painter.drawLine(rect.right(), rect.bottom() - corner, rect.right(), rect.bottom())

class TechPanel(StaticPanel):
    """带渐变标题栏的模块面板。"""

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setObjectName("TechPanel")
        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(5, 5, 5, 5)
        self.root.setSpacing(4)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("PanelTitle")
        self.title_label.setFixedHeight(27)

        self.content = QWidget()
        self.content.setObjectName("PanelContent")
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(12, 12, 12, 12)
        self.content_layout.setSpacing(10)

        self.root.addWidget(self.title_label)
        self.root.addWidget(self.content, 1)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        title = QRectF(5, 5, self.width() - 10, 27)
        if title.width() <= 10:
            return

        # 参考图里的模块标题不是普通矩形, 这里绘制左亮右暗的斜切标题条。
        path = QPainterPath()
        path.moveTo(title.left(), title.top())
        path.lineTo(title.right(), title.top())
        path.lineTo(title.right() - 18, title.bottom())
        path.lineTo(title.left(), title.bottom())
        path.closeSubpath()
        grad = QLinearGradient(title.left(), 0, title.right(), 0)
        grad.setColorAt(0.0, QColor(0, 95, 155, 145))
        grad.setColorAt(0.45, QColor(0, 120, 185, 120))
        grad.setColorAt(1.0, QColor(6, 27, 52, 45))
        painter.fillPath(path, grad)
        painter.setPen(QPen(QColor(20, 169, 255, 85), 1))
        painter.drawLine(QPointF(title.left(), title.bottom()), QPointF(title.right() - 20, title.bottom()))

        # 左侧黄色/青色小三角箭头, 更贴近原图标题栏。
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 223, 77, 210))
        painter.drawPolygon([
            QPointF(title.left() + 10, title.top() + 8),
            QPointF(title.left() + 18, title.top() + 13.5),
            QPointF(title.left() + 10, title.top() + 19),
        ])
        painter.setBrush(QColor(37, 234, 255, 165))
        painter.drawPolygon([
            QPointF(title.left() + 21, title.top() + 8),
            QPointF(title.left() + 29, title.top() + 13.5),
            QPointF(title.left() + 21, title.top() + 19),
        ])


class DashboardRoot(QWidget):
    """整体大屏背景, 仅保留静态背景和外框。"""

    def __init__(self, parent=None):
        super().__init__(parent)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        bg = QLinearGradient(0, 0, w, h)
        bg.setColorAt(0.0, QColor("#010511"))
        bg.setColorAt(0.34, QColor("#041025"))
        bg.setColorAt(0.68, QColor("#061a33"))
        bg.setColorAt(1.0, QColor("#010511"))
        painter.fillRect(self.rect(), bg)

        # 静态网格线。
        painter.setPen(QPen(QColor(30, 120, 180, 14), 1))
        grid = 40
        for x in range(0, w, grid):
            painter.drawLine(x, 0, x, h)
        for y in range(0, h, grid):
            painter.drawLine(0, y, w, y)

        frame = QRectF(8, 8, w - 16, h - 16)
        painter.setPen(QPen(QColor(0, 120, 220, 58), 2))
        painter.drawRect(frame)
        painter.setPen(QPen(QColor(20, 169, 255, 55), 1))
        painter.drawRect(frame.adjusted(5, 5, -5, -5))

        # 左右侧边机械感装饰
        painter.setPen(QPen(QColor(19, 134, 211, 90), 2))
        for side_x in (14, w - 14):
            sign = 1 if side_x < w / 2 else -1
            painter.drawLine(side_x, 90, side_x, h - 130)
            for i in range(9):
                y = 115 + i * 45
                painter.drawLine(side_x, y, side_x + sign * 9, y + 8)

        # 底部中央光带
        path = QPainterPath()
        path.moveTo(w * 0.38, h - 28)
        path.lineTo(w * 0.44, h - 28)
        path.lineTo(w * 0.46, h - 40)
        path.lineTo(w * 0.54, h - 40)
        path.lineTo(w * 0.56, h - 28)
        path.lineTo(w * 0.62, h - 28)
        painter.setPen(QPen(QColor(20, 169, 255, 120), 3))
        painter.drawPath(path)


class LogoBlock(QWidget):
    """顶部左侧品牌文字块。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(230)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QColor(120, 215, 255, 60))
        painter.setFont(QFont("Microsoft YaHei", 15, QFont.Bold))
        painter.drawText(QRectF(2, 12, self.width(), 28), Qt.AlignLeft | Qt.AlignVCenter, "工业智慧巡检平台")
        painter.setPen(QColor("#edfaff"))
        painter.drawText(QRectF(0, 10, self.width(), 28), Qt.AlignLeft | Qt.AlignVCenter, "工业智慧巡检平台")
        painter.setPen(QColor("#9ecff2"))
        painter.setFont(QFont("Consolas", 10, QFont.Bold))
        painter.drawText(QRectF(0, 42, self.width(), 24), Qt.AlignLeft | Qt.AlignVCenter, "Tonjin Industrial Robot")


class MenuGroup(QWidget):
    """顶部左右菜单组, 使用布局管理避免和标题重叠。"""

    def __init__(self, items, active_text=None, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(18)
        for text in items:
            label = QLabel(text)
            label.setAlignment(align_center())
            label.setObjectName("TopMenuActive" if text == active_text else "TopMenu")
            label.setMinimumWidth(78)
            label.setFixedHeight(34)
            layout.addWidget(label)


class TitleBadge(QWidget):
    """顶部中间梯形标题框。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(360)
        self.setMaximumWidth(430)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        rect = QRectF(8, 6, w - 16, h - 14)

        path = QPainterPath()
        path.moveTo(rect.left() + 48, rect.top())
        path.lineTo(rect.right() - 48, rect.top())
        path.lineTo(rect.right(), rect.center().y())
        path.lineTo(rect.right() - 72, rect.bottom())
        path.lineTo(rect.left() + 72, rect.bottom())
        path.lineTo(rect.left(), rect.center().y())
        path.closeSubpath()

        grad = QLinearGradient(rect.left(), 0, rect.right(), 0)
        grad.setColorAt(0.0, QColor(5, 22, 48, 205))
        grad.setColorAt(0.50, QColor(12, 95, 160, 210))
        grad.setColorAt(1.0, QColor(5, 22, 48, 205))
        painter.fillPath(path, grad)

        painter.setPen(QPen(QColor(0, 190, 255, 75), 5))
        painter.drawPath(path)
        painter.setPen(QPen(QColor(0, 180, 255, 150), 1.5))
        painter.drawPath(path)

        painter.setPen(QColor(155, 230, 255, 80))
        painter.setFont(QFont("Microsoft YaHei", 21, QFont.Bold))
        painter.drawText(rect.adjusted(1, 1, 1, 1), align_center(), "智能巡检-综合管理平台")
        painter.setPen(QColor("#f1fbff"))
        painter.drawText(rect, align_center(), "智能巡检-综合管理平台")


class TopBar(QWidget):
    """使用 QHBoxLayout 分区的顶部标题栏, 避免标题框压到菜单文字。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(86)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(28, 6, 24, 8)
        layout.setSpacing(25)

        self.logo = LogoBlock()
        self.left_menu = MenuGroup(["系统总览", "参数设置", "趋势曲线"], active_text="参数设置")
        self.title_badge = TitleBadge()
        self.right_menu = MenuGroup(["数据报表", "报警信息", "巡检记录"])
        self.time_label = QLabel("--")
        self.time_label.setObjectName("TopTime")
        self.time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.time_label.setMinimumWidth(175)

        layout.addWidget(self.logo, 0)
        layout.addWidget(self.left_menu, 0)
        layout.addSpacing(8)
        layout.addWidget(self.title_badge, 1)
        layout.addSpacing(8)
        layout.addWidget(self.right_menu, 0)
        layout.addWidget(self.time_label, 0)

    def set_time(self, text):
        self.time_label.setText(text)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        grad = QLinearGradient(0, 0, w, 0)
        grad.setColorAt(0.0, QColor(2, 8, 22, 230))
        grad.setColorAt(0.5, QColor(5, 20, 42, 235))
        grad.setColorAt(1.0, QColor(2, 8, 22, 230))
        painter.fillRect(self.rect(), grad)

        painter.setPen(QPen(QColor(0, 150, 240, 65), 2))
        painter.drawLine(16, 10, w * 0.16, 10)
        painter.drawLine(w * 0.16, 10, w * 0.18, 24)
        painter.drawLine(w * 0.18, 24, w * 0.34, 24)
        painter.drawLine(w * 0.66, 24, w * 0.82, 24)
        painter.drawLine(w * 0.82, 24, w * 0.84, 10)
        painter.drawLine(w * 0.84, 10, w - 16, 10)

        painter.setPen(QPen(QColor(0, 160, 255, 38), 1))
        painter.drawLine(18, h - 10, w - 18, h - 10)


class NumberCard(QFrame):
    """左侧和统计区使用的小数据卡片。"""

    def __init__(self, name, value, color=CYAN, parent=None):
        super().__init__(parent)
        self.setObjectName("NumberCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)
        title = QLabel(name)
        title.setObjectName("SmallText")
        number = QLabel(value)
        number.setObjectName("NumberText")
        number.setAlignment(align_center())
        number.setStyleSheet(f"color: {color};")
        layout.addWidget(title)
        layout.addWidget(number)


class MiniMetric(QWidget):
    """机器人模块顶部小指标, 比 NumberCard 更紧凑清晰。"""

    def __init__(self, name, value, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(1)
        title = QLabel(name)
        title.setObjectName("SmallText")
        title.setAlignment(align_center())
        val = QLabel(value)
        val.setObjectName("NumberText")
        val.setAlignment(align_center())
        val.setStyleSheet(f"color: {CYAN}; font-size: 13pt;")
        layout.addWidget(title)
        layout.addWidget(val)


class RobotIcon(QWidget):
    """机器人圆形模拟图标。"""

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        r = min(w, h) * 0.38

        painter.setPen(QPen(QColor("#106db5"), 2))
        painter.setBrush(QColor("#082d58"))
        painter.drawEllipse(QPointF(cx, cy), r, r)
        painter.setPen(QPen(QColor(CYAN), 2))
        painter.drawEllipse(QPointF(cx, cy), r * 0.72, r * 0.72)
        painter.setBrush(QColor("#39ddff"))
        painter.drawEllipse(QPointF(cx, cy - r * 0.25), r * 0.24, r * 0.18)
        painter.drawRoundedRect(QRectF(cx - r * 0.22, cy - r * 0.08, r * 0.44, r * 0.42), 6, 6)
        painter.setBrush(QColor("#031325"))
        painter.drawEllipse(QPointF(cx - r * 0.08, cy - r * 0.25), 3, 3)
        painter.drawEllipse(QPointF(cx + r * 0.08, cy - r * 0.25), 3, 3)
        painter.setPen(QPen(QColor("#1aa5ff"), 2))
        for i in range(3):
            rr = r + 12 + i * 8
            painter.drawArc(QRectF(cx - rr, cy - rr, rr * 2, rr * 2), 210 * 16, 120 * 16)


class ChargeIcon(QWidget):
    """充电站状态圆形图标。"""

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        r = min(w, h) * 0.36
        painter.setPen(QPen(QColor("#0e82d3"), 2))
        painter.setBrush(QColor("#082d58"))
        painter.drawEllipse(QPointF(cx, cy), r, r)
        painter.setPen(QPen(QColor(CYAN), 2))
        painter.drawEllipse(QPointF(cx, cy), r * 0.62, r * 0.62)
        painter.setPen(QColor(GREEN))
        painter.setFont(QFont("Microsoft YaHei", 12, QFont.Bold))
        painter.drawText(QRectF(0, cy - 14, w, 28), align_center(), "状态正常")


class RingGauge(QWidget):
    """圆环温度仪表盘, 支持3级报警闪烁动画。

    级别:
        0 - 正常 (绿色, 不闪烁)
        1 - 二级预警 (黄色闪烁)
        2 - 三级报警 (红色闪烁)
    """

    ALARM_DIM_RED = QColor("#3a1010")     # 红色闪烁暗色
    ALARM_DIM_YELLOW = QColor("#3a3010")  # 黄色闪烁暗色
    ALARM_FLASH_MS = 500                  # 闪烁周期 (ms)

    def __init__(self, title, color, value=0.0, parent=None):
        super().__init__(parent)
        self.title = title
        self.color = QColor(color)
        self.normal_color = QColor(color)
        self.value = value
        self._alarm_level = 0          # 0=正常, 1=黄闪, 2=红闪
        self._flash_on = True
        self._flash_timer = QTimer(self)
        self._flash_timer.timeout.connect(self._tick_flash)
        self.setMinimumHeight(102)

    def set_color(self, color):
        if self._alarm_level == 0:
            self.normal_color = QColor(color)
        self.color = QColor(color)
        self.update()

    def set_alarm_level(self, level):
        """设置报警级别: 0=正常, 1=黄色闪烁, 2=红色闪烁。"""
        level = max(0, min(2, level))
        if level == self._alarm_level:
            return
        self._alarm_level = level
        if level == 0:
            self._flash_timer.stop()
            self._flash_on = True
            self.color = self.normal_color
        else:
            self._flash_on = True
            self._flash_timer.start(self.ALARM_FLASH_MS)
            self.color = QColor(YELLOW) if level == 1 else QColor(RED)
        self.update()

    def set_alarm(self, is_alarm):
        """兼容旧接口: True=级别2(红闪), False=级别0(正常)。"""
        self.set_alarm_level(2 if is_alarm else 0)

    def _tick_flash(self):
        self._flash_on = not self._flash_on
        if self._alarm_level == 1:
            self.color = QColor(YELLOW) if self._flash_on else self.ALARM_DIM_YELLOW
        elif self._alarm_level == 2:
            self.color = QColor(RED) if self._flash_on else self.ALARM_DIM_RED
        self.update()

    def set_value(self, value):
        self.value = float(value)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        size = max(56, min(w, h - 10) - 18)
        rect = QRectF((w - size) / 2, 4, size, size)
        for grow, alpha in [(8, 18), (4, 35)]:
            glow_rect = rect.adjusted(-grow, -grow, grow, grow)
            painter.setPen(QPen(QColor(self.color.red(), self.color.green(), self.color.blue(), alpha), 3))
            painter.drawArc(glow_rect, 90 * 16, int(-300 * min(self.value / 100.0, 1.0) * 16))
        painter.setPen(QPen(QColor("#173a58"), 8))
        painter.drawArc(rect, 0, 360 * 16)
        painter.setPen(QPen(self.color, 8))
        painter.drawArc(rect, 90 * 16, int(-300 * min(self.value / 100.0, 1.0) * 16))
        painter.setPen(QColor(TEXT))
        painter.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        painter.drawText(rect, align_center(), self.title)
        painter.setPen(self.color)
        painter.setFont(QFont("Consolas", 11, QFont.Bold))
        painter.drawText(rect.adjusted(0, 22, 0, 22), align_center(), f"{self.value:.1f}℃")


class StatCircle(QWidget):
    """右上巡检统计圆形图标。"""

    def __init__(self, title, value, color, symbol, parent=None):
        super().__init__(parent)
        self.title = title
        self.value = value
        self.color = QColor(color)
        self.symbol = symbol
        self.setMinimumHeight(122)

    def set_display(self, title, value, color, symbol):
        self.title = title
        self.value = value
        self.color = QColor(color)
        self.symbol = symbol
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2, 38
        painter.setBrush(QColor(self.color.red(), self.color.green(), self.color.blue(), 90))
        for rr, alpha in [(44, 24), (39, 45)]:
            painter.setPen(QPen(QColor(self.color.red(), self.color.green(), self.color.blue(), alpha), 3))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(QPointF(cx, cy), rr, rr)
        painter.setBrush(QColor(self.color.red(), self.color.green(), self.color.blue(), 90))
        painter.setPen(QPen(self.color, 2))
        painter.drawEllipse(QPointF(cx, cy), 35, 35)
        painter.setPen(self.color)
        painter.setFont(QFont("Arial", 24, QFont.Bold))
        painter.drawText(QRectF(0, 12, w, 52), align_center(), self.symbol)
        painter.setPen(QColor(TEXT))
        painter.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        painter.drawText(QRectF(0, 74, w, 20), align_center(), self.title)
        painter.setPen(self.color)
        painter.setFont(QFont("Consolas", 17, QFont.Bold))
        painter.drawText(QRectF(0, 94, w, 24), align_center(), self.value)


class CameraView(QLabel):
    """相机画面显示控件。铺满显示, 不留黑边。"""

    def __init__(self, text, parent=None, show_replay=False):
        super().__init__(text, parent)
        self.setObjectName("CameraView")
        self.setAlignment(align_center())
        self.setMinimumSize(240, 180)
        self.show_replay = show_replay

    def set_rgb_image(self, rgb_image):
        if rgb_image is None:
            return
        h, w, c = rgb_image.shape
        qimg = QImage(rgb_image.data, w, h, c * w, qimage_rgb888()).copy()
        pixmap = QPixmap.fromImage(qimg).scaled(self.size(), Qt.KeepAspectRatioByExpanding, Qt.FastTransformation)
        if pixmap.width() > self.width() or pixmap.height() > self.height():
            x = max(0, (pixmap.width() - self.width()) // 2)
            y = max(0, (pixmap.height() - self.height()) // 2)
            pixmap = pixmap.copy(x, y, self.width(), self.height())
        self.setPixmap(pixmap)

    def show_text(self, text):
        self.setPixmap(QPixmap())
        self.setText(text)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # 相机窗口内边框和角标, 接近参考图里的实时视频窗。
        painter.setPen(QPen(QColor(35, 184, 255, 160), 1))
        painter.drawRect(self.rect().adjusted(1, 1, -2, -2))
        painter.setPen(QPen(QColor(LINE), 3))
        length = 34
        painter.drawLine(8, 8, 8 + length, 8)
        painter.drawLine(8, 8, 8, 8 + length)
        painter.drawLine(w - 8 - length, h - 8, w - 8, h - 8)
        painter.drawLine(w - 8, h - 8 - length, w - 8, h - 8)

        # 右上角“实时”状态块。
        badge = QRectF(w - 70, 8, 42, 22)
        painter.fillRect(badge, QColor("#0b95df"))
        painter.setPen(QColor("#ffffff"))
        painter.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        painter.drawText(badge, align_center(), "实时")
        if self.show_replay:
            replay = QRectF(w - 28, 8, 22, 22)
            painter.fillRect(replay, QColor("#284a67"))
            painter.setPen(QColor(MUTED))
            painter.drawText(replay, align_center(), "回")


class AspectRatioBox(QWidget):
    """让内部控件保持固定宽高比, 多余空间留给父级背景。"""

    def __init__(self, ratio=4 / 3, parent=None):
        super().__init__(parent)
        self.ratio = ratio
        self.child = None
        self.setMinimumSize(240, 180)

    def set_widget(self, widget):
        self.child = widget
        widget.setParent(self)
        self._update_child_geometry()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_child_geometry()

    def _update_child_geometry(self):
        if self.child is None:
            return
        area_w = self.width()
        area_h = self.height()
        if area_w <= 0 or area_h <= 0:
            return

        target_w = area_w
        target_h = int(target_w / self.ratio)
        if target_h > area_h:
            target_h = area_h
            target_w = int(target_h * self.ratio)

        x = (area_w - target_w) // 2
        y = (area_h - target_h) // 2
        self.child.setGeometry(x, y, target_w, target_h)


class ThermalWidget(CameraView):
    """热成像显示控件, 支持右键菜单设置热区追踪参数。"""

    def __init__(self, parent=None):
        super().__init__("热成像相机", parent=parent, show_replay=True)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_right_click)
        self._main_window = None

    def bind_main_window(self, main_window):
        """绑定 MainWindow 引用, 用于弹出设置对话框。"""
        self._main_window = main_window

    def _on_right_click(self, pos):
        if self._main_window is None:
            return
        menu = QMenu()
        params = self._main_window.zone_tracker.get_params()
        info_action = menu.addAction(f"检测阈值: {params['threshold']:.0f}℃")
        info_action.setEnabled(False)
        menu.addSeparator()
        pop_action = menu.addAction("详情窗口")
        pop_action.triggered.connect(self._main_window._show_thermal_pop_window)
        menu.addAction("热区追踪参数...").triggered.connect(
            lambda: self._main_window._show_zone_settings_menu(self.mapToGlobal(pos))
        )
        menu.exec_(self.mapToGlobal(pos))


class ThermalPopWindow(QMainWindow):
    """独立热成像窗口：显示大尺寸热成像 + 温度仪表盘 + 热区信息。"""

    def __init__(self, thermal_camera_mgr, zone_tracker, alarm_ctrl, parent=None):
        super().__init__(parent)
        self.setWindowTitle("热成像 - 详情")
        self.resize(800, 720)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)

        # 热成像画面（大尺寸）
        self.thermal_view = CameraView("热成像", show_replay=False)
        self.thermal_view.setMinimumSize(640, 480)
        layout.addWidget(self.thermal_view, stretch=3)

        # 仪表盘行
        gauge_row = QHBoxLayout()
        self.max_gauge = RingGauge("最高温", YELLOW, 0.0)
        self.min_gauge = RingGauge("最低温", BLUE, 0.0)
        self.avg_gauge = RingGauge("平均温", CYAN, 0.0)
        for g in [self.max_gauge, self.min_gauge, self.avg_gauge]:
            g.setFixedSize(120, 120)
        gauge_row.addWidget(self.max_gauge)
        gauge_row.addStretch()
        gauge_row.addWidget(self.min_gauge)
        gauge_row.addStretch()
        gauge_row.addWidget(self.avg_gauge)
        layout.addLayout(gauge_row)

        # 热区信息标签
        self.hot_info = QLabel()
        self.hot_info.setAlignment(Qt.AlignCenter)
        self.hot_info.setStyleSheet("color: #aabbcc; background: #0a1520; padding: 4px; border-radius: 4px;")
        layout.addWidget(self.hot_info)

        # 定时器：每 150ms 更新（与主窗口一致）
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update)
        self.timer.start(150)

        self._camera_mgr = thermal_camera_mgr
        self._zone_tracker = zone_tracker
        self._alarm_ctrl = alarm_ctrl
        self.threshold_warn = 40.0
        self.threshold_alarm = 70.0

    def _update(self):
        frame = self._camera_mgr.get_frame()
        if frame is None:
            return

        # 热区追踪
        self._zone_tracker.update(frame.temps)
        annotated = self._zone_tracker.draw_overlay(frame.rgb)
        self.thermal_view.set_rgb_image(annotated)

        # 更新仪表盘
        max_level, max_color = temperature_alarm_state(frame.max_temp, self.threshold_warn, self.threshold_alarm)
        _min_level, min_color = temperature_alarm_state(frame.min_temp, self.threshold_warn, self.threshold_alarm)
        _avg_level, avg_color = temperature_alarm_state(frame.avg_temp, self.threshold_warn, self.threshold_alarm)

        self.max_gauge.set_value(frame.max_temp)
        self.min_gauge.set_value(frame.min_temp)
        self.avg_gauge.set_value(frame.avg_temp)
        self.max_gauge.set_color(max_color)
        self.min_gauge.set_color(min_color)
        self.avg_gauge.set_color(avg_color)

        _level_map = {"正常": 0, "二级预警": 1, "三级报警": 2}
        self.max_gauge.set_alarm_level(_level_map.get(max_level, 0))
        self.min_gauge.set_alarm_level(_level_map.get(_min_level, 0))
        self.avg_gauge.set_alarm_level(_level_map.get(_avg_level, 0))

        self.hot_info.setText(
            f"当前预警等级: {max_level}　　"
            f"最高: {frame.max_temp:.1f}℃ / 最低: {frame.min_temp:.1f}℃ / 平均: {frame.avg_temp:.1f}℃　　"
            f"预警≥{self.threshold_warn:.0f}℃ / 报警≥{self.threshold_alarm:.0f}℃"
        )

    def closeEvent(self, event):
        self.timer.stop()
        self.deleteLater()
        super().closeEvent(event)


class RouteMapWidget(QWidget):
    """1-20 点位巡检路线实时模拟。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_index = 0
        self.abnormal_indexes = {2, 12}
        self.speed = 0.0
        self.position = 0.0
        self.setMinimumHeight(150)

    def set_state(self, current_index, abnormal_indexes, speed, position):
        self.current_index = current_index
        self.abnormal_indexes = set(abnormal_indexes)
        self.speed = speed
        self.position = position
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(self.rect(), QColor("#05172d"))
        box = QRectF(28, 18, w - 56, h - 58)
        painter.setPen(QPen(QColor("#0f73b8"), 3))
        painter.drawRoundedRect(box, 8, 8)

        points = []
        for row in range(2):
            for col in range(10):
                x = box.left() + 34 + col * (box.width() - 68) / 9
                y = box.top() + 26 + row * (box.height() - 52)
                points.append((x, y))

        painter.setPen(QPen(QColor("#1b5b86"), 2))
        for i in range(9):
            painter.drawLine(QPointF(*points[i]), QPointF(*points[i + 1]))
            painter.drawLine(QPointF(*points[10 + i]), QPointF(*points[11 + i]))
        painter.drawLine(QPointF(*points[9]), QPointF(*points[19]))

        for i, (x, y) in enumerate(points, start=1):
            color = RED if i in self.abnormal_indexes else GREEN
            if i == self.current_index:
                color = YELLOW
            painter.setBrush(QColor(color))
            painter.setPen(QPen(QColor("#071426"), 2))
            painter.drawEllipse(QPointF(x, y), 10, 10)
            painter.setPen(QColor(TEXT))
            painter.setFont(QFont("Consolas", 10, QFont.Bold))
            painter.drawText(QRectF(x - 15, y + 12, 30, 18), Qt.AlignCenter, str(i))

        painter.setPen(QColor(CYAN))
        painter.setFont(QFont("Consolas", 14, QFont.Bold))
        painter.drawText(QRectF(40, h - 34, 200, 28), Qt.AlignLeft | Qt.AlignVCenter, f"当前速度: {self.speed:.2f}")
        painter.drawText(QRectF(w - 250, h - 34, 220, 28), Qt.AlignRight | Qt.AlignVCenter, f"当前位置: {self.position:.2f}")


class TrendChart(QWidget):
    """通用折线图, 用于 CH4 曲线和噪声检测。"""

    def __init__(self, labels=None, show_threshold=True, parent=None):
        super().__init__(parent)
        self.labels = labels or []
        self.show_threshold = show_threshold
        self.values_a = [random.uniform(8, 20) for _ in range(60)]
        self.values_b = [random.uniform(3, 10) for _ in range(60)]
        self.max_value = 100
        self.setMinimumHeight(120)

    def push(self, a, b):
        self.values_a = (self.values_a + [a])[-60:]
        self.values_b = (self.values_b + [b])[-60:]
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(self.rect(), QColor("#06182f"))
        left, right, top, bottom = 34, 10, 12, 24
        chart = QRectF(left, top, w - left - right, h - top - bottom)

        painter.setPen(QPen(QColor("#173a5c"), 1))
        for i, value in enumerate([0, 25, 50, 75, 100]):
            y = chart.bottom() - value / 100 * chart.height()
            painter.drawLine(QPointF(chart.left(), y), QPointF(chart.right(), y))
            painter.setPen(QColor(MUTED))
            painter.setFont(QFont("Consolas", 8))
            painter.drawText(QRectF(0, y - 8, left - 6, 16), Qt.AlignRight | Qt.AlignVCenter, str(value))
            painter.setPen(QPen(QColor("#173a5c"), 1))

        if self.show_threshold:
            painter.setPen(QPen(QColor(RED), 1))
            y50 = chart.bottom() - 50 / 100 * chart.height()
            painter.drawLine(QPointF(chart.left(), y50), QPointF(chart.right(), y50))
            painter.setPen(QPen(QColor(YELLOW), 1))
            y25 = chart.bottom() - 25 / 100 * chart.height()
            painter.drawLine(QPointF(chart.left(), y25), QPointF(chart.right(), y25))

        self._draw_curve(painter, chart, self.values_a, LINE)
        self._draw_curve(painter, chart, self.values_b, YELLOW)

        if self.labels:
            painter.setPen(QColor(MUTED))
            painter.setFont(QFont("Microsoft YaHei", 8, QFont.Bold))
            for i, text in enumerate(self.labels):
                x = chart.left() + i * chart.width() / max(1, len(self.labels) - 1)
                painter.drawText(QRectF(x - 28, chart.bottom() + 3, 56, 18), Qt.AlignCenter, text)

    def _draw_curve(self, painter, chart, values, color):
        if len(values) < 2:
            return
        path = QPainterPath()
        for i, value in enumerate(values):
            x = chart.left() + i * chart.width() / (len(values) - 1)
            y = chart.bottom() - max(0, min(value, self.max_value)) / self.max_value * chart.height()
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        glow = QColor(color)
        glow.setAlpha(55)
        painter.setPen(QPen(glow, 6))
        painter.drawPath(path)
        painter.setPen(QPen(QColor(color), 2))
        painter.drawPath(path)


class CH4Gauge(QWidget):
    """CH4 圆环进度。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = 0.0
        self.setMinimumSize(120, 120)

    def set_value(self, value):
        self.value = value
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        size = min(w, h) - 20
        rect = QRectF((w - size) / 2, 8, size, size)
        painter.setPen(QPen(QColor("#153a5b"), 7))
        painter.drawArc(rect, 0, 360 * 16)
        painter.setPen(QPen(QColor(CYAN), 7))
        painter.drawArc(rect, 90 * 16, int(-300 * min(self.value / 100.0, 1.0) * 16))
        painter.setPen(QColor(TEXT))
        painter.setFont(QFont("Consolas", 12, QFont.Bold))
        painter.drawText(rect, align_center(), f"{self.value:.1f}%LEL")
        painter.setPen(QColor(MUTED))
        painter.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
        painter.drawText(QRectF(0, h - 25, w, 20), align_center(), "CH4 浓度")


class EnvBar(QWidget):
    """环境检测横向进度条, 左侧带六边形科技图标。"""

    def __init__(self, title, unit, color, parent=None):
        super().__init__(parent)
        self.title = title
        self.unit = unit
        self.color = QColor(color)
        self.value = 0.0
        self.max_value = 100.0
        self.setMinimumHeight(42)

    def set_value(self, value, max_value=100.0):
        self.value = float(value)
        self.max_value = float(max_value)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        cx, cy = 22, h / 2
        path = QPainterPath()
        for i in range(6):
            angle = math.pi / 6 + i * math.pi / 3
            x = cx + 16 * math.cos(angle)
            y = cy + 16 * math.sin(angle)
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        path.closeSubpath()
        painter.setPen(QPen(self.color, 2))
        painter.setBrush(QColor(self.color.red(), self.color.green(), self.color.blue(), 55))
        painter.drawPath(path)

        painter.setPen(QColor(TEXT))
        painter.setFont(QFont("Microsoft YaHei", 10, QFont.Bold))
        painter.drawText(QRectF(50, 0, 70, h), Qt.AlignLeft | Qt.AlignVCenter, self.title)

        bar = QRectF(118, h / 2 - 6, max(80, w - 220), 12)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#183a5c"))
        painter.drawRoundedRect(bar, 6, 6)
        ratio = max(0, min(self.value / self.max_value, 1))
        painter.setBrush(self.color)
        glow = QColor(self.color)
        glow.setAlpha(45)
        painter.setBrush(glow)
        painter.drawRoundedRect(QRectF(bar.left(), bar.top() - 3, bar.width() * ratio, bar.height() + 6), 8, 8)
        painter.setBrush(self.color)
        painter.drawRoundedRect(QRectF(bar.left(), bar.top(), bar.width() * ratio, bar.height()), 6, 6)

        painter.setPen(self.color)
        painter.setFont(QFont("Consolas", 13, QFont.Bold))
        painter.drawText(QRectF(w - 95, 0, 90, h), Qt.AlignRight | Qt.AlignVCenter, f"{self.value:.2f}{self.unit}")


# ----------------------------- 主窗口 -----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能巡检-综合管理平台")
        self.resize(1600, 900)
        self.setMinimumSize(1280, 720)

        self.thermal_serial = ThermalSerialReader()
        self.stm32 = Stm32SerialController()
        self.current_camera_index = VISIBLE_CAMERA_INDEX

        self.hot_regions = []
        self.route_index = 1
        self.control_state = "系统待命"

        # 超温报警阈值 (右键温度仪表盘可修改, 超过后下发 $ALARM 到下位机)
        self.threshold_alarm = HIGH_TEMP_LIMIT  # 70℃ (三级报警-红闪)
        self.threshold_warn = TEMP_WARN_LEVEL_1  # 40℃ (二级预警-黄闪)
        self.no_hat_first_seen = None
        self.no_hat_alert_active = False
        self.no_hat_last_seen = None
        self.no_hat_audio_path = Path(__file__).resolve().parent / "alarm1.mp3"
        self.no_hat_audio_player = None
        self.no_hat_audio_output = None
        self.no_hat_audio_available = False
        self.no_hat_audio_playing = False
        self.no_hat_audio_cooldown_until = 0.0
        self.no_hat_audio_repeat_requested = False
        self._init_no_hat_audio()

        # ════════════════════════════════════════════════════════
        # [扩展接口] 初始化扩展模块
        # 在此处创建 ThermalCameraManager / MCUSerialBridge / AlarmController / ThermalZoneTracker
        # 详见 docs/02-扩展模块接口.md §5.2
        # ════════════════════════════════════════════════════════
        if EXTENSIONS_AVAILABLE:
            self.thermal_camera_mgr = ThermalCameraManager(config={
                "wifi_host": "192.168.3.166",
                "wifi_port": 5001,
                "lepton_baud": 921600,
                "default_device": "simulator",
            })
            self.mcu_bridge = MCUSerialBridge(baudrate=115200)
            self.alarm_ctrl = AlarmController(threshold=70.0, mcu_bridge=self.mcu_bridge)
            self.zone_tracker = ThermalZoneTracker(threshold=70.0)
            self.ext_alarm_state = None  # 最新报警状态
            # STM32 机器人控制串口 (替代旧的 Stm32SerialController 占位)
            self.ext_stm32 = ExtStm32Controller(port="COM3", baudrate=115200)
        # ════════════════════════════════════════════════════════

        self._build_ui()
        self._apply_qss()
        self.refresh_camera_list()
        self.update_thermal()
        self.update_auxiliary_data()
        self.update_environment()

        self.aux_timer = QTimer(self)
        self.aux_timer.timeout.connect(self.update_auxiliary_data)
        self.aux_timer.start(AUX_REFRESH_MS)

        self.camera_worker = CameraWorker(VISIBLE_CAMERA_INDEX, self)
        self.camera_worker.frame_ready.connect(self.on_camera_frame)
        self.camera_worker.error_ready.connect(self.on_camera_error)
        self.camera_worker.start()

        # ════════════════════════════════════════════════════════
        # [扩展接口] 热像仪/环境数据定时刷新
        # 150ms 周期与参考项目一致, 保证画面流畅
        # ════════════════════════════════════════════════════════
        if EXTENSIONS_AVAILABLE:
            self.ext_thermal_timer = QTimer(self)
            self.ext_thermal_timer.timeout.connect(self.update_thermal)
            self.ext_thermal_timer.timeout.connect(self.update_environment)
            self.ext_thermal_timer.start(150)
        # ════════════════════════════════════════════════════════

    def _build_ui(self):
        central = DashboardRoot()
        central.setObjectName("Root")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 10, 18, 16)
        root.setSpacing(7)

        self.header = TopBar()
        root.addWidget(self.header)

        body = QGridLayout()
        body.setSpacing(14)
        body.setColumnStretch(0, 22)
        body.setColumnStretch(1, 56)
        body.setColumnStretch(2, 22)
        root.addLayout(body, 1)

        body.addWidget(self._build_left(), 0, 0)
        body.addWidget(self._build_center(), 0, 1)
        body.addWidget(self._build_right(), 0, 2)

    # ------------------------- 左侧区域 -------------------------
    def _build_left(self):
        col = QWidget()
        layout = QVBoxLayout(col)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(self._basic_info_panel(), 20)
        layout.addWidget(self._robot_panel(), 48)
        layout.addWidget(self._charge_panel(), 18)
        return col

    def _basic_info_panel(self):
        panel = TechPanel("基本信息")
        grid = QGridLayout()
        grid.setSpacing(12)
        panel.content_layout.addLayout(grid)
        items = [("巡检路线", "1 条"), ("巡检设备", "1 台"), ("巡检区域", "1#无人作业平台"), ("巡检点位", "20 个")]
        for i, (name, value) in enumerate(items):
            grid.addWidget(NumberCard(name, value), i // 2, i % 2)
        return panel

    def _robot_panel(self):
        panel = TechPanel("智能巡检机器人")
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)
        panel.content_layout.addLayout(grid)

        grid.addWidget(MiniMetric("累计运行", "43 天"), 0, 0)
        grid.addWidget(MiniMetric("累计运行", "404 次"), 0, 1)
        robot = RobotIcon()
        grid.addWidget(robot, 0, 2, 2, 1)
        grid.addWidget(MiniMetric("运行速度", "43 RPM"), 1, 0)
        grid.addWidget(MiniMetric("当前位置", "43 M"), 1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)

        states = [
            ("通信状态", "正常", GREEN), ("控制模式", "控温模式", CYAN), ("运行状态", "正常", GREEN),
            ("急停状态", "未触发", GREEN), ("正向运行", "停止", RED), ("反向运行", "停止", RED),
            ("前向避障", "触发", YELLOW), ("后向避障", "触发", YELLOW),
        ]
        for i, (name, value, color) in enumerate(states):
            label = QLabel(f"{name}  {value}")
            label.setObjectName("StateLabel")
            label.setStyleSheet(f"color: {color};")
            label.setMinimumHeight(32)
            label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            grid.addWidget(label, 2 + i // 2, i % 2, 1, 1)

        self.battery_label = QLabel()
        self.voltage_label = QLabel()
        self.cell_temp_label = QLabel()
        self.cabin_temp_label = QLabel()
        for i, label in enumerate([self.battery_label, self.voltage_label, self.cell_temp_label, self.cabin_temp_label]):
            label.setObjectName("DataLine")
            grid.addWidget(label, 6 + i, 0, 1, 3)
        return panel

    def _charge_panel(self):
        panel = TechPanel("智能充电站")
        row = QHBoxLayout()
        row.setSpacing(12)
        icon = ChargeIcon()
        row.addWidget(QLabel("系统电压\n49.40 V"), 1)
        row.addWidget(icon, 2)
        row.addWidget(QLabel("系统电流\n0.70 A"), 1)
        for i in range(row.count()):
            item = row.itemAt(i).widget()
            if isinstance(item, QLabel):
                item.setObjectName("ChargeText")
                item.setAlignment(align_center())
        panel.content_layout.addLayout(row, 1)
        return panel

    # ------------------------- 中间区域 -------------------------
    def _build_center(self):
        col = QWidget()
        layout = QVBoxLayout(col)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        top = QHBoxLayout()
        top.setSpacing(12)

        self.visible_panel_title = "可见光相机"
        visible_panel = TechPanel(self.visible_panel_title)
        self.visible_panel = visible_panel
        self.visible_panel.title_label.setText(f"{self.visible_panel_title}   检测中")
        tools = QHBoxLayout()
        tools.setContentsMargins(0, 0, 0, 0)
        tools.setSpacing(6)
        tools.addWidget(QLabel("系统摄像头"))
        self.camera_combo = QComboBox()
        self.camera_refresh_btn = QPushButton("刷新")
        self.camera_connect_btn = QPushButton("连接")
        self.yolo_on_btn = QPushButton("开启YOLO")
        self.yolo_off_btn = QPushButton("关闭YOLO")
        self.camera_state_label = QLabel("等待连接")
        self.camera_refresh_btn.setObjectName("BlueButton")
        self.camera_connect_btn.setObjectName("BlueButton")
        self.yolo_on_btn.setObjectName("BlueButton")
        self.yolo_off_btn.setObjectName("BlueButton")
        self.camera_state_label.setObjectName("SmallText")
        self.camera_combo.setFixedHeight(28)
        self.camera_combo.setMinimumWidth(150)
        self.camera_refresh_btn.setFixedSize(54, 28)
        self.camera_connect_btn.setFixedSize(54, 28)
        self.yolo_on_btn.setFixedSize(84, 28)
        self.yolo_off_btn.setFixedSize(84, 28)
        tools.addWidget(self.camera_combo)
        tools.addWidget(self.camera_refresh_btn)
        tools.addWidget(self.camera_connect_btn)
        tools.addWidget(self.yolo_on_btn)
        tools.addWidget(self.yolo_off_btn)
        tools.addWidget(self.camera_state_label, 1)
        self.visible_view = CameraView("可见光相机")
        self.visible_view_box = AspectRatioBox(4 / 3)
        self.visible_view_box.set_widget(self.visible_view)
        visible_panel.content_layout.addLayout(tools)
        visible_panel.content_layout.addWidget(self.visible_view_box, 1)
        self.camera_combo.currentIndexChanged.connect(lambda _index: self.reconnect_camera())
        self.camera_refresh_btn.clicked.connect(self.refresh_camera_list)
        self.camera_connect_btn.clicked.connect(self.reconnect_camera)
        self.yolo_on_btn.clicked.connect(lambda: self.set_yolo_enabled(True))
        self.yolo_off_btn.clicked.connect(lambda: self.set_yolo_enabled(False))

        thermal_panel = TechPanel("热成像相机")

        # ════════════════════════════════════════════════════════
        # [扩展接口] 热像仪设备选择 UI
        # 在此处添加: 热像仪模式下拉框 (WiFi/Lepton/模拟) + 串口选择 + 连接按钮
        # 详见 docs/02-扩展模块接口.md §5.3
        # ════════════════════════════════════════════════════════
        if EXTENSIONS_AVAILABLE:
            thermal_tools = QHBoxLayout()
            thermal_tools.setContentsMargins(0, 0, 0, 0)
            thermal_tools.setSpacing(6)
            thermal_tools.addWidget(QLabel("设备:"))
            self.thermal_device_combo = QComboBox()
            self.thermal_device_combo.addItem("WiFi 热像仪", "wifi")
            self.thermal_device_combo.addItem("串口红外 (Lepton)", "lepton")
            self.thermal_device_combo.addItem("远程 Lepton (中继)", "lepton_remote")
            self.thermal_device_combo.addItem("模拟数据", "simulator")
            self.thermal_device_combo.setFixedHeight(28)
            self.thermal_device_combo.setMinimumWidth(130)
            thermal_tools.addWidget(self.thermal_device_combo)
            thermal_tools.addWidget(QLabel("串口:"))
            self.lepton_port_combo = QComboBox()
            self.lepton_port_combo.setFixedHeight(28)
            self.lepton_port_combo.setMinimumWidth(90)
            for dev, desc in _list_serial_ports():
                self.lepton_port_combo.addItem(f"{dev} - {desc}", dev)
            if self.lepton_port_combo.count() == 0:
                self.lepton_port_combo.addItem("无可用串口", "")
            thermal_tools.addWidget(self.lepton_port_combo)
            self.lepton_refresh_btn = QPushButton("刷新")
            self.lepton_refresh_btn.setObjectName("BlueButton")
            self.lepton_refresh_btn.setFixedSize(54, 28)
            self.lepton_refresh_btn.clicked.connect(self._ext_refresh_lepton_ports)
            thermal_tools.addWidget(self.lepton_refresh_btn)
            self.thermal_connect_btn = QPushButton("连接")
            self.thermal_connect_btn.setObjectName("BlueButton")
            self.thermal_connect_btn.setFixedSize(54, 28)
            thermal_tools.addWidget(self.thermal_connect_btn)
            self.thermal_state_label = QLabel("未连接")
            self.thermal_state_label.setObjectName("SmallText")
            self.thermal_state_label.setContextMenuPolicy(Qt.CustomContextMenu)
            self.thermal_state_label.customContextMenuRequested.connect(self._ext_lepton_context_menu)
            thermal_tools.addWidget(self.thermal_state_label, 1)
            thermal_panel.content_layout.addLayout(thermal_tools)

            self.thermal_device_combo.currentIndexChanged.connect(self._ext_on_thermal_device_changed)
            self.thermal_connect_btn.clicked.connect(self._ext_on_thermal_connect)
        # ════════════════════════════════════════════════════════

        self.thermal_view = ThermalWidget()
        self.thermal_view.bind_main_window(self)
        thermal_panel.content_layout.addWidget(self.thermal_view, 1)

        right_stack = QVBoxLayout()
        right_stack.setContentsMargins(0, 0, 0, 0)
        right_stack.setSpacing(12)
        abnormal_panel = TechPanel("巡检异常数据")
        self.abnormal_table = QTableWidget(5, 4)
        self.abnormal_table.setHorizontalHeaderLabels(["点位", "时间", "结果", "操作"])
        self.abnormal_table.verticalHeader().setVisible(False)
        self.abnormal_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        abnormal_panel.content_layout.addWidget(self.abnormal_table)
        right_stack.addWidget(thermal_panel, 58)
        right_stack.addWidget(abnormal_panel, 42)

        top.addWidget(visible_panel, 56)
        top.addLayout(right_stack, 44)
        layout.addLayout(top, 72)

        layout.addWidget(self._control_panel(), 15)
        layout.addWidget(self._system_panel(), 13)
        return col

    def _control_panel(self):
        panel = TechPanel("机器人控制")

        # ════════════════════════════════════════════════════════
        # [扩展接口] STM32 机器人控制串口连接 UI
        # ════════════════════════════════════════════════════════
        if EXTENSIONS_AVAILABLE:
            stm32_row = QHBoxLayout()
            stm32_row.setContentsMargins(0, 0, 0, 0)
            stm32_row.setSpacing(6)
            stm32_row.addWidget(QLabel("控制串口:"))
            self.stm32_port_combo = QComboBox()
            self.stm32_port_combo.setFixedHeight(28)
            self.stm32_port_combo.setMinimumWidth(100)
            for dev, desc in _list_serial_ports():
                self.stm32_port_combo.addItem(f"{dev} - {desc}", dev)
            if self.stm32_port_combo.count() == 0:
                self.stm32_port_combo.addItem("无可用串口", "")
            stm32_row.addWidget(self.stm32_port_combo)
            self.stm32_refresh_btn = QPushButton("刷新")
            self.stm32_refresh_btn.setObjectName("BlueButton")
            self.stm32_refresh_btn.setFixedSize(54, 28)
            self.stm32_refresh_btn.clicked.connect(self._ext_refresh_stm32_ports)
            stm32_row.addWidget(self.stm32_refresh_btn)
            self.stm32_connect_btn = QPushButton("连接")
            self.stm32_connect_btn.setObjectName("BlueButton")
            self.stm32_connect_btn.setFixedSize(54, 28)
            self.stm32_connect_btn.clicked.connect(self._ext_on_stm32_connect)
            stm32_row.addWidget(self.stm32_connect_btn)
            self.stm32_state_label = QLabel("未连接")
            self.stm32_state_label.setObjectName("SmallText")
            stm32_row.addWidget(self.stm32_state_label, 1)
            panel.content_layout.addLayout(stm32_row)
        # ════════════════════════════════════════════════════════

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        panel.content_layout.addLayout(grid)
        buttons = ["关闭本模式", "前进", "停止", "后退", "自动巡检模式", "启动", "暂停", "继续", "停止"]
        for i, text in enumerate(buttons):
            btn = QPushButton(text)
            btn.setObjectName("DangerButton" if text in ["关闭本模式", "停止"] else "BlueButton")
            btn.clicked.connect(lambda checked=False, t=text: self.on_robot_button(t))
            grid.addWidget(btn, i // 5, i % 5)
        self.control_status = QLabel("控制状态: 系统待命")
        self.control_status.setObjectName("SmallText")
        grid.addWidget(self.control_status, 2, 0, 1, 5)
        return panel

    def _system_panel(self):
        panel = TechPanel("系统控制")

        # ════════════════════════════════════════════════════════
        # [扩展接口] MCU 串口连接 UI
        # 右键 MCU 状态标签可切换远程工控机连接模式
        # 详见 docs/02-扩展模块接口.md §2
        # ════════════════════════════════════════════════════════
        if EXTENSIONS_AVAILABLE:
            mcu_row = QHBoxLayout()
            mcu_row.setContentsMargins(0, 0, 0, 0)
            mcu_row.setSpacing(6)
            mcu_row.addWidget(QLabel("MCU串口:"))
            self.mcu_port_combo = QComboBox()
            self.mcu_port_combo.setFixedHeight(28)
            self.mcu_port_combo.setMinimumWidth(100)
            for dev, desc in _list_serial_ports():
                self.mcu_port_combo.addItem(f"{dev} - {desc}", dev)
            if self.mcu_port_combo.count() == 0:
                self.mcu_port_combo.addItem("无可用串口", "")
            mcu_row.addWidget(self.mcu_port_combo)
            self.mcu_refresh_btn = QPushButton("刷新")
            self.mcu_refresh_btn.setObjectName("BlueButton")
            self.mcu_refresh_btn.setFixedSize(54, 28)
            self.mcu_refresh_btn.clicked.connect(self._ext_refresh_mcu_ports)
            mcu_row.addWidget(self.mcu_refresh_btn)
            self.mcu_connect_btn = QPushButton("连接")
            self.mcu_connect_btn.setObjectName("BlueButton")
            self.mcu_connect_btn.setFixedSize(54, 28)
            mcu_row.addWidget(self.mcu_connect_btn)
            self.mcu_state_label = QLabel("未连接")
            self.mcu_state_label.setObjectName("SmallText")
            self.mcu_state_label.setContextMenuPolicy(Qt.CustomContextMenu)
            self.mcu_state_label.customContextMenuRequested.connect(self._ext_mcu_context_menu)
            mcu_row.addWidget(self.mcu_state_label, 1)
            self.mcu_connect_btn.clicked.connect(self._ext_on_mcu_connect)
            panel.content_layout.addLayout(mcu_row)
        # ════════════════════════════════════════════════════════

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        panel.content_layout.addLayout(grid)
        buttons = ["系统急停", "急停复位", "一键巡仓", "避障联锁投入", "避障联锁切图", "全局消音", "语音播报投入", "语音播报切换", "提示语复位"]
        for i, text in enumerate(buttons):
            btn = QPushButton(text)
            if text == "系统急停":
                btn.setObjectName("DangerButton")
            elif text in ["急停复位", "一键巡仓"]:
                btn.setObjectName("GreenButton")
            else:
                btn.setObjectName("BlueButton")
            btn.clicked.connect(lambda checked=False, t=text: self.on_system_button(t))
            grid.addWidget(btn, 0, i)
        return panel

    # ------------------------- 右侧区域 -------------------------
    def _build_right(self):
        col = QWidget()
        layout = QVBoxLayout(col)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(self._stat_panel(), 16)
        layout.addWidget(self._temperature_panel(), 41)
        layout.addWidget(self._env_monitor_panel(), 20)
        layout.addWidget(self._env_data_panel(), 23)
        return col

    def _stat_panel(self):
        panel = TechPanel("安全监测")
        row = QHBoxLayout()
        row.setSpacing(12)
        self.safety_status_circle = StatCircle("安全状态", "正常", GREEN, "✓")
        self.detect_status_circle = StatCircle("检测状态", "待检测", GREEN, "□")
        row.addWidget(self.safety_status_circle)
        row.addWidget(self.detect_status_circle)
        panel.content_layout.addLayout(row, 1)
        return panel

    def _temperature_panel(self):
        panel = TechPanel("温度检测")

        # 主布局由 TechPanel.content_layout 提供，这里明确拉开圆环区和文本区的距离，
        # 避免高温提示压到圆环仪表盘。
        panel.content_layout.setContentsMargins(12, 8, 12, 10)
        panel.content_layout.setSpacing(16)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        self.max_gauge = RingGauge("最高温", YELLOW, 24.6)
        self.min_gauge = RingGauge("最低温", BLUE, 21.5)
        self.avg_gauge = RingGauge("平均温", CYAN, 22.7)
        row.addWidget(self.max_gauge)
        row.addWidget(self.min_gauge)
        row.addWidget(self.avg_gauge)
        gauge_box = QWidget()
        gauge_box.setFixedHeight(100)
        gauge_box.setLayout(row)
        gauge_box.setContextMenuPolicy(Qt.CustomContextMenu)
        gauge_box.customContextMenuRequested.connect(self._show_temp_threshold_menu)

        self.hot_info = QTextEdit()
        self.hot_info.setObjectName("AlertDetail")
        self.hot_info.setReadOnly(True)
        self.hot_info.setLineWrapMode(QTextEdit.WidgetWidth)
        self.hot_info.setText("高温区域数量: 0\n高温点位位置: 无\n过高区域: 无")
        self.hot_info.setFixedHeight(82)
        self.hot_info.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.hot_info.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        panel.content_layout.addWidget(gauge_box)
        panel.content_layout.addSpacing(6)
        panel.content_layout.addWidget(self.hot_info)
        return panel

    def _env_monitor_panel(self):
        panel = TechPanel("环境监测数据")
        panel.content_layout.setContentsMargins(12, 8, 12, 8)
        panel.content_layout.setSpacing(6)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.ch4_gauge = CH4Gauge()
        self.ch4_chart = TrendChart(show_threshold=True)
        row.addWidget(self.ch4_gauge, 35)
        row.addWidget(self.ch4_chart, 65)
        panel.content_layout.addLayout(row, 1)
        return panel

    def _env_data_panel(self):
        panel = TechPanel("环境检测数据")
        panel.content_layout.setContentsMargins(12, 8, 12, 8)
        panel.content_layout.setSpacing(5)
        self.temp_bar = EnvBar("温度", "℃", YELLOW)
        self.humidity_bar = EnvBar("湿度", "%", CYAN)
        self.smoke_bar = EnvBar("烟雾", "ppm", ORANGE)
        self.co_bar = EnvBar("CO", "ppm", GREEN)
        for bar in [self.temp_bar, self.humidity_bar, self.smoke_bar, self.co_bar]:
            panel.content_layout.addWidget(bar)
        panel.content_layout.addStretch(1)
        return panel

    # ------------------------- 实时刷新逻辑 -------------------------
    def update_auxiliary_data(self):
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        self.header.set_time(now)
        self.update_left_status()

    def on_camera_frame(self, frame, index, detection_count, tooltip, detected, detections=None):
        self.current_camera_index = index
        self.sync_camera_combo(index)
        self.update_safety_stats(detections or [], detected)
        source_name = camera_display_name(index)
        frame_height, frame_width = frame.shape[:2]
        frame_size_text = f"分辨率: {frame_width}x{frame_height}"
        detection_text = f"检测目标: {detection_count}" if detected else "检测中"
        self.visible_panel.title_label.setText(f"{self.visible_panel_title}   {detection_text}")
        self.visible_view.set_rgb_image(frame)
        self.camera_state_label.setText(
            f"状态: 已连接 {source_name} | {frame_size_text}"
        )
        self.camera_state_label.setToolTip(tooltip)

    def on_camera_error(self, index, error):
        self.current_camera_index = index
        self.sync_camera_combo(index)
        source_name = camera_display_name(index)
        self.visible_panel.title_label.setText(f"{self.visible_panel_title}   检测暂停")
        self.visible_view.show_text("摄像头无画面")
        self.camera_state_label.setText(f"状态: 连接异常 {source_name}")
        self.camera_state_label.setToolTip(error)

    def update_safety_stats(self, detections, detected):
        if not hasattr(self, "safety_status_circle"):
            return
        # YOLO 关闭时跳过安全检测逻辑，保持"待检测"状态
        if hasattr(self, "camera_worker") and not self.camera_worker.is_yolo_enabled():
            self.safety_status_circle.set_display("安全状态", "正常", GREEN, "✓")
            self.detect_status_circle.set_display("检测状态", "待检测", GREEN, "□")
            self._reset_no_hat_audio_state()
            return

        now = time.monotonic()
        has_no_hat = any(is_no_hat_detection(detection) for detection in detections)

        if has_no_hat:
            if self.no_hat_first_seen is None:
                self.no_hat_first_seen = now
            self.no_hat_last_seen = now
            duration = now - self.no_hat_first_seen
            self.no_hat_alert_active = duration >= NO_HAT_ALERT_SECONDS
        else:
            if self.no_hat_last_seen is not None and now - self.no_hat_last_seen <= NO_HAT_LOST_GRACE_SECONDS:
                duration = now - self.no_hat_first_seen
                self.no_hat_alert_active = duration >= NO_HAT_ALERT_SECONDS
            else:
                self.no_hat_first_seen = None
                self.no_hat_last_seen = None
                duration = 0.0
                self.no_hat_alert_active = False

        if self.no_hat_alert_active:
            self.safety_status_circle.set_display("安全状态", "未戴帽", RED, "!")
            self.detect_status_circle.set_display("检测状态", "异常", RED, "!")
            self._update_no_hat_audio(now)
        else:
            self.safety_status_circle.set_display("安全状态", "正常", GREEN, "✓")
            status_text = "检测中" if detected else "待检测"
            self.detect_status_circle.set_display("检测状态", status_text, GREEN, "□")
            self._reset_no_hat_audio_state()

    def _init_no_hat_audio(self):
        if not self.no_hat_audio_path.exists() or QMediaPlayer is None:
            return
        try:
            if QT_LIB == "PySide6":
                if QAudioOutput is None:
                    return
                self.no_hat_audio_player = QMediaPlayer(self)
                self.no_hat_audio_output = QAudioOutput(self)
                self.no_hat_audio_player.setAudioOutput(self.no_hat_audio_output)
                self.no_hat_audio_output.setVolume(1.0)
                self.no_hat_audio_player.setSource(QUrl.fromLocalFile(str(self.no_hat_audio_path)))
                self.no_hat_audio_player.mediaStatusChanged.connect(self._on_no_hat_audio_status_changed)
            else:
                self.no_hat_audio_player = QMediaPlayer(self)
                self.no_hat_audio_player.setMedia(QMediaContent(QUrl.fromLocalFile(str(self.no_hat_audio_path))))
                self.no_hat_audio_player.mediaStatusChanged.connect(self._on_no_hat_audio_status_changed)
                try:
                    self.no_hat_audio_player.setVolume(100)
                except Exception:
                    pass
            self.no_hat_audio_available = True
        except Exception as exc:
            print(f"[Audio] 初始化未戴帽告警音失败: {exc}")
            self.no_hat_audio_player = None
            self.no_hat_audio_output = None
            self.no_hat_audio_available = False

    def _update_no_hat_audio(self, now):
        if not self.no_hat_audio_available or self.no_hat_audio_player is None:
            return
        self.no_hat_audio_repeat_requested = True
        if self.no_hat_audio_playing:
            return
        if now < self.no_hat_audio_cooldown_until:
            return
        self._play_no_hat_audio()

    def _play_no_hat_audio(self):
        if not self.no_hat_audio_available or self.no_hat_audio_player is None:
            return
        try:
            self.no_hat_audio_playing = True
            self.no_hat_audio_player.stop()
            self.no_hat_audio_player.setPosition(0)
            self.no_hat_audio_player.play()
        except Exception as exc:
            print(f"[Audio] 播放未戴帽告警音失败: {exc}")
            self.no_hat_audio_playing = False
            self.no_hat_audio_cooldown_until = time.monotonic() + NO_HAT_ALERT_SECONDS

    def _on_no_hat_audio_status_changed(self, status):
        if not self.no_hat_audio_player:
            return
        end_status = getattr(QMediaPlayer, "EndOfMedia", None)
        media_status_enum = getattr(QMediaPlayer, "MediaStatus", None)
        if media_status_enum is not None and hasattr(media_status_enum, "EndOfMedia"):
            end_status = media_status_enum.EndOfMedia
        if status == end_status:
            self.no_hat_audio_playing = False
            self.no_hat_audio_cooldown_until = (
                time.monotonic() + NO_HAT_ALERT_SECONDS
                if self.no_hat_audio_repeat_requested else 0.0
            )

    def _reset_no_hat_audio_state(self):
        self.no_hat_audio_repeat_requested = False
        self.no_hat_audio_cooldown_until = 0.0

    def update_thermal(self):
        # ════════════════════════════════════════════════════════
        # [扩展接口] 热像仪帧获取 + 热区追踪 + 报警检测
        # 在此处调用 ThermalCameraManager.get_frame() 获取真实热成像数据
        # 然后调用 ThermalZoneTracker.update() 追踪热区
        # 再调用 AlarmController.check() 检测报警
        # 详见 docs/02-扩展模块接口.md §5.4
        # ════════════════════════════════════════════════════════
        if EXTENSIONS_AVAILABLE and hasattr(self, 'thermal_camera_mgr'):
            frame = self.thermal_camera_mgr.get_frame()
            if frame is not None:
                # 热区追踪
                self.hot_regions = self.zone_tracker.update(frame.temps)
                # 叠加绘制
                annotated_rgb = self.zone_tracker.draw_overlay(frame.rgb)
                self.thermal_view.set_rgb_image(annotated_rgb)
                # 温度仪表盘
                self.update_temperature_alarm(frame.max_temp, frame.min_temp, frame.avg_temp)
                # 报警检测
                self.ext_alarm_state = self.alarm_ctrl.check(frame.max_temp)
                # 高温信息
                self.hot_info.setText(self.zone_tracker.get_hot_info_text())
                self.update_abnormal_table()
                self.hot_info.setToolTip("")
                return
        # ════════════════════════════════════════════════════════

        self.hot_regions = []
        self.thermal_view.show_text("热成像未接入")
        self.update_temperature_alarm(0.0, 0.0, 0.0)
        self.hot_info.setText(
            f"高温区域: 0\n"
            f"预警阈值: {self.threshold_warn:.0f}℃ / 报警阈值: {self.threshold_alarm:.0f}℃\n"
            f"右键温度仪表盘可修改阈值"
        )
        self.hot_info.setToolTip("")
        self.update_abnormal_table()

    def update_temperature_alarm(self, max_temp, min_temp, avg_temp):
        max_level, max_color = temperature_alarm_state(max_temp, self.threshold_warn, self.threshold_alarm)
        _min_level, min_color = temperature_alarm_state(min_temp, self.threshold_warn, self.threshold_alarm)
        _avg_level, avg_color = temperature_alarm_state(avg_temp, self.threshold_warn, self.threshold_alarm)

        self.max_gauge.set_value(max_temp)
        self.min_gauge.set_value(min_temp)
        self.avg_gauge.set_value(avg_temp)
        self.max_gauge.set_color(max_color)
        self.min_gauge.set_color(min_color)
        self.avg_gauge.set_color(avg_color)

        # 3级报警: 0=正常(绿), 1=二级预警(黄闪), 2=三级报警(红闪)
        _level_map = {"正常": 0, "二级预警": 1, "三级报警": 2}
        self.max_gauge.set_alarm_level(_level_map.get(max_level, 0))
        self.min_gauge.set_alarm_level(_level_map.get(_min_level, 0))
        self.avg_gauge.set_alarm_level(_level_map.get(_avg_level, 0))

        self.hot_info.setText(
            f"当前预警等级: {max_level}\n"
            f"最高温度: {max_temp:.1f}℃\n"
            f"预警阈值: {self.threshold_warn:.0f}℃ / 报警阈值: {self.threshold_alarm:.0f}℃\n"
            f"<{self.threshold_warn:.0f}℃绿色 / "
            f"≥{self.threshold_warn:.0f}℃黄色闪烁 / "
            f"≥{self.threshold_alarm:.0f}℃红色闪烁\n"
            f"右键温度仪表盘可修改阈值"
        )

    def update_abnormal_table(self):
        self.abnormal_table.setRowCount(5)
        for row in range(5):
            if row < len(self.hot_regions):
                region = self.hot_regions[row]
                values = [f"P{row + 1:02d}", time.strftime("%H:%M:%S"), f"{region.level} {region.temp:.1f}℃", "查看"]
            else:
                values = ["--", "--", "--", "--"]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(align_center())
                if row < len(self.hot_regions) and col == 2:
                    item.setForeground(QColor(RED))
                self.abnormal_table.setItem(row, col, item)

    def update_left_status(self):
        t = time.time()
        self.battery_label.setText(f"电池电量: {64.38 + math.sin(t / 8) * 0.5:.2f}%")
        self.voltage_label.setText(f"电池电压: {49.40 + math.sin(t / 7) * 0.2:.2f}V")
        self.cell_temp_label.setText(f"电芯温度: {22.00 + math.sin(t / 6) * 0.4:.2f}℃")
        self.cabin_temp_label.setText(f"机舱温度: {22.43 + math.sin(t / 5) * 0.4:.2f}℃")

    def update_environment(self):
        # ════════════════════════════════════════════════════════
        # [扩展接口] MCU 传感器数据读取
        # 在此处调用 MCUSerialBridge.get_sensor_data() 获取真实传感器数据
        # 详见 docs/02-扩展模块接口.md §2
        # ════════════════════════════════════════════════════════
        if EXTENSIONS_AVAILABLE and hasattr(self, 'mcu_bridge') and self.mcu_bridge.is_connected():
            sensor = self.mcu_bridge.get_sensor_data()
            if sensor.valid:
                env_temp = sensor.dht11_temp
                humidity = sensor.dht11_humi
                smoke = float(sensor.smoke)
                co = 18.0  # CO 暂无传感器, 保留模拟
                ch4 = 0.42  # CH4 暂无传感器, 保留模拟
                self.temp_bar.set_value(env_temp, 60)
                self.humidity_bar.set_value(humidity, 100)
                self.smoke_bar.set_value(smoke, 1023)
                self.co_bar.set_value(co, 100)
                self.ch4_gauge.set_value(ch4)
                self.ch4_chart.push(ch4 * 18, 28.0)
                return
        # ════════════════════════════════════════════════════════

        env_temp = 22.90
        humidity = 43.70
        ch4 = 0.42
        smoke = 12.0
        co = 18.0
        self.temp_bar.set_value(env_temp, 60)
        self.humidity_bar.set_value(humidity, 100)
        self.smoke_bar.set_value(smoke, 100)
        self.co_bar.set_value(co, 100)
        self.ch4_gauge.set_value(ch4)
        self.ch4_chart.values_a = [ch4 * 18] * 60
        self.ch4_chart.values_b = [28.0] * 60
        self.ch4_chart.update()

    # ------------------------- 交互逻辑 -------------------------
    def refresh_camera_list(self):
        was_blocked = self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        self.camera_combo.addItem("网络摄像头", NETWORK_CAMERA_URL)
        local_indices = discover_local_camera_indices(6)
        if not isinstance(self.current_camera_index, str) and self.current_camera_index not in local_indices:
            local_indices.append(int(self.current_camera_index))
        for index in sorted(local_indices):
            text = f"本地摄像头 {index}"
            if index == LOCAL_CAMERA_INDEX:
                text += "（备用）"
            self.camera_combo.addItem(text, index)
        if isinstance(self.current_camera_index, str):
            self.camera_combo.setCurrentIndex(0)
        else:
            target = self.camera_combo.findData(self.current_camera_index)
            self.camera_combo.setCurrentIndex(target if target >= 0 else 0)
        self.camera_combo.blockSignals(was_blocked)

    def reconnect_camera(self):
        if not hasattr(self, "camera_worker"):
            return
        index = self.camera_combo.currentData()
        self.current_camera_index = index
        self.camera_worker.request_open(index)
        self.visible_panel.title_label.setText(f"{self.visible_panel_title}   检测中")
        self.camera_state_label.setText(f"正在连接: {camera_display_name(index)}")
        self.camera_state_label.setToolTip("")

    def set_yolo_enabled(self, enabled):
        if not hasattr(self, "camera_worker"):
            return
        self.camera_worker.set_yolo_enabled(enabled)
        self._sync_yolo_buttons(enabled)

    def _sync_yolo_buttons(self, enabled):
        if not hasattr(self, "yolo_on_btn") or not hasattr(self, "yolo_off_btn"):
            return
        self.yolo_on_btn.setEnabled(not enabled)
        self.yolo_off_btn.setEnabled(enabled)
        # 关闭 YOLO 时重置安全状态
        if not enabled and hasattr(self, "safety_status_circle"):
            self.safety_status_circle.set_display("安全状态", "正常", GREEN, "✓")
            self.detect_status_circle.set_display("检测状态", "待检测", GREEN, "□")
            self._reset_no_hat_audio_state()

    def sync_camera_combo(self, index):
        if not hasattr(self, "camera_combo"):
            return
        if isinstance(index, str):
            target = 0
        else:
            target = self.camera_combo.findData(index)
            if target < 0:
                return
        if self.camera_combo.currentIndex() != target:
            was_blocked = self.camera_combo.blockSignals(True)
            self.camera_combo.setCurrentIndex(target)
            self.camera_combo.blockSignals(was_blocked)

    def on_robot_button(self, text):
        self.control_state = text
        self.control_status.setText(f"控制状态: {text}")
        # 使用扩展模块的 STM32 控制器 (如果可用), 否则用旧的占位
        stm32 = self.ext_stm32 if (EXTENSIONS_AVAILABLE and hasattr(self, 'ext_stm32')) else self.stm32
        if text == "前进":
            stm32.send_command("F:1,S:0")
        elif text == "后退":
            stm32.send_command("F:0,S:1")
        elif text == "停止":
            stm32.send_command("F:0,S:0")

    def on_system_button(self, text):
        self.control_status.setText(f"系统状态: {text}")

    def closeEvent(self, event):
        self.camera_worker.stop()
        # ════════════════════════════════════════════════════════
        # [扩展接口] 清理扩展模块资源
        # 详见 docs/02-扩展模块接口.md §5.5
        # ════════════════════════════════════════════════════════
        if EXTENSIONS_AVAILABLE:
            if hasattr(self, 'thermal_camera_mgr'):
                self.thermal_camera_mgr.disconnect_all()
            if hasattr(self, 'mcu_bridge'):
                self.mcu_bridge.disconnect()
            if hasattr(self, 'ext_stm32'):
                self.ext_stm32.disconnect()
        # ════════════════════════════════════════════════════════
        event.accept()

    # ════════════════════════════════════════════════════════════
    # [扩展接口] 扩展模块交互方法
    # 以下方法供扩展模块 UI 控件的信号连接使用
    # 详见 docs/02-扩展模块接口.md §5
    # ════════════════════════════════════════════════════════════
    def _ext_on_thermal_device_changed(self, index):
        """热像仪设备切换回调。"""
        if not EXTENSIONS_AVAILABLE:
            return
        device = self.thermal_device_combo.currentData()
        if device:
            self.thermal_camera_mgr.set_mode(device)
            self.thermal_state_label.setText(f"切换到: {self.thermal_device_combo.currentText()}")

    def _ext_on_thermal_connect(self):
        """热像仪连接按钮回调。"""
        if not EXTENSIONS_AVAILABLE:
            return
        device = self.thermal_device_combo.currentData()
        if device == "wifi":
            self.thermal_state_label.setText("WiFi 连接中...")
            self.thermal_connect_btn.setEnabled(False)
            threading.Thread(target=self._ext_connect_wifi_async, daemon=True).start()
        elif device == "lepton":
            port = self.lepton_port_combo.currentData()
            if not port:
                self.thermal_state_label.setText("请选择串口")
                return
            self.thermal_state_label.setText("Lepton 连接中...")
            self.thermal_connect_btn.setEnabled(False)
            threading.Thread(target=self._ext_connect_lepton_async, args=(port,), daemon=True).start()
        elif device == "lepton_remote":
            # 远程 Lepton: 先检查是否已建立 TCP 连接
            lepton = self.thermal_camera_mgr._lepton
            if lepton.get_mode() == 'remote' and lepton.is_connected():
                # TCP 已连接，点击按钮是要打开或关闭串口
                port = self.lepton_port_combo.currentData()
                if not port:
                    self.thermal_state_label.setText("请选择串口")
                    return
                if getattr(lepton, '_remote_port_opened', False):
                    # 关闭串口但保持 TCP
                    lepton._remote_port_opened = False
                    self.thermal_state_label.setText("远程 Lepton 串口已关闭")
                    self.thermal_connect_btn.setText("连接")
                    self.lepton_port_combo.setEnabled(True)
                    self.lepton_refresh_btn.setEnabled(True)
                    self._ext_refresh_lepton_ports()
                else:
                    # 打开串口
                    self.thermal_state_label.setText("连接中...")
                    self.thermal_connect_btn.setEnabled(False)
                    QApplication.processEvents()
                    ok = lepton.open_remote_port(port)
                    if ok:
                        lepton._remote_port_opened = True
                        info = lepton.get_connection_info()
                        self.thermal_state_label.setText(f"远程 Lepton {port} 已打开")
                        self.thermal_connect_btn.setText("断开")
                        self.lepton_port_combo.setEnabled(False)
                        self.lepton_refresh_btn.setEnabled(False)
                    else:
                        self.thermal_state_label.setText(f"远程串口 {port} 打开失败")
                    self.thermal_connect_btn.setEnabled(True)
            else:
                # TCP 未连接，提示用户通过右键菜单连接
                self.thermal_state_label.setText("请右键连接远程工控机")
        else:
            self.thermal_state_label.setText("模拟模式")

    def _ext_connect_wifi_async(self):
        ok = self.thermal_camera_mgr.connect_wifi()
        QTimer.singleShot(0, lambda: self._finish_thermal_connect(
            "WiFi 已连接" if ok else "WiFi 连接失败"))

    def _ext_connect_lepton_async(self, port):
        ok = self.thermal_camera_mgr.connect_lepton(port)
        QTimer.singleShot(0, lambda: self._finish_thermal_connect(
            f"Lepton {port} 已连接" if ok else "Lepton 连接失败"))

    def _finish_thermal_connect(self, text):
        self.thermal_state_label.setText(text)
        self.thermal_connect_btn.setEnabled(True)

    def _ext_on_mcu_connect(self):
        """MCU 串口连接/断开回调。根据模式自动判断本地/远程。"""
        if not EXTENSIONS_AVAILABLE:
            return

        mode = self.mcu_bridge.get_mode()
        port = self.mcu_port_combo.currentData()

        # --- 远程模式: TCP 已连接，点击按钮是要打开/关闭工控机上的串口 ---
        if mode == 'remote' and self.mcu_bridge.is_connected():
            if not port:
                self.mcu_state_label.setText("请选择串口")
                return

            # 检查是否已经打开了串口（通过 _remote_port_opened 标记）
            # 如果已打开，则断开串口（不断开 TCP 连接）
            if getattr(self.mcu_bridge, '_remote_port_opened', False):
                # 关闭远程串口
                self.mcu_bridge._remote_port_opened = False
                self.mcu_state_label.setText(f"远程: {self.mcu_bridge._remote_host} — 串口已关闭")
                self.mcu_connect_btn.setText("连接")
                self.mcu_port_combo.setEnabled(True)
                self.mcu_refresh_btn.setEnabled(True)
                return

            # 远程 TCP 已连接但串口未打开 → 打开串口
            self.mcu_state_label.setText("连接中...")
            QApplication.processEvents()
            ok = self.mcu_bridge.open_remote_port(port)
            if ok:
                self.mcu_bridge._remote_port_opened = True
                info = self.mcu_bridge.get_connection_info()
                host = info.get('host', '?')
                self.mcu_state_label.setText(f"远程 {host} | MCU {port}")
                self.mcu_connect_btn.setText("断开")
                self.mcu_port_combo.setEnabled(False)
                self.mcu_refresh_btn.setEnabled(False)
            else:
                self.mcu_state_label.setText(f"远程串口 {port} 打开失败")
            return

        # --- 本地模式 或 远程 TCP 未连接 ---

        # 断开逻辑（本地串口 或 远程 TCP）
        if self.mcu_bridge.is_connected():
            self.mcu_bridge.disconnect()
            self.mcu_state_label.setText("已断开")
            self.mcu_connect_btn.setText("连接")
            self.mcu_port_combo.setEnabled(True)
            self.mcu_refresh_btn.setEnabled(True)
            # 恢复显示本地串口列表
            self._ext_refresh_mcu_ports()
            return

        # --- 连接逻辑 ---
        if not port:
            self.mcu_state_label.setText("请选择串口")
            return

        self.mcu_state_label.setText("连接中...")
        QApplication.processEvents()

        if mode == 'remote':
            # 远程模式: 先建立 TCP 连接，再打开串口
            # 注意：这里 port 是 TCP 端口，不是串口
            # 但从 UI 看，用户选择的是串口，所以实际上是走 open_remote_port
            # 实际上 open_remote_port 需要先有 TCP 连接才能用
            # 所以这里其实不会被走到，因为上面已经处理了 remote + connected 的情况
            self.mcu_state_label.setText("远程连接已断开，请重试")
        else:
            # 本地模式: 打开本地串口
            ok = self.mcu_bridge.connect(port)
            self.mcu_state_label.setText(f"MCU {port} 已连接" if ok else "MCU 连接失败")
            self.mcu_connect_btn.setText("断开" if ok else "连接")
            if ok:
                self.mcu_port_combo.setEnabled(False)
                self.mcu_refresh_btn.setEnabled(False)

    def _ext_mcu_context_menu(self, pos):
        """MCU 串口状态标签右键菜单 — 远程工控机连接。"""
        if not EXTENSIONS_AVAILABLE:
            return
        menu = QMenu(self)
        mode = self.mcu_bridge.get_mode()

        if mode == 'remote' and self.mcu_bridge.is_connected():
            info = self.mcu_bridge.get_connection_info()
            info_action = menu.addAction(f"远程工控机: {info.get('host', '?')}:{info.get('port', '?')}")
            info_action.setEnabled(False)
            menu.addSeparator()
            menu.addAction("刷新远程串口列表", self._ext_refresh_mcu_ports)
            menu.addAction("断开远程连接", self._ext_disconnect_remote)
        elif mode == 'local' and self.mcu_bridge.is_connected():
            info_action = menu.addAction(f"本地串口: {self.mcu_bridge.port_name}")
            info_action.setEnabled(False)
            menu.addSeparator()
            disconnect_action = menu.addAction("断开本地连接")
            disconnect_action.triggered.connect(self._ext_on_mcu_connect)
        else:
            menu.addAction("连接远程工控机...", self._ext_connect_remote_dialog)
            menu.addSeparator()
            refresh_action = menu.addAction("刷新本地串口列表")
            refresh_action.triggered.connect(self._ext_refresh_mcu_ports)

        menu.exec_(self.mcu_state_label.mapToGlobal(pos))

    def _ext_connect_remote_dialog(self):
        """弹出远程工控机连接对话框, 连接后自动刷新串口列表为远程端口。"""
        if not EXTENSIONS_AVAILABLE:
            return

        # 从 config.json 读取默认值
        default_host = "192.168.3.100"
        default_port = "6001"
        try:
            config_path = Path(__file__).parent / "config.json"
            if config_path.exists():
                import json as _json
                with open(config_path, 'r', encoding='utf-8') as f:
                    cfg = _json.load(f)
                relay_cfg = cfg.get("sensor_relay", {})
                default_host = relay_cfg.get("host", default_host)
                default_port = str(relay_cfg.get("port", 6001))
        except Exception:
            pass

        # 输入 IP
        host, ok1 = QInputDialog.getText(
            self, "连接远程工控机", "工控机 IP 地址:",
            QLineEdit.Normal, default_host
        )
        if not ok1 or not host.strip():
            return

        # 输入端口
        port_str, ok2 = QInputDialog.getText(
            self, "连接远程工控机", "TCP 端口:",
            QLineEdit.Normal, default_port
        )
        if not ok2 or not port_str.strip():
            return

        try:
            port = int(port_str.strip())
        except ValueError:
            QMessageBox.warning(self, "错误", "端口号必须是数字")
            return

        self.mcu_state_label.setText(f"连接 {host.strip()}:{port}...")
        QApplication.processEvents()

        ok = self.mcu_bridge.open_remote(host.strip(), port)
        if ok:
            self.mcu_bridge._remote_port_opened = False  # 重置串口打开标记
            self.mcu_state_label.setText(f"远程: {host.strip()}:{port} — 请选择串口")
            # 自动刷新下拉框为远程工控机的串口列表
            self._ext_refresh_mcu_ports()
        else:
            self.mcu_state_label.setText("远程连接失败")

    def _ext_disconnect_remote(self):
        """断开远程工控机连接, 恢复本地串口列表。"""
        if not EXTENSIONS_AVAILABLE:
            return
        self.mcu_bridge._remote_port_opened = False  # 重置串口打开标记
        self.mcu_bridge.disconnect()
        self.mcu_state_label.setText("已断开")
        self.mcu_connect_btn.setText("连接")
        self.mcu_port_combo.setEnabled(True)
        self.mcu_refresh_btn.setEnabled(True)
        # 恢复显示本地串口列表
        self._ext_refresh_mcu_ports()

    def _ext_refresh_lepton_ports(self):
        """刷新 Lepton 串口列表。远程模式下查询工控机串口，本地模式下查询本机串口。"""
        if not EXTENSIONS_AVAILABLE:
            return
        self.lepton_port_combo.clear()
        lepton = self.thermal_camera_mgr._lepton
        if lepton.get_mode() == 'remote' and lepton.is_connected():
            # 远程模式: 查询工控机上的串口
            remote_ports = lepton.list_remote_ports()
            for dev in remote_ports:
                self.lepton_port_combo.addItem(dev, dev)
            if self.lepton_port_combo.count() == 0:
                self.lepton_port_combo.addItem("无可用串口 (远程)", "")
        else:
            # 本地模式: 查询本机串口
            for dev, desc in _list_serial_ports():
                self.lepton_port_combo.addItem(f"{dev} - {desc}", dev)
            if self.lepton_port_combo.count() == 0:
                self.lepton_port_combo.addItem("无可用串口", "")

    def _ext_lepton_context_menu(self, pos):
        """Lepton 串口状态标签右键菜单 — 远程工控机连接。"""
        if not EXTENSIONS_AVAILABLE:
            return
        menu = QMenu(self)
        lepton = self.thermal_camera_mgr._lepton
        mode = lepton.get_mode()

        if mode == 'remote' and lepton.is_connected():
            info = lepton.get_connection_info()
            info_action = menu.addAction(f"远程工控机: {info.get('host', '?')}:{info.get('port', '?')}")
            info_action.setEnabled(False)
            menu.addSeparator()
            refresh_action = menu.addAction("刷新远程串口列表")
            refresh_action.triggered.connect(self._ext_refresh_lepton_ports)
            disconnect_action = menu.addAction("断开远程连接")
            disconnect_action.triggered.connect(self._ext_disconnect_lepton_remote)
        elif mode == 'local' and lepton.is_connected():
            info_action = menu.addAction(f"本地串口: {lepton.port_name}")
            info_action.setEnabled(False)
            menu.addSeparator()
            disconnect_action = menu.addAction("断开本地连接")
            disconnect_action.triggered.connect(lambda: self._ext_on_thermal_device_changed(1))
        else:
            menu.addAction("连接远程工控机...", self._ext_connect_lepton_remote_dialog)
            menu.addSeparator()
            refresh_action = menu.addAction("刷新本地串口列表")
            refresh_action.triggered.connect(self._ext_refresh_lepton_ports)

        menu.exec_(self.thermal_state_label.mapToGlobal(pos))

    def _ext_connect_lepton_remote_dialog(self):
        """弹出远程 Lepton 中继连接对话框。"""
        if not EXTENSIONS_AVAILABLE:
            return

        # 从 config.json 读取默认值
        default_host = "192.168.3.100"
        default_port = "6002"
        try:
            config_path = Path(__file__).parent / "config.json"
            if config_path.exists():
                import json as _json
                with open(config_path, 'r', encoding='utf-8') as f:
                    cfg = _json.load(f)
                relay_cfg = cfg.get("lepton_relay", {})
                default_host = relay_cfg.get("host", default_host)
                default_port = str(relay_cfg.get("port", 6002))
        except Exception:
            pass

        host, ok1 = QInputDialog.getText(
            self, "连接远程 Lepton 中继", "工控机 IP 地址:",
            QLineEdit.Normal, default_host
        )
        if not ok1 or not host.strip():
            return

        port_str, ok2 = QInputDialog.getText(
            self, "连接远程 Lepton 中继", "TCP 端口:",
            QLineEdit.Normal, default_port
        )
        if not ok2 or not port_str.strip():
            return

        try:
            port = int(port_str.strip())
        except ValueError:
            QMessageBox.warning(self, "错误", "端口号必须是数字")
            return

        self.thermal_state_label.setText(f"连接 {host.strip()}:{port}...")
        self.thermal_connect_btn.setEnabled(False)
        QApplication.processEvents()

        ok = self.thermal_camera_mgr.connect_lepton_remote(host.strip(), port)
        if ok:
            self.thermal_state_label.setText(f"远程: {host.strip()}:{port} — 请选择串口")
            self._ext_refresh_lepton_ports()
        else:
            self.thermal_state_label.setText("远程连接失败")
        self.thermal_connect_btn.setEnabled(True)

    def _ext_disconnect_lepton_remote(self):
        """断开远程 Lepton 连接。"""
        if not EXTENSIONS_AVAILABLE:
            return
        lepton = self.thermal_camera_mgr._lepton
        lepton._remote_port_opened = False
        lepton.disconnect()
        self.thermal_state_label.setText("已断开")
        self.thermal_connect_btn.setText("连接")
        self._ext_refresh_lepton_ports()

    def _ext_refresh_mcu_ports(self):
        """刷新 MCU 串口列表。远程模式下查询工控机串口, 本地模式下查询本机串口。"""
        if not EXTENSIONS_AVAILABLE:
            return
        self.mcu_port_combo.clear()

        mode = self.mcu_bridge.get_mode()
        if mode == 'remote' and self.mcu_bridge.is_connected():
            # 远程模式: 查询工控机上的串口
            remote_ports = self.mcu_bridge.list_remote_ports()
            for dev in remote_ports:
                self.mcu_port_combo.addItem(dev, dev)
            if self.mcu_port_combo.count() == 0:
                self.mcu_port_combo.addItem("无可用串口 (远程)", "")
        else:
            # 本地模式: 查询本机串口
            for dev, desc in _list_serial_ports():
                self.mcu_port_combo.addItem(f"{dev} - {desc}", dev)
            if self.mcu_port_combo.count() == 0:
                self.mcu_port_combo.addItem("无可用串口", "")

    def _ext_on_stm32_connect(self):
        """STM32 串口连接/断开回调。"""
        if not EXTENSIONS_AVAILABLE:
            return
        port = self.stm32_port_combo.currentData()
        if not port:
            self.stm32_state_label.setText("请选择串口")
            return
        if self.ext_stm32.is_connected():
            self.ext_stm32.disconnect()
            self.stm32_state_label.setText("已断开")
            self.stm32_connect_btn.setText("连接")
            return
        self.stm32_state_label.setText("连接中...")
        QApplication.processEvents()
        ok = self.ext_stm32.connect(port)
        self.stm32_state_label.setText(f"STM32 {port} 已连接" if ok else "STM32 连接失败")
        self.stm32_connect_btn.setText("断开" if ok else "连接")

    def _ext_refresh_stm32_ports(self):
        """刷新 STM32 串口列表。"""
        if not EXTENSIONS_AVAILABLE:
            return
        self.stm32_port_combo.clear()
        for dev, desc in _list_serial_ports():
            self.stm32_port_combo.addItem(f"{dev} - {desc}", dev)
        if self.stm32_port_combo.count() == 0:
            self.stm32_port_combo.addItem("无可用串口", "")

    def _show_temp_threshold_menu(self, pos):
        """温度仪表盘区域右键菜单 — 设置3级报警阈值。"""
        menu = QMenu(self)
        info1 = menu.addAction(f"二级预警(黄闪): ≥{self.threshold_warn:.0f}℃")
        info1.setEnabled(False)
        info2 = menu.addAction(f"三级报警(红闪): ≥{self.threshold_alarm:.0f}℃")
        info2.setEnabled(False)
        menu.addSeparator()
        menu.addAction("设置阈值...", self._show_threshold_dialog)
        menu.exec_(self.sender().mapToGlobal(pos))

    def _show_threshold_dialog(self):
        """3级报警阈值设置对话框。修改后同步到 AlarmController, 超温时下发 $ALARM 到下位机。"""
        dialog = QDialog(self)
        dialog.setWindowTitle("报警阈值设置")
        dialog.setMinimumWidth(300)

        form = QVBoxLayout(dialog)
        form_layout = QFormLayout()

        # 二级预警阈值 (黄闪)
        warn_spin = QDoubleSpinBox()
        warn_spin.setRange(20.0, 200.0)
        warn_spin.setDecimals(1)
        warn_spin.setSuffix(" ℃")
        warn_spin.setValue(self.threshold_warn)
        warn_spin.setToolTip("温度 ≥ 此值时仪表盘黄色闪烁 (二级预警)")
        form_layout.addRow("二级预警阈值(黄闪):", warn_spin)

        # 三级报警阈值 (红闪)
        alarm_spin = QDoubleSpinBox()
        alarm_spin.setRange(20.0, 200.0)
        alarm_spin.setDecimals(1)
        alarm_spin.setSuffix(" ℃")
        alarm_spin.setValue(self.threshold_alarm)
        alarm_spin.setToolTip("温度 ≥ 此值时仪表盘红色闪烁, 并向下发 $ALARM (三级报警)")
        form_layout.addRow("三级报警阈值(红闪):", alarm_spin)

        form.addLayout(form_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addWidget(buttons)

        if dialog.exec_() == QDialog.Accepted:
            new_warn = warn_spin.value()
            new_alarm = alarm_spin.value()
            if new_warn >= new_alarm:
                QMessageBox.warning(self, "参数错误", "预警阈值必须小于报警阈值")
                return
            self.threshold_warn = new_warn
            self.threshold_alarm = new_alarm
            # 同步到 AlarmController (超温时下发 $ALARM 到下位机)
            if EXTENSIONS_AVAILABLE and hasattr(self, 'alarm_ctrl'):
                self.alarm_ctrl.set_threshold(new_alarm)
            # 同步到 ThermalZoneTracker (热像仪画面圈出热区的检测阈值)
            if EXTENSIONS_AVAILABLE and hasattr(self, 'zone_tracker'):
                self.zone_tracker.set_threshold(new_alarm)
            self.hot_info.setText(
                f"阈值已更新: 预警≥{new_warn:.0f}℃(黄闪) / 报警≥{new_alarm:.0f}℃(红闪)"
            )

    def _show_zone_settings_menu(self, global_pos):
        """热像仪画面右键菜单 — 热区追踪参数设置。"""
        menu = QMenu(self)
        if EXTENSIONS_AVAILABLE and hasattr(self, 'zone_tracker'):
            params = self.zone_tracker.get_params()
            info = menu.addAction(f"检测阈值: {params['threshold']:.0f}℃ | 最大区域: {params['max_regions']}")
            info.setEnabled(False)
        else:
            info = menu.addAction("热区追踪未启用")
            info.setEnabled(False)
        menu.addSeparator()
        menu.addAction("热区追踪参数...", self._show_zone_settings_dialog)
        menu.exec_(global_pos)

    def _show_thermal_pop_window(self):
        """弹出独立热成像详情窗口（右键"详情窗口"）。"""
        if hasattr(self, '_thermal_pop_window') and self._thermal_pop_window is not None:
            self._thermal_pop_window.activateWindow()
            self._thermal_pop_window.raise_()
            return
        self._thermal_pop_window = ThermalPopWindow(
            self.thermal_camera_mgr,
            self.zone_tracker,
            self.alarm_ctrl,
            parent=self
        )
        self._thermal_pop_window.destroyed.connect(lambda _: setattr(self, '_thermal_pop_window', None))
        self._thermal_pop_window.show()

    def _show_zone_settings_dialog(self):
        """热区追踪参数设置对话框。"""
        if not EXTENSIONS_AVAILABLE or not hasattr(self, 'zone_tracker'):
            return

        params = self.zone_tracker.get_params()
        dialog = QDialog(self)
        dialog.setWindowTitle("热区追踪参数设置")
        dialog.setMinimumWidth(320)

        form = QVBoxLayout(dialog)
        form_layout = QFormLayout()

        # 检测阈值
        threshold_spin = QDoubleSpinBox()
        threshold_spin.setRange(30.0, 200.0)
        threshold_spin.setDecimals(1)
        threshold_spin.setSuffix(" ℃")
        threshold_spin.setValue(params['threshold'])
        threshold_spin.setToolTip("温度超过此值的区域将被标记为热区")
        form_layout.addRow("检测阈值:", threshold_spin)

        # 最大追踪区域数
        max_regions_spin = QSpinBox()
        max_regions_spin.setRange(1, 20)
        max_regions_spin.setValue(params['max_regions'])
        max_regions_spin.setToolTip("同时追踪的最大高温区域数量")
        form_layout.addRow("最大区域数:", max_regions_spin)

        # 区域合并距离
        merge_spin = QSpinBox()
        merge_spin.setRange(3, 30)
        merge_spin.setValue(params['merge_distance'])
        merge_spin.setSuffix(" 像素")
        merge_spin.setToolTip("距离小于此值的相邻热区将合并为一个")
        form_layout.addRow("合并距离:", merge_spin)

        # 轨迹历史长度
        track_spin = QSpinBox()
        track_spin.setRange(5, 100)
        track_spin.setValue(params['track_history'])
        track_spin.setSuffix(" 帧")
        track_spin.setToolTip("每个热区保留的移动轨迹点数")
        form_layout.addRow("轨迹长度:", track_spin)

        form.addLayout(form_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addWidget(buttons)

        if dialog.exec_() == QDialog.Accepted:
            self.zone_tracker.set_threshold(threshold_spin.value())
            self.zone_tracker.set_max_regions(max_regions_spin.value())
            self.zone_tracker.set_merge_distance(merge_spin.value())
            self.zone_tracker.set_track_history(track_spin.value())
            # 同步检测阈值到 AlarmController
            if hasattr(self, 'alarm_ctrl'):
                self.alarm_ctrl.set_threshold(threshold_spin.value())
            self.threshold_alarm = threshold_spin.value()
    # ════════════════════════════════════════════════════════════

    # ------------------------- QSS 科技蓝样式 -------------------------
    def _apply_qss(self):
        self.setStyleSheet(f"""
            QMainWindow, QWidget#Root {{
                background: {BG};
            }}
            QWidget {{
                color: {TEXT};
                font-family: "Microsoft YaHei";
                font-size: 10pt;
            }}
            QLabel#TopMenu {{
                color: rgba(160, 205, 235, 190);
                font-size: 11pt;
                font-weight: bold;
                background: transparent;
            }}
            QLabel#TopMenuActive {{
                color: #ffffff;
                font-size: 11pt;
                font-weight: bold;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(6, 54, 95, 110),
                    stop:0.5 rgba(14, 130, 214, 180),
                    stop:1 rgba(6, 54, 95, 110));
                border: 1px solid rgba(20, 169, 255, 125);
                padding: 4px 8px;
            }}
            QLabel#TopTime {{
                color: #9edcff;
                font-family: Consolas;
                font-size: 11pt;
                font-weight: bold;
                background: transparent;
            }}
            QWidget#TechPanel {{
                border: none;
                background: transparent;
            }}
            QLabel#PanelTitle {{
                color: #e9f9ff;
                font-weight: bold;
                background: transparent;
                border: none;
                padding-left: 30px;
            }}
            QWidget#PanelContent {{
                background: transparent;
            }}
            QFrame#NumberCard {{
                border: none;
                border-radius: 0px;
                background: rgba(8, 39, 72, 48);
            }}
            QLabel#SmallText {{
                color: {MUTED};
                font-weight: bold;
                font-size: 9pt;
            }}
            QLabel#NumberText {{
                color: {CYAN};
                font-family: Consolas;
                font-size: 15pt;
                font-weight: bold;
            }}
            QLabel#StateLabel {{
                border: none;
                border-left: 1px solid rgba(13, 106, 162, 120);
                border-radius: 0px;
                padding: 4px 8px;
                background: rgba(6, 31, 58, 55);
                font-size: 11pt;
                font-weight: bold;
            }}
            QLabel#DataLine, QLabel#ChargeText {{
                border: none;
                border-bottom: 1px solid rgba(13, 76, 122, 100);
                border-radius: 0px;
                padding: 7px;
                color: {CYAN};
                background: rgba(7, 36, 66, 50);
                font-family: Consolas;
                font-weight: bold;
            }}
            QLabel#CameraView {{
                border: 1px solid rgba(22, 132, 201, 105);
                border-radius: 2px;
                background: #06111d;
                color: {MUTED};
                font-size: 14pt;
                font-weight: bold;
            }}
            QTableWidget {{
                background: rgba(5, 20, 38, 62);
                alternate-background-color: rgba(8, 41, 71, 50);
                gridline-color: rgba(12, 74, 120, 55);
                border: none;
                color: {TEXT};
                selection-background-color: #0b67a6;
            }}
            QHeaderView::section {{
                background: rgba(10, 52, 91, 105);
                color: {TEXT};
                border: none;
                border-bottom: 1px solid rgba(20, 169, 255, 80);
                padding: 4px;
                font-weight: bold;
            }}
            QComboBox {{
                color: {TEXT};
                background: rgba(7, 31, 61, 130);
                border: 1px solid rgba(27, 131, 202, 120);
                border-radius: 2px;
                padding: 6px 26px 6px 8px;
                font-weight: bold;
            }}
            QComboBox::drop-down {{
                width: 22px;
                border-left: 1px solid rgba(27, 131, 202, 100);
                background: rgba(4, 20, 42, 150);
            }}
            QComboBox::down-arrow {{
                width: 0px;
                height: 0px;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid {CYAN};
            }}
            QComboBox QAbstractItemView {{
                color: {TEXT};
                background: #061a32;
                border: 1px solid rgba(27, 131, 202, 180);
                selection-color: #ffffff;
                selection-background-color: #0b67a6;
                outline: none;
                padding: 4px;
            }}
            QComboBox QAbstractItemView::item {{
                min-height: 26px;
                padding: 5px 8px;
            }}
            QComboBox QAbstractItemView::item:hover {{
                background: rgba(37, 234, 255, 55);
            }}
            QPushButton#BlueButton {{
                color: #ffffff;
                background: rgba(7, 29, 55, 165);
                border: 1px solid rgba(27, 131, 202, 170);
                border-radius: 2px;
                padding: 5px;
                font-weight: bold;
            }}
            QPushButton#BlueButton:hover {{
                background: rgba(11, 103, 166, 190);
                border: 1px solid #25eaff;
            }}
            QPushButton#GreenButton {{
                color: #ffffff;
                background: rgba(8, 115, 66, 185);
                border: 1px solid {GREEN};
                border-radius: 2px;
                padding: 5px;
                font-weight: bold;
            }}
            QPushButton#DangerButton {{
                color: #ffffff;
                background: rgba(144, 17, 23, 190);
                border: 1px solid {RED};
                border-radius: 2px;
                padding: 5px;
                font-weight: bold;
            }}
            QLabel#AlertDetail, QTextEdit#AlertDetail {{
                color: {YELLOW};
                border: 1px solid #725d16;
                background: rgba(48, 40, 7, 120);
                padding: 8px;
                font-size: 9pt;
                font-weight: bold;
            }}
            QTextEdit#AlertDetail {{
                selection-background-color: rgba(255, 223, 77, 80);
            }}
        """)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("智能巡检-综合管理平台")
    window = MainWindow()
    window.show()

    if cv2 is None:
        QMessageBox.information(window, "提示", "当前未安装 opencv-python")

    sys.exit(run_qt_app(app))


if __name__ == "__main__":
    main()
