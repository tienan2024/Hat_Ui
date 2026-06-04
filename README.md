# Hat_Ui — 工业安全监测系统

> 工控机上位机界面：可见光相机 + YOLO 安全帽检测 + 热成像 + 环境传感器 + MCU 通信

## 功能概览

| 功能 | 说明 |
|------|------|
| **可见光相机** | USB / 网络摄像头，支持 MJPEG 流 |
| **YOLO 安全帽检测** | 实时检测人员是否佩戴安全帽，YOLOv8 模型 |
| **热成像显示** | 支持 WiFi 热像仪、Lepton 串口、本地模拟三种模式 |
| **温度仪表盘** | 3 级报警（绿/黄闪/红闪），支持右键修改阈值 |
| **环境监测** | 温湿度、烟雾、CH4、CO 传感器数据 |
| **MCU 通信** | 串口连接下位机，支持报警指令下发 |
| **远程工控机模式** | 通过 TCP relay 远程访问工控机上的 Lepton / MCU / 视频流 |

## 项目结构

```
Hat_Ui_V4.0/
├── hat_main.py          # 主界面（PySide6/PyQt5）
├── hat_extensions.py     # 扩展模块（热像仪、MCU、报警、热区追踪）
├── config.json          # 配置文件
├── alarm1.mp3           # 未戴帽报警音
├── best2.pt            # YOLO 安全帽检测模型
├── docs/               # 开发文档
│   ├── 00-索引.md
│   ├── 01-系统架构.md
│   ├── 02-扩展模块接口.md
│   └── 09-更新日志.md
└── 工控机推流脚本/      # 工控机端部署脚本
    ├── relay-server.js  # 传感器 + Lepton 中继服务（Node.js）
    └── video-streamer.py # USB 摄像头 MJPEG 推流（Python）
```

## 快速开始

### 环境要求

```bash
pip install PySide6 opencv-python pyserial ultralytics
```

> 注意：YOLO 依赖（`ultralytics`）会连带安装 PyTorch（GPU 支持约 1.5GB，CPU 版本约 300MB）。如仅需 CPU 推理，可单独安装 CPU 版本：
>
> ```bash
> pip install ultralytics torch torchvision --index-url https://download.pytorch.org/whl/cpu
> ```

> PyQt5 可作为备选（自动降级）

### 运行

```bash
python hat_main.py
```

### YOLO 安全帽检测

1. 放置模型文件 `best2.pt` 到 `hat_main.py` 同目录
2. 连接摄像头
3. 点击 **"开启YOLO"** 按钮开始检测
4. 检测到未戴帽持续 2 秒后触发红色报警 + 警报音

YOLO 配置参数（`hat_main.py` 顶部）：

```python
YOLO_MODEL_PATH = "best2.pt"           # 模型路径
YOLO_CONFIDENCE = 0.55                # 置信度阈值
YOLO_DEVICE = "auto"                   # "auto"=GPU, "cpu"=CPU
YOLO_HALF = True                      # GPU 半精度加速
YOLO_DETECT_INTERVAL = 1               # 每隔几帧检测一次
```

### 热成像模式

| 模式 | 说明 |
|------|------|
| WiFi 热像仪 | 连接远程热像仪设备 |
| Lepton 串口 | 本地串口连接 Lepton 2.5 |
| Lepton (中继) | 通过工控机 relay 远程访问 Lepton |
| 模拟器 | 内置模拟数据，用于开发调试 |

### 远程工控机模式

**工控机部署：**

```bash
# 启动传感器 + Lepton 中继
cd 工控机推流脚本
node relay-server.js

# 启动 USB 摄像头推流（独立运行）
python video-streamer.py --device 0 --width 640 --height 480 --port 8080
```

**上位机配置（config.json）：**

```json
{
  "camera": {
    "network_url": "http://192.168.3.166:8080/stream"
  },
  "lepton_relay": {
    "host": "192.168.3.166",
    "port": 6002
  },
  "sensor_relay": {
    "host": "192.168.3.166",
    "port": 6001
  }
}
```

## 技术栈

- **UI 框架**：PySide6（优先）/ PyQt5（备选）
- **图像处理**：OpenCV
- **YOLO 检测**：ultralytics YOLOv8
- **串口通信**：pyserial
- **热像仪**：FLIR Lepton 2.5 + 定制驱动
- **中继服务**：Node.js（传感器/Lepton TCP 透传）
- **视频推流**：Python + OpenCV HTTP MJPEG

## 报警规则

| 级别 | 条件 | 表现 |
|------|------|------|
| 正常 | 温度 < 40℃ | 绿色 |
| 二级预警 | 40℃ ≤ 温度 < 70℃ | 黄色闪烁 |
| 三级报警 | 温度 ≥ 70℃ | 红色闪烁 + 报警音 |

安全帽检测：未戴帽持续 2 秒 → 红色"未戴帽"报警 + `alarm1.mp3` 警报音
