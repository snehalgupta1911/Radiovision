# Radiovision — RF-Pose & WiFi CSI Gesture Sensing

> **Through-wall human pose estimation and gesture recognition using WiFi Channel State Information (CSI) on the XIAO ESP32-S3.**

---

## Table of Contents
1. [Project Overview](#overview)
2. [Repository Structure](#structure)
3. [Hardware Requirements](#hardware)
4. [Software Setup](#setup)
5. [CSI Logger — Recording Datapoints](#csi-logger)
6. [Batch Dataset Collection](#batch-recording)
7. [Model Architecture](#models)
8. [Training](#training)
9. [Quick Reference Commands](#quick-reference)

---

## Overview

Radiovision implements the **RF-Pose** architecture (Zhao et al., CVPR 2018 / MIT CSAIL) adapted for consumer Wi-Fi hardware (XIAO ESP32-S3).  
The pipeline is:

```
ESP32-S3 ──[Serial/USB]──▶ csi_logger.py ──▶ data/<label>/*.npy
                                                      │
                                             CSIGestureDataset
                                                      │
                                          CSIMultiTaskModel (1D CNN + Bi-LSTM)
                                                      │
                                    gesture_logits / coarse_pose / motion_trajectory
```

---

## Repository Structure

```
Radiovision/
├── csi_logger.py          # ◀ Main CSI recording tool
├── record_dataset.py      # Batch session recorder
├── rfpose/
│   ├── models/
│   │   ├── rfpose_student.py      # RF-Pose 3D CNN Student Network
│   │   ├── rfpose_teacher.py      # Visual Teacher Network
│   │   ├── csi_multitask.py       # 1D CNN + Bi-LSTM Multi-Task Model
│   │   └── esp32s3_gesture_net.py # Lightweight ESP32-S3 Gesture Net
│   ├── csi_esp32/
│   │   ├── esp32s3_parser.py      # Serial CSI data parser
│   │   └── preprocessor.py        # Filter / normalize / segment pipeline
│   ├── data/
│   │   └── csi_dataset.py         # PyTorch Dataset for training
│   └── losses/
│       └── cross_modal_loss.py    # Cross-modal supervision loss
├── data/                  # ◀ Recorded datapoints (created automatically)
│   ├── labels.csv
│   ├── wave/
│   │   ├── wave_0_amp.npy
│   │   ├── wave_0_iq.npy
│   │   └── wave_0_meta.json
│   └── ...
└── requirements.txt
```

---

## Hardware Requirements

| Component | Specification |
|-----------|--------------|
| **Wi-Fi Sensor** | XIAO ESP32-S3 (or any ESP32 with CSI firmware) |
| **Connection** | USB-C to host machine |
| **Firmware** | ESP-IDF with `esp_wifi_set_csi()` enabled |
| **Baud Rate** | `921600` (default) |

---

## Software Setup

```bash
# 1. Clone
git clone https://github.com/snehalgupta1911/Radiovision.git
cd Radiovision

# 2. Install Python dependencies
pip install -r requirements.txt
```

**requirements.txt** includes: `torch`, `numpy`, `scipy`, `pyserial`

---

## CSI Logger — Recording Datapoints

### Single session (recommended approach)

```bash
# macOS / Linux
python csi_logger.py --label wave --duration 10 --port /dev/tty.usbmodem1101

# Windows
python csi_logger.py --label wave --duration 10 --port COM4
```

### Interactive wizard (if you don't know your port)

```bash
python csi_logger.py
```

The wizard will:
1. Show all available serial ports
2. Ask you to choose a gesture label
3. Ask for recording duration
4. Start recording with a live progress display

### Arguments

| Argument | Short | Default | Description |
|----------|-------|---------|-------------|
| `--label` | `-l` | wizard | Gesture label (e.g. `wave`, `stand`) |
| `--duration` | `-d` | `10` | Recording length in **seconds** |
| `--port` | `-p` | wizard | Serial port of ESP32-S3 |
| `--baud` | `-b` | `921600` | Serial baud rate |
| `--out` | `-o` | `data/` | Output directory |
| `--quiet` | `-q` | — | Suppress verbose output |

### Why 10 seconds?

The model's input window is **100 frames ≈ 2 seconds** at ~50 Hz CSI rate.  
A 10-second recording produces **~17 overlapping windows** (stride=25), giving the model enough examples to learn the complete gesture trajectory including:
- Pre-gesture static phase
- Active gesture motion (peak variation)
- Post-gesture return to static

For slow gestures (e.g. `walk`, `sit`) use `--duration 15` or longer.

### Output files

```
data/wave/
├── wave_0_amp.npy     # Amplitude matrix  (T, N_subcarriers) float32
├── wave_0_iq.npy      # Raw I/Q matrix    (T, N_subcarriers, 2) int16
└── wave_0_meta.json   # Metadata (fps, subcarriers, num_windows, …)

data/labels.csv        # Running index: label, file, timestamp
```

---

## Batch Dataset Collection

Record **multiple gesture labels × multiple repetitions** in one automated session:

```bash
# Record 5 gestures × 3 reps × 10 s each, with a 3-second get-ready countdown
python record_dataset.py \
    --labels wave swipe_left swipe_right push stand normal \
    --reps 3 \
    --duration 10 \
    --countdown 3 \
    --port /dev/tty.usbmodem1101
```

The script will:
1. Show the full plan before starting
2. Count down before **each** session so you can position yourself
3. Record and save automatically
4. Print a completion summary with any failures

---

## Model Architecture

### CSIMultiTaskModel (`rfpose/models/csi_multitask.py`)

```
Input: (B, N_subcarriers, T=100)
   │
   ▼
1D CNN Spatial Extractor  →  (B, 256, T)
   │
   ▼
Bidirectional LSTM (2-layer)  →  (B, T, 256)
   │
   ├──▶ Gesture Head      →  (B, num_gestures)       [classification]
   ├──▶ Pose Head         →  (B, num_keypoints, 2)   [2D keypoint regression]
   └──▶ Motion Head       →  (B, T, 3)               [3D trajectory]
```

### RFPoseStudent (`rfpose/models/rfpose_student.py`)

Dual-branch 3D CNN that processes vertical + horizontal RF heatmaps for full pose estimation.

---

## Training

```python
from rfpose.data.csi_dataset import CSIGestureDataset
from rfpose.models.csi_multitask import CSIMultiTaskModel
from torch.utils.data import DataLoader
import torch

# Load dataset
ds = CSIGestureDataset("data/", window_size=100, stride=25, augment=True)
ds.summary()
dl = DataLoader(ds, batch_size=32, shuffle=True)

# Model
model = CSIMultiTaskModel(
    in_subcarriers = ds[0][0].shape[0],
    num_gestures   = len(ds.labels)
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn   = torch.nn.CrossEntropyLoss()

for x, y in dl:
    out  = model(x)
    loss = loss_fn(out["gesture_logits"], y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

---

## Quick Reference Commands

```bash
# ── Find your ESP32 port ──────────────────────────────────────────────────
python -c "import serial.tools.list_ports; print([p.device for p in serial.tools.list_ports.comports()])"

# ── Single recording ──────────────────────────────────────────────────────
python csi_logger.py --label wave --duration 10 --port /dev/tty.usbmodem1101

# ── Batch dataset ─────────────────────────────────────────────────────────
python record_dataset.py --labels wave swipe_left push stand normal --reps 5 --duration 10 --port /dev/tty.usbmodem1101

# ── Check collected data ──────────────────────────────────────────────────
python -c "
from rfpose.data.csi_dataset import CSIGestureDataset
ds = CSIGestureDataset('data/')
ds.summary()
"
```
