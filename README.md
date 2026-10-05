# Spinach Leaf Disease Detection on Raspberry Pi 4 (YOLOv5 + PyTorch)

![Detections on validation images](media_2_detection_grid.jpg)

A low-cost crop disease detector for small farms. We trained a custom **YOLOv5s** model on 418 spinach leaf photos we collected ourselves, then deployed it on a **Raspberry Pi 4 (4 GB)** with a camera so it can flag diseased leaves in the field without a GPU or a cloud connection.

| | |
|---|---|
| Classes | `good`, `infected`, `yellow` |
| Dataset | 418 farm photos, labelled and augmented in Roboflow ([public on Roboflow Universe](https://universe.roboflow.com/ghrce-a8zlp/vf-ystlc/dataset/2), CC BY 4.0) |
| Model | YOLOv5s, 7.25M parameters, 640 px input, trained in PyTorch on Google Colab |
| Result | **0.70 mAP@0.5** on the 70-image validation set (precision 0.64, recall 0.70) |
| Hardware | Raspberry Pi 4 (4 GB), 64-bit Raspberry Pi OS, USB or Pi camera |
| Trained weights | [`weights/best.pt`](weights/best.pt) (14.9 MB) |

## Contents

- [Repository layout](#repository-layout)
- [Dataset](#dataset)
- [Training](#training)
- [Results](#results)
- [Run detection on a PC](#run-detection-on-a-pc)
- [Deploy on Raspberry Pi 4](#deploy-on-raspberry-pi-4)
- [Troubleshooting](#troubleshooting)

## Repository layout

```
.
├── weights/best.pt            # trained model (best epoch), use this one
├── weights/last.pt            # weights from the final epoch
├── data/spinach.yaml          # class names and dataset paths
├── customModel.ipynb          # full Colab training notebook
├── detect.py                  # run the model on images, video or a camera
├── train.py, val.py, export.py
├── models/custom_yolov5s.yaml # YOLOv5s config with nc: 3
├── yolov5rpi4/install.sh      # installs PyTorch + TorchVision on the Pi
├── runs/train/dt_result5/     # training logs, curves, confusion matrix
└── runs/detect/exp/           # predictions on the 39 test images
```

The code is a snapshot of [ultralytics/yolov5](https://github.com/ultralytics/yolov5) at commit `fbe67e4` (July 2022), the version the model was trained with.

## Dataset

- **Collection:** 418 high-resolution photos of spinach leaves taken by our team on local farms.
- **Labels:** every leaf was boxed in [Roboflow](https://roboflow.com) as `good`, `infected` or `yellow`.
- **Split:** Roboflow's automatic train / valid / test split (70 validation images, 39 test images).
- **Preprocessing:** auto-orient and resize to 640×640.
- **Augmentation:** random noise, blur, exposure, shear, crop, 90° rotations and flips.

Download it from Roboflow in **YOLOv5 PyTorch** format and unzip it into a folder named `vf-2` at the repo root. `data/spinach.yaml` already points there.

## Training

Everything we ran is in [`customModel.ipynb`](customModel.ipynb) (Colab, NVIDIA A100). To reproduce it from this repo:

```bash
pip install -r requirements.txt
pip install roboflow
```

```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")   # never commit a real key
rf.workspace("ghrce-a8zlp").project("vf-ystlc").version(2).download("yolov5")  # creates ./vf-2
```

```bash
python train.py --img 640 --batch 32 --epochs 500 \
  --data data/spinach.yaml \
  --cfg models/custom_yolov5s.yaml \
  --weights yolov5s.pt \
  --hyp runs/train/dt_result5/hyp.yaml \
  --name dt_result --cache
```

Training starts from the COCO-pretrained `yolov5s.pt` included in the repo. Early stopping (patience 100) ended our run at epoch 213, about 16 minutes on the A100; the best checkpoint came from epoch 112.

## Results

Validation of `best.pt` on 70 images (233 labelled leaves), from the notebook output:

| Class | Labels | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|---:|
| **all** | 233 | 0.635 | 0.701 | **0.702** | 0.456 |
| good | 69 | 0.586 | 0.739 | 0.733 | 0.512 |
| infected | 152 | 0.637 | 0.697 | 0.676 | 0.373 |
| yellow | 12 | 0.681 | 0.667 | 0.697 | 0.482 |

`yellow` has only 12 validation labels, so its numbers are noisy. On the Colab A100, inference took about 13 ms per 640×640 image. We did not benchmark speed on the Pi.

| Training curves | Confusion matrix |
|---|---|
| ![results](runs/train/dt_result5/results.png) | ![confusion matrix](runs/train/dt_result5/confusion_matrix.png) |

More plots (PR, F1, precision and recall curves) and the per-epoch `results.csv` are in [`runs/train/dt_result5`](runs/train/dt_result5). Predictions on all 39 test images are in [`runs/detect/exp`](runs/detect/exp).

## Run detection on a PC

```bash
git clone https://github.com/14harshaldhote/Yolo-Pytorch-Crop-Disease-DETECTION_model-on-raspberryPi4.git
cd Yolo-Pytorch-Crop-Disease-DETECTION_model-on-raspberryPi4
pip install -r requirements.txt

python detect.py --weights weights/best.pt --data data/spinach.yaml --conf-thres 0.4 \
  --source path/to/leaf_images/
```

Annotated images are saved to `runs/detect/exp2`, `exp3`, and so on. `--source` accepts an image, a folder, a video file, or a camera index such as `0`.

## Deploy on Raspberry Pi 4

**You need:** a Raspberry Pi 4 (4 GB recommended), a 16 GB+ microSD card, a USB webcam or Pi camera, and **64-bit Raspberry Pi OS (Bullseye)** from the [Raspberry Pi Imager](https://www.raspberrypi.com/software/).

> The PyTorch and TorchVision wheels that `install.sh` downloads are built for 64-bit ARM (`aarch64`) and **Python 3.9**, which is what Bullseye ships. Check with `uname -m` (should print `aarch64`) and `python3 --version` (should print 3.9.x). Newer Bookworm images ship Python 3.11 and these wheels will not install there.

**1. Clone the repo**

```bash
git clone https://github.com/14harshaldhote/Yolo-Pytorch-Crop-Disease-DETECTION_model-on-raspberryPi4.git
cd Yolo-Pytorch-Crop-Disease-DETECTION_model-on-raspberryPi4
```

**2. Install PyTorch and TorchVision**

```bash
cd yolov5rpi4
chmod +x install.sh
./install.sh        # asks for your sudo password
cd ..
```

The script installs the system libraries PyTorch needs (OpenBLAS, OpenMPI, OpenMP, libjpeg), then downloads and installs `torch 1.8.0` and `torchvision 0.9.0` wheels for the Pi from Google Drive.

**3. Install the remaining Python packages**

```bash
pip3 install -r requirements.txt
```

**4. Run the model**

On saved images:

```bash
python3 detect.py --weights weights/best.pt --data data/spinach.yaml --conf-thres 0.4 \
  --source path/to/leaf_images/
```

On a live camera (index `0` is the first camera):

```bash
python3 detect.py --weights weights/best.pt --data data/spinach.yaml --conf-thres 0.4 --source 0
```

Each frame's detections (for example `1 good, 2 infecteds`) print in the terminal, and the annotated images or video are saved under `runs/detect/`. Add `--view-img` to show a live preview window when a monitor is attached, or `--nosave` to skip saving.

## Troubleshooting

- **`FileNotFoundError: data/coco128.yaml`:** pass `--data data/spinach.yaml`. The upstream default points at a COCO file that is not in this repo.
- **`ERROR: ... is not a supported wheel on this platform` from `install.sh`:** you are not on 64-bit Pi OS with Python 3.9. See the note at the top of the Pi section.
- **Camera not found with `--source 0`:** run `ls /dev/video*` to see which index your camera uses. For the Pi camera module on Bullseye, enable the legacy camera interface in `sudo raspi-config` so it appears as `/dev/video0`.
- **Pi runs out of memory:** use the 4 GB model, close the desktop, or lower the input size with `--img 320` (faster but less accurate).

## License

GPL-3.0, inherited from YOLOv5. See [LICENSE](LICENSE).
