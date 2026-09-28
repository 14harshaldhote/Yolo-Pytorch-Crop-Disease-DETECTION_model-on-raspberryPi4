# LinkedIn "Add project": Spinach Leaf Disease Detection

Source: https://github.com/14harshaldhote/Yolo-Pytorch-Crop-Disease-DETECTION_model-on-raspberryPi4
All numbers below come from the repo itself (customModel.ipynb output and runs/train/dt_result5).

## Form fields

**Project name**
Spinach Leaf Disease Detection on Raspberry Pi 4 (YOLOv5, PyTorch)

**Description** (about 1,000 characters, under LinkedIn's 2,000 limit)

Team project: a low-cost crop disease detector that runs on a Raspberry Pi 4 (4 GB RAM), built for small farms.

- Collected 418 spinach leaf photos from farms ourselves and labelled them in Roboflow into 3 classes: good, infected and yellow.
- Used Roboflow for the train/valid/test split, preprocessing and augmentation (noise, blur, exposure, shear, crop, rotation, flip).
- Trained a custom YOLOv5s object detection model in PyTorch on Google Colab (640 px images, early stopping at 213 epochs).
- Result on the 70-image validation set: 0.70 mAP@0.5 (precision 0.64, recall 0.70).
- Deployed the model on the Raspberry Pi 4 with 64-bit Raspberry Pi OS: wrote a Bash install script for the ARM builds of PyTorch and TorchVision, and ran detection on saved images or a live camera feed.
- Wrote a step-by-step README so anyone can deploy their own custom YOLO model on a Pi.

**Skills** (pick these 5)
Python, PyTorch, Computer Vision, Object Detection, Raspberry Pi
(Other honest options: YOLOv5, Roboflow, Google Colab, Linux, Bash)

**Media / links** (in this order)
1. Link: the GitHub repo URL above
2. Image: media_1_single_detection.jpg (one leaf flagged "infected 0.86", reads well as a thumbnail)
3. Image: media_2_detection_grid.jpg (16 validation images with predicted boxes)
4. Optional: media_3_training_curves.png

**Start / end date**
Training ran in Feb 2023 and the last commit is Mar 2023. Use your real start month (about Jan or Feb 2023) and Mar 2023 as the end.

**Associated with**
Your college if it was an academic project, otherwise leave it empty.

**Contributors**
Add your teammates if they are on LinkedIn (the README says "we").

## What not to claim
- No production or company use, and no "AI engineer" framing.
- No speed or FPS figure on the Pi: the repo doesn't measure it.
- No accuracy % other than the mAP above (mAP is not "accuracy").

## Repo fixes before recruiters click through
1. **Urgent:** customModel.ipynb has a real Roboflow API key in plain text (the cell after the masked one). Revoke/regenerate it in Roboflow settings, then delete that cell. Deleting alone isn't enough because it stays in git history.
2. Put a demo image and a 3-line summary at the top of the README (see README_top_section.md).
3. Delete the committed `models/__pycache__/` folder and add a `.gitignore` with `__pycache__/`.
4. Fix typos in the README ("necessarcy", "annoating", "augmenttaion", "visti", "treminal").
5. Add a short repo description and topics on GitHub (About ⚙: `yolov5 pytorch raspberry-pi object-detection computer-vision`).
6. Pin the repo on your GitHub profile.
