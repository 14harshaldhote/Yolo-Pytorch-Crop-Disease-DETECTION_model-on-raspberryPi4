# Spinach Leaf Disease Detection on Raspberry Pi 4 (YOLOv5 + PyTorch)

![Detection example](runs/detect/exp/DSC_1362_JPG.rf.8546a40f637e8916b154ecefefb542f1.jpg)

A low-cost crop disease detector: a custom YOLOv5s model trained on 418 spinach leaf photos we collected from farms, deployed on a Raspberry Pi 4 (4 GB) with a camera.

| | |
|---|---|
| Classes | good, infected, yellow |
| Dataset | 418 images, labelled and augmented in Roboflow |
| Model | YOLOv5s (PyTorch), 640 px, trained on Google Colab |
| Validation result | 0.70 mAP@0.5 on 70 images (precision 0.64, recall 0.70) |
| Hardware | Raspberry Pi 4, 64-bit Raspberry Pi OS, USB/Pi camera |

<!-- Paste this above the existing README content. Keep the rest (data, training, deployment steps) below it. -->
