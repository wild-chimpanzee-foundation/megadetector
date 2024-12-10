# Batch Processing Camera Trap Videos (with Megadetector)

This repository contains a simple commandline tool for applying megadetector to camera trap video data without extracting the frames first. It is based on MegaDetectorV6 (corresponding to PyTorchWildlife version 1.1.0). Please note that this repository does not have the functionality to use a different detector.

## Installation

Please make sure PytorchWildlife is installed following the official instructions [here](https://github.com/microsoft/CameraTraps).

## Usage

Examples of how to use the commandline tool:
```bash
python batch_video_detection.py \
    --weights="weights/MDV6b-yolov9c.pt" \ # pretrained megadetector weights
    --data_path="path/to/videos" \ # path to videos (currently only looking for .mp4 files)
    --output_fps=4 \ # frames per second
    --batch_size=8 \ # num of videos in batch 
```

For more advice run:
```bash
python batch_video_detection.py --help
```

