# EVL Movement Tracking
This program uses AI to detect and track object movement to determine if the object moved left or right. The program is currently set to detect and track people in the video feed provided to it. You can either use an MP4 video file or a live camera feed(either online or a physically connected camera). 

The current combination is YOLOv8 for AI object detection and DeepSort + torchreid for tracking the objects movement.

For best results use a clear and smooth video that isnt too far from where people can be detected. The less clear and laggy the video the less accurate the AI is.

## Data
This program creates a csv file of data from the video inputed. The data include:
- Date
- Time
- CPU usage
- Memory usage
- Swap memory usage
- CPU temperature
- GPU temperature

## Dependencies
(Need to finalize dependencies)

## Prerequisites
(Will talk about how you need to have the universal AI dependencies and other files)

## Setup
(Have the setup scripts here and any manual installations/configurations)
