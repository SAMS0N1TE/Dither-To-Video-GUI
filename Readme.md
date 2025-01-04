## Introduction

The **Advanced Video Dithering App** is a Python-based GUI application that allows users to apply various dithering algorithms to videos. Enhance your videos with artistic dithering effects, adjust image parameters, and preview changes in real-time.

## Features

- **Multiple Dithering Algorithms:** Choose from Ordered, Atkinson, Floyd-Steinberg, Jarvis-Judice-Ninke, Stucki, and Patterned Dithering.
- **Customizable Palette:** Select up to 8 custom colors for dithering.
- **Image Adjustments:** Adjust brightness, contrast, saturation, hue, and resolution scale.
- **Noise Reduction:** Option to apply Gaussian Blur for cleaner outputs.
- **Live Preview:** Real-time preview of adjustments and dithering effects.
- **Pan and Zoom:** Navigate the preview pane with pan and zoom controls.
- **Lock Zoom:** Maintain current zoom and pan settings during live updates.
- **Preserve Original Colors:** Blend dithered frames with original colors for enhanced effects.
- **Progress Tracking:** Visual progress bar during video processing.


## Install Dependencies
You should probably use a venv
```
python -m venv venv
source venv/bin/activate
```
# On Windows:
```
venv\Scripts\activate
```
# Run:
```sh 
pip install -r requirements.txt
```

## Usage
You need to have [ffmpeg](https://www.ffmpeg.org/) installed and only `.mp4` files are supported.
You should place videos in the input folder, but you also do have the options of chosing your own directory. However it defaults to the iput and output folders. 

```
Run python src/main.py
```
