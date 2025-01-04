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

### Steps

### Windows Executable Download
- [HERE](https://github.com/SAMS0N1TE/Dither-To-Video-GUI/releases/tag/Dithering)

You need to have [ffmpeg](https://www.ffmpeg.org/) installed and only `.mp4` files are supported.
You should place videos in the input folder, but you also do have the options of chosing your own directory. However it defaults to the iput and output folders. 

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/dither-video-app.git
   cd dither-video-app

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

```
Run python src/main.py
```


# Examples
![Screenshot 2025-01-04 115553](https://github.com/user-attachments/assets/f79d88cc-f562-4be0-b6f6-299b074c87d9)

Exported as MP4 at the lowest resolution, pattern dithering, then converted to a GIF.
![lowres](https://github.com/user-attachments/assets/b8cf7481-6b8b-483c-9a1d-c3bb3b9d6f11)
