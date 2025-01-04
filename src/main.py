import numpy as np
import os
import cv2
from numba import jit
import glob
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser
from tkinter import ttk
from PIL import Image, ImageTk

# ------------------ Dithering Functions ------------------

@jit(nopython=True)
def clamp(val, low=0, high=255):
    return max(low, min(high, val))

@jit(nopython=True)
def to_gray(r, g, b):
    """Convert (R, G, B) to grayscale using 0.299*r + 0.587*g + 0.114*b."""
    return 0.299 * r + 0.587 * g + 0.114 * b

@jit(nopython=True)
def find_nearest_color(gray_val, palette):
    """
    Find nearest color in 'palette' based on grayscale distance.
    palette is a list of (R, G, B).
    Return (R, G, B).
    """
    min_dist = 1e12
    nearest = (0, 0, 0)
    for i in range(len(palette)):
        R, G, B = palette[i]
        c_gray = to_gray(R, G, B)
        dist = abs(gray_val - c_gray)
        if dist < min_dist:
            min_dist = dist
            nearest = (R, G, B)
    return nearest

@jit(nopython=True)
def ordered_dithering(image, palette, strength=100.0):
    """
    Ordered dithering with a 2x2 Bayer matrix.
    'strength' in range 0..100.
    """
    bayer2 = np.array([[0, 128],
                       [192, 64]], dtype=np.float32)
    h, w, c = image.shape
    for y in range(h):
        for x in range(w):
            old_r = image[y, x, 2]
            old_g = image[y, x, 1]
            old_b = image[y, x, 0]
            gray = to_gray(old_r, old_g, old_b)

            # "threshold" from the Bayer matrix
            threshold = bayer2[y % 2, x % 2] * (strength / 100.0)

            # Apply threshold
            final_gray = gray + threshold * (1.0 / 255.0) * 255.0

            # Nearest color
            nr, ng, nb = find_nearest_color(final_gray, palette)

            image[y, x, 2] = clamp(nr)
            image[y, x, 1] = clamp(ng)
            image[y, x, 0] = clamp(nb)
    return image

@jit(nopython=True)
def atkinson_dithering(image, palette, strength=100.0):
    """
    Atkinson dithering, distributing error among 6 neighbors.
    'strength' in range 0..100
    """
    h, w, c = image.shape
    for y in range(h):
        for x in range(w):
            old_r = image[y, x, 2]
            old_g = image[y, x, 1]
            old_b = image[y, x, 0]
            old_gray = to_gray(old_r, old_g, old_b)

            # Nearest color
            nr, ng, nb = find_nearest_color(old_gray, palette)
            image[y, x, 2] = clamp(nr)
            image[y, x, 1] = clamp(ng)
            image[y, x, 0] = clamp(nb)

            new_gray = to_gray(nr, ng, nb)
            error = (old_gray - new_gray) * (strength / 100.0)

            # Spread error: Atkinson suggests each neighbor gets error / 8
            e = error / 8.0

            for dy, dx in [(0, 1), (0, 2), (1, -1), (1, 0), (1, 1), (2, 0)]:
                ny = y + dy
                nx = x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    image[ny, nx, 2] = clamp(image[ny, nx, 2] + e)
                    image[ny, nx, 1] = clamp(image[ny, nx, 1] + e)
                    image[ny, nx, 0] = clamp(image[ny, nx, 0] + e)

    return image

@jit(nopython=True)
def floyd_steinberg_dither(image, palette, strength=100.0):
    h, w, c = image.shape
    for y in range(h):
        for x in range(w):
            old_r = image[y, x, 2]
            old_g = image[y, x, 1]
            old_b = image[y, x, 0]
            old_gray = to_gray(old_r, old_g, old_b)

            # Nearest color
            nr, ng, nb = find_nearest_color(old_gray, palette)
            image[y, x, 2] = clamp(nr)
            image[y, x, 1] = clamp(ng)
            image[y, x, 0] = clamp(nb)

            new_gray = to_gray(nr, ng, nb)
            error = (old_gray - new_gray) * (strength / 100.0)

            # Right
            if x + 1 < w:
                image[y, x+1, 2] = clamp(image[y, x+1, 2] + error * 7/16)
                image[y, x+1, 1] = clamp(image[y, x+1, 1] + error * 7/16)
                image[y, x+1, 0] = clamp(image[y, x+1, 0] + error * 7/16)

            # Bottom-left
            if x - 1 >= 0 and y + 1 < h:
                image[y+1, x-1, 2] = clamp(image[y+1, x-1, 2] + error * 3/16)
                image[y+1, x-1, 1] = clamp(image[y+1, x-1, 1] + error * 3/16)
                image[y+1, x-1, 0] = clamp(image[y+1, x-1, 0] + error * 3/16)

            # Bottom
            if y + 1 < h:
                image[y+1, x, 2] = clamp(image[y+1, x, 2] + error * 5/16)
                image[y+1, x, 1] = clamp(image[y+1, x, 1] + error * 5/16)
                image[y+1, x, 0] = clamp(image[y+1, x, 0] + error * 5/16)

            # Bottom-right
            if x + 1 < w and y + 1 < h:
                image[y+1, x+1, 2] = clamp(image[y+1, x+1, 2] + error * 1/16)
                image[y+1, x+1, 1] = clamp(image[y+1, x+1, 1] + error * 1/16)
                image[y+1, x+1, 0] = clamp(image[y+1, x+1, 0] + error * 1/16)
    return image

@jit(nopython=True)
def jarvis_judice_ninke_dither(image, palette, strength=100.0):
    h, w, c = image.shape
    for y in range(h - 2):
        for x in range(2, w - 2):
            old_r = image[y, x, 2]
            old_g = image[y, x, 1]
            old_b = image[y, x, 0]
            old_gray = to_gray(old_r, old_g, old_b)

            nr, ng, nb = find_nearest_color(old_gray, palette)
            image[y, x, 2] = clamp(nr)
            image[y, x, 1] = clamp(ng)
            image[y, x, 0] = clamp(nb)

            new_gray = to_gray(nr, ng, nb)
            error = (old_gray - new_gray) * (strength / 100.0)

            # JJN distribution
            distribution = [
                (0, 1, 7/48), (0, 2, 5/48),
                (1, -2, 3/48), (1, -1, 5/48), (1, 0, 7/48),
                (1, 1, 5/48), (1, 2, 3/48),
                (2, -2, 1/48), (2, -1, 3/48), (2, 0, 5/48),
                (2, 1, 3/48), (2, 2, 1/48)
            ]
            for dy, dx, wgt in distribution:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    image[ny, nx, 2] = clamp(image[ny, nx, 2] + error * wgt)
                    image[ny, nx, 1] = clamp(image[ny, nx, 1] + error * wgt)
                    image[ny, nx, 0] = clamp(image[ny, nx, 0] + error * wgt)
    return image

@jit(nopython=True)
def stucki_dither(image, palette, strength=100.0):
    h, w, c = image.shape
    for y in range(h - 2):
        for x in range(2, w - 2):
            old_r = image[y, x, 2]
            old_g = image[y, x, 1]
            old_b = image[y, x, 0]
            old_gray = to_gray(old_r, old_g, old_b)

            nr, ng, nb = find_nearest_color(old_gray, palette)
            image[y, x, 2] = clamp(nr)
            image[y, x, 1] = clamp(ng)
            image[y, x, 0] = clamp(nb)

            new_gray = to_gray(nr, ng, nb)
            error = (old_gray - new_gray) * (strength / 100.0)

            distribution = [
                (0, 1, 8/42), (0, 2, 4/42),
                (1, -2, 2/42), (1, -1, 4/42), (1, 0, 8/42),
                (1, 1, 4/42), (1, 2, 2/42),
                (2, -2, 1/42), (2, -1, 2/42), (2, 0, 4/42),
                (2, 1, 2/42), (2, 2, 1/42)
            ]
            for dy, dx, wgt in distribution:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w:
                    image[ny, nx, 2] = clamp(image[ny, nx, 2] + error * wgt)
                    image[ny, nx, 1] = clamp(image[ny, nx, 1] + error * wgt)
                    image[ny, nx, 0] = clamp(image[ny, nx, 0] + error * wgt)
    return image

@jit(nopython=True)
def patterned_dithering(image, palette, strength=100.0):
    """
    Patterned dithering using a 4x4 Bayer matrix for more intricate patterns.
    'strength' in range 0..100.
    """
    bayer4 = np.array([
        [0,  8,  2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5]
    ], dtype=np.float32)
    bayer4 = bayer4 * (strength / 100.0) * (255.0 / 16.0)  # Normalize thresholds

    h, w, c = image.shape
    for y in range(h):
        for x in range(w):
            old_r = image[y, x, 2]
            old_g = image[y, x, 1]
            old_b = image[y, x, 0]
            gray = to_gray(old_r, old_g, old_b)

            # "threshold" from the Bayer matrix
            threshold = bayer4[y % 4, x % 4]

            # Apply threshold
            final_gray = gray + threshold

            # Nearest color
            nr, ng, nb = find_nearest_color(final_gray, palette)

            image[y, x, 2] = clamp(nr)
            image[y, x, 1] = clamp(ng)
            image[y, x, 0] = clamp(nb)
    return image

# ------------------ Dithering Dictionary ------------------

DITHERING_TYPES = {
    "Ordered": ordered_dithering,
    "Atkinson": atkinson_dithering,
    "Floyd-Steinberg": floyd_steinberg_dither,
    "Jarvis-Judice-Ninke": jarvis_judice_ninke_dither,
    "Stucki": stucki_dither,
    "Patterned Dithering": patterned_dithering  # New Dithering Method
}

# ------------------ Image Adjustment Functions ------------------

def adjust_brightness(image, value):
    return cv2.convertScaleAbs(image, alpha=1, beta=value)

def adjust_contrast(image, value):
    return cv2.convertScaleAbs(image, alpha=value, beta=0)

def adjust_saturation(image, value):
    image_bgr = image.copy()
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = hsv[:, :, 1] * value
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    image_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return image_bgr

def adjust_hue(image, value):
    """
    Shift hue by 'value' degrees in range -180..180
    """
    image_bgr = image.copy()
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + value) % 180
    hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 179)
    image_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return image_bgr

def apply_noise_reduction(image):
    """
    Apply a Gaussian Blur to reduce noise.
    """
    return cv2.GaussianBlur(image, (5, 5), 0)

# ------------------ GUI Application ------------------

class DitherApp:
    def __init__(self, master):
        self.master = master
        master.title("Advanced Video Dithering App")
        master.geometry("1600x900")
        master.resizable(True, True)

        # Variables
        self.input_video_path = tk.StringVar()
        self.output_dir_path = tk.StringVar()

        self.selected_dither = tk.StringVar(value="Floyd-Steinberg")
        self.selected_palette = []

        self.brightness = tk.IntVar(value=0)
        self.contrast = tk.DoubleVar(value=1.0)
        self.saturation = tk.DoubleVar(value=1.0)
        self.hue = tk.DoubleVar(value=0.0)
        self.dither_strength = tk.DoubleVar(value=100.0)  # 0..100
        self.live_preview = tk.BooleanVar(value=False)
        self.preserve_original = tk.BooleanVar(value=False)
        self.res_scale = tk.DoubleVar(value=1.0)  # output resolution scale: 0.5..2.0
        self.lock_zoom = tk.BooleanVar(value=False)  # New variable for Lock Zoom
        self.noise_reduction = tk.BooleanVar(value=False)  # New variable for Noise Reduction

        # Preview Pan and Zoom variables
        self.preview_image = None
        self.preview_pil = None
        self.offset_x = 0
        self.offset_y = 0
        self.zoom_factor = 1.0
        self.last_mouse_x = None
        self.last_mouse_y = None

        # Layout: 3 columns -> leftFrame (IO + dithering), centerFrame (Preview), rightFrame (adjustments).
        self.leftFrame = tk.Frame(master, bg="#f0f0f0", width=400)
        self.leftFrame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.centerFrame = tk.Frame(master, bg="#d0d0d0")
        self.centerFrame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.rightFrame = tk.Frame(master, bg="#f0f0f0", width=400)
        self.rightFrame.grid(row=0, column=2, sticky="nsew", padx=10, pady=10)

        master.grid_columnconfigure(1, weight=1)  # Make center frame expandable
        master.grid_rowconfigure(0, weight=1)

        # Build the left frame
        self.build_left_frame()
        # Build the center frame (preview)
        self.build_center_frame()
        # Build the right frame (adjustments)
        self.build_right_frame()

    def build_left_frame(self):
        # Input / Output
        io_label = tk.Label(self.leftFrame, text="Input / Output", bg="#f0f0f0", font=("Helvetica", 16, "bold"))
        io_label.pack(pady=15)

        # Input
        input_btn = tk.Button(self.leftFrame, text="Select Input Video", command=self.browse_input, width=30, font=("Helvetica", 12))
        input_btn.pack(pady=5)
        input_entry = tk.Entry(self.leftFrame, textvariable=self.input_video_path, width=40, font=("Helvetica", 12))
        input_entry.pack(pady=5)

        # Output
        output_btn = tk.Button(self.leftFrame, text="Select Output Directory", command=self.browse_output, width=30, font=("Helvetica", 12))
        output_btn.pack(pady=5)
        output_entry = tk.Entry(self.leftFrame, textvariable=self.output_dir_path, width=40, font=("Helvetica", 12))
        output_entry.pack(pady=5)

        # Dithering
        dith_label = tk.Label(self.leftFrame, text="Dithering Options", bg="#f0f0f0", font=("Helvetica", 16, "bold"))
        dith_label.pack(pady=20)

        dith_dropdown = ttk.Combobox(self.leftFrame, textvariable=self.selected_dither,
                                     values=list(DITHERING_TYPES.keys()), state="readonly", width=28, font=("Helvetica", 12))
        dith_dropdown.pack(pady=5)
        dith_dropdown.bind("<<ComboboxSelected>>", self.on_live_preview)

        # Strength
        str_frame = tk.Frame(self.leftFrame, bg="#f0f0f0")
        str_frame.pack(pady=15)
        tk.Label(str_frame, text="Strength (0..100):", bg="#f0f0f0", font=("Helvetica", 12)).pack(side=tk.LEFT)
        str_scale = tk.Scale(str_frame, from_=0, to=100, variable=self.dither_strength,
                             orient=tk.HORIZONTAL, length=200, command=self.on_live_preview)
        str_scale.pack(side=tk.LEFT, padx=5)
        # Show numeric value
        self.dither_strength_entry = tk.Entry(str_frame, width=5, font=("Helvetica", 12))
        self.dither_strength_entry.insert(0, str(int(self.dither_strength.get())))
        self.dither_strength_entry.pack(side=tk.LEFT)
        self.dither_strength_entry.bind("<Return>", self.on_strength_entry_change)

        # Preserve Original
        self.preserve_chk = tk.Checkbutton(self.leftFrame, text="Preserve Original Colors",
                                           bg="#f0f0f0", variable=self.preserve_original,
                                           command=self.on_live_preview, font=("Helvetica", 12))
        self.preserve_chk.pack(pady=10)

        # Noise Reduction
        self.noise_reduction_chk = tk.Checkbutton(self.leftFrame, text="Apply Noise Reduction",
                                                  bg="#f0f0f0", variable=self.noise_reduction,
                                                  command=self.on_live_preview, font=("Helvetica", 12))
        self.noise_reduction_chk.pack(pady=10)

        # Palette
        pal_label = tk.Label(self.leftFrame, text="Select Palette Colors (Up to 8):", bg="#f0f0f0",
                             font=("Helvetica", 16, "bold"))
        pal_label.pack(pady=15)
        self.palette_buttons = []
        pal_frame = tk.Frame(self.leftFrame, bg="#f0f0f0")
        pal_frame.pack()
        for i in range(8):
            btn = tk.Button(pal_frame, text=f"Color {i+1}", command=lambda i=i: self.choose_color(i), width=12, font=("Helvetica", 10))
            btn.grid(row=i//2, column=i%2, padx=5, pady=5)
            self.palette_buttons.append(btn)

        # Start / Process
        start_btn = tk.Button(self.leftFrame, text="Start Dithering", bg="green", fg="white",
                              command=self.start_dithering, font=("Helvetica", 14, "bold"), width=25)
        start_btn.pack(pady=25)

        # Progress + Status
        self.progress = ttk.Progressbar(self.leftFrame, orient=tk.HORIZONTAL, length=300, mode='determinate')
        self.progress.pack(pady=10)

        self.status_label = tk.Label(self.leftFrame, text="Status: Idle", bg="#f0f0f0", font=("Helvetica", 12))
        self.status_label.pack(pady=5)

    def build_center_frame(self):
        # Preview in the middle
        self.preview_canvas = tk.Canvas(self.centerFrame, bg="#808080")
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)

        # Bind pan and zoom events
        self.preview_canvas.bind("<Button-1>", self.on_left_button_down)
        self.preview_canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.preview_canvas.bind("<MouseWheel>", self.on_mouse_wheel)      # Windows
        self.preview_canvas.bind("<Button-4>", self.on_mouse_wheel)        # Linux (scroll up)
        self.preview_canvas.bind("<Button-5>", self.on_mouse_wheel)        # Linux (scroll down)

        # Load Preview + Live toggle + Lock Zoom toggle
        bottom_frame = tk.Frame(self.centerFrame, bg="#d0d0d0")
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        load_prev_btn = tk.Button(bottom_frame, text="Load Preview Frame", command=self.load_preview, width=20, font=("Helvetica", 12))
        load_prev_btn.pack(side=tk.LEFT, padx=10)

        live_chk = tk.Checkbutton(bottom_frame, text="Live Preview", variable=self.live_preview,
                                  bg="#d0d0d0", command=self.on_live_toggle, font=("Helvetica", 12))
        live_chk.pack(side=tk.LEFT, padx=10)

        # Lock Zoom Toggle
        lock_zoom_chk = tk.Checkbutton(bottom_frame, text="Lock Zoom", variable=self.lock_zoom,
                                       bg="#d0d0d0", command=self.on_lock_zoom_toggle, font=("Helvetica", 12))
        lock_zoom_chk.pack(side=tk.LEFT, padx=10)

    def build_right_frame(self):
        # Adjustments
        adj_label = tk.Label(self.rightFrame, text="Image Adjustments", bg="#f0f0f0",
                             font=("Helvetica", 16, "bold"))
        adj_label.pack(pady=15)

        # Brightness
        bright_frame = tk.Frame(self.rightFrame, bg="#f0f0f0")
        bright_frame.pack(pady=10)
        tk.Label(bright_frame, text="Brightness:", bg="#f0f0f0", font=("Helvetica", 12)).pack(side=tk.LEFT)
        bright_scale = tk.Scale(bright_frame, from_=-100, to=100, variable=self.brightness,
                                orient=tk.HORIZONTAL, length=250, command=self.on_live_preview)
        bright_scale.pack(side=tk.LEFT, padx=5)
        self.brightness_entry = tk.Entry(bright_frame, width=5, font=("Helvetica", 12))
        self.brightness_entry.insert(0, str(int(self.brightness.get())))
        self.brightness_entry.pack(side=tk.LEFT)
        self.brightness_entry.bind("<Return>", self.on_brightness_entry_change)

        # Contrast
        cont_frame = tk.Frame(self.rightFrame, bg="#f0f0f0")
        cont_frame.pack(pady=10)
        tk.Label(cont_frame, text="Contrast:", bg="#f0f0f0", font=("Helvetica", 12)).pack(side=tk.LEFT)
        cont_scale = tk.Scale(cont_frame, from_=0.5, to=3.0, resolution=0.1,
                              variable=self.contrast, orient=tk.HORIZONTAL, length=250,
                              command=self.on_live_preview)
        cont_scale.pack(side=tk.LEFT, padx=5)
        self.contrast_entry = tk.Entry(cont_frame, width=5, font=("Helvetica", 12))
        self.contrast_entry.insert(0, f"{self.contrast.get():.1f}")
        self.contrast_entry.pack(side=tk.LEFT)
        self.contrast_entry.bind("<Return>", self.on_contrast_entry_change)

        # Saturation
        sat_frame = tk.Frame(self.rightFrame, bg="#f0f0f0")
        sat_frame.pack(pady=10)
        tk.Label(sat_frame, text="Saturation:", bg="#f0f0f0", font=("Helvetica", 12)).pack(side=tk.LEFT)
        sat_scale = tk.Scale(sat_frame, from_=0.0, to=3.0, resolution=0.1,
                             variable=self.saturation, orient=tk.HORIZONTAL, length=250,
                             command=self.on_live_preview)
        sat_scale.pack(side=tk.LEFT, padx=5)
        self.saturation_entry = tk.Entry(sat_frame, width=5, font=("Helvetica", 12))
        self.saturation_entry.insert(0, f"{self.saturation.get():.1f}")
        self.saturation_entry.pack(side=tk.LEFT)
        self.saturation_entry.bind("<Return>", self.on_saturation_entry_change)

        # Hue
        hue_frame = tk.Frame(self.rightFrame, bg="#f0f0f0")
        hue_frame.pack(pady=10)
        tk.Label(hue_frame, text="Hue:", bg="#f0f0f0", font=("Helvetica", 12)).pack(side=tk.LEFT)
        hue_scale = tk.Scale(hue_frame, from_=-180, to=180, resolution=1,
                             variable=self.hue, orient=tk.HORIZONTAL, length=250,
                             command=self.on_live_preview)
        hue_scale.pack(side=tk.LEFT, padx=5)
        self.hue_entry = tk.Entry(hue_frame, width=5, font=("Helvetica", 12))
        self.hue_entry.insert(0, str(int(self.hue.get())))
        self.hue_entry.pack(side=tk.LEFT)
        self.hue_entry.bind("<Return>", self.on_hue_entry_change)

        # Resolution Scale
        res_frame = tk.Frame(self.rightFrame, bg="#f0f0f0")
        res_frame.pack(pady=20)
        tk.Label(res_frame, text="Resolution Scale (0.5 - 2.0):", bg="#f0f0f0",
                 font=("Helvetica", 12)).pack(side=tk.LEFT)
        res_scale = tk.Scale(res_frame, from_=0.5, to=2.0, resolution=0.1,
                             variable=self.res_scale, orient=tk.HORIZONTAL, length=250,
                             command=self.on_live_preview)
        res_scale.pack(side=tk.LEFT, padx=5)
        self.res_entry = tk.Entry(res_frame, width=5, font=("Helvetica", 12))
        self.res_entry.insert(0, f"{self.res_scale.get():.1f}")
        self.res_entry.pack(side=tk.LEFT)
        self.res_entry.bind("<Return>", self.on_res_entry_change)

    # ------------------ Event Handlers for Numeric Entries ------------------

    def on_strength_entry_change(self, event):
        try:
            val = float(self.dither_strength_entry.get())
            if 0 <= val <= 100:
                self.dither_strength.set(val)
        except ValueError:
            pass
        self.dither_strength_entry.delete(0, tk.END)
        self.dither_strength_entry.insert(0, str(int(self.dither_strength.get())))
        self.on_live_preview()

    def on_brightness_entry_change(self, event):
        try:
            val = float(self.brightness_entry.get())
            if -100 <= val <= 100:
                self.brightness.set(val)
        except ValueError:
            pass
        self.brightness_entry.delete(0, tk.END)
        self.brightness_entry.insert(0, str(int(self.brightness.get())))
        self.on_live_preview()

    def on_contrast_entry_change(self, event):
        try:
            val = float(self.contrast_entry.get())
            if 0.5 <= val <= 3.0:
                self.contrast.set(val)
        except ValueError:
            pass
        self.contrast_entry.delete(0, tk.END)
        self.contrast_entry.insert(0, f"{self.contrast.get():.1f}")
        self.on_live_preview()

    def on_saturation_entry_change(self, event):
        try:
            val = float(self.saturation_entry.get())
            if 0.0 <= val <= 3.0:
                self.saturation.set(val)
        except ValueError:
            pass
        self.saturation_entry.delete(0, tk.END)
        self.saturation_entry.insert(0, f"{self.saturation.get():.1f}")
        self.on_live_preview()

    def on_hue_entry_change(self, event):
        try:
            val = float(self.hue_entry.get())
            if -180 <= val <= 180:
                self.hue.set(val)
        except ValueError:
            pass
        self.hue_entry.delete(0, tk.END)
        self.hue_entry.insert(0, str(int(self.hue.get())))
        self.on_live_preview()

    def on_res_entry_change(self, event):
        try:
            val = float(self.res_entry.get())
            if 0.5 <= val <= 2.0:
                self.res_scale.set(val)
        except ValueError:
            pass
        self.res_entry.delete(0, tk.END)
        self.res_entry.insert(0, f"{self.res_scale.get():.1f}")
        self.on_live_preview()

    def on_lock_zoom_toggle(self):
        """
        Called when the "Lock Zoom" checkbox is toggled.
        If locked, prevent zoom and pan changes during live preview updates.
        """
        if self.lock_zoom.get():
            self.status_label.config(text="Status: Zoom Locked")
        else:
            self.status_label.config(text="Status: Zoom Unlocked")

    # ------------------ GUI Callback Functions ------------------

    def browse_input(self):
        file_path = filedialog.askopenfilename(title="Select MP4 Video", filetypes=[("MP4 files", "*.mp4")])
        if file_path:
            self.input_video_path.set(file_path)

    def browse_output(self):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir_path.set(directory)

    def choose_color(self, index):
        color_code = colorchooser.askcolor(title=f"Choose Palette Color {index+1}")
        if color_code[1] is not None:
            if index < len(self.selected_palette):
                self.selected_palette[index] = color_code[1]
            else:
                self.selected_palette.append(color_code[1])
            self.palette_buttons[index].configure(bg=color_code[1])
            if self.live_preview.get():
                self.on_live_preview()

    def load_preview(self):
        """
        Load first frame from the input video, apply adjustments, dithering, noise reduction (if enabled),
        and show it in the canvas.
        """
        input_video = self.input_video_path.get()
        if not input_video or not os.path.isfile(input_video):
            self.status_label.config(text="Status: Please select a valid input video.")
            return

        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            self.status_label.config(text=f"Status: Cannot open video file {input_video}")
            return

        ret, frame = cap.read()
        cap.release()

        if not ret:
            self.status_label.config(text="Status: Cannot read frame from video.")
            return

        # Scale the frame by res_scale
        sf = self.res_scale.get()
        if abs(sf - 1.0) > 1e-3:
            new_w = int(frame.shape[1] * sf)
            new_h = int(frame.shape[0] * sf)
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Adjust
        frame = adjust_brightness(frame, self.brightness.get())
        frame = adjust_contrast(frame, self.contrast.get())
        frame = adjust_saturation(frame, self.saturation.get())
        frame = adjust_hue(frame, self.hue.get())

        # Build palette (BGR from hex)
        bgr_palette = []
        for hexcol in self.selected_palette:
            r = int(hexcol[1:3], 16)
            g = int(hexcol[3:5], 16)
            b = int(hexcol[5:7], 16)
            bgr_palette.append((r, g, b))

        # If no palette, fallback to black/white
        if not bgr_palette:
            bgr_palette = [(0, 0, 0), (255, 255, 255)]

        # Create copy for dithering
        dith_frame = frame.copy()
        dith_algo = DITHERING_TYPES[self.selected_dither.get()]
        dith_frame = dith_algo(dith_frame, bgr_palette, self.dither_strength.get())

        # If Preserve Original is checked, blend with original at 50%
        if self.preserve_original.get() and len(bgr_palette) > 1:
            alpha = 0.5
            dith_frame = cv2.addWeighted(frame, alpha, dith_frame, 1.0 - alpha, 0)

        # Apply Noise Reduction if enabled
        if self.noise_reduction.get():
            dith_frame = apply_noise_reduction(dith_frame)

        # Convert to PIL for display
        disp_bgr = dith_frame
        disp_rgb = cv2.cvtColor(disp_bgr, cv2.COLOR_BGR2RGB)
        self.preview_pil = Image.fromarray(disp_rgb)

        # If Lock Zoom is not active, reset pan and zoom
        if not self.lock_zoom.get():
            self.offset_x = 0
            self.offset_y = 0
            self.zoom_factor = 1.0

        # Resize to fit the preview canvas
        self.draw_preview()

        self.status_label.config(text="Status: Preview Loaded")

    def start_dithering(self):
        input_video = self.input_video_path.get()
        output_dir = self.output_dir_path.get()

        if not input_video or not os.path.isfile(input_video):
            messagebox.showerror("Error", "Please select a valid input video.")
            return

        if not output_dir or not os.path.isdir(output_dir):
            messagebox.showerror("Error", "Please select a valid output directory.")
            return

        # If palette is empty, just use black & white
        if not self.selected_palette:
            self.selected_palette = ["#000000", "#FFFFFF"]

        # Start background thread
        self.set_ui_state(False)
        self.progress['value'] = 0
        self.status_label.config(text="Status: Processing...")

        threading.Thread(target=self.process_video, args=(input_video, output_dir), daemon=True).start()

    def process_video(self, video_path, output_dir):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Cannot open video file {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Build palette BGR
            bgr_palette = []
            for hexcol in self.selected_palette:
                r = int(hexcol[1:3], 16)
                g = int(hexcol[3:5], 16)
                b = int(hexcol[5:7], 16)
                bgr_palette.append((r, g, b))

            # Dithering function
            dith_algo = DITHERING_TYPES[self.selected_dither.get()]

            current_frame = 0
            temp_dir = os.path.join(output_dir, "temp_frames")
            os.makedirs(temp_dir, exist_ok=True)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                current_frame += 1

                # Scale
                sf = self.res_scale.get()
                if abs(sf - 1.0) > 1e-3:
                    new_w = int(frame.shape[1] * sf)
                    new_h = int(frame.shape[0] * sf)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # Adjust
                frame = adjust_brightness(frame, self.brightness.get())
                frame = adjust_contrast(frame, self.contrast.get())
                frame = adjust_saturation(frame, self.saturation.get())
                frame = adjust_hue(frame, self.hue.get())

                # Dither
                dframe = frame.copy()
                dframe = dith_algo(dframe, bgr_palette, self.dither_strength.get())

                # If Preserve Original is checked, blend with original at 50%
                if self.preserve_original.get() and len(bgr_palette) > 1:
                    alpha = 0.5
                    dframe = cv2.addWeighted(frame, alpha, dframe, 1.0 - alpha, 0)

                # Apply Noise Reduction if enabled
                if self.noise_reduction.get():
                    dframe = apply_noise_reduction(dframe)

                # Save
                out_path = os.path.join(temp_dir, f"frame_{current_frame:06d}.png")
                cv2.imwrite(out_path, dframe)

                # Update progress
                self.update_progress((current_frame / total_frames) * 100)

            cap.release()
            cv2.destroyAllWindows()

            # ffmpeg
            output_video_name = os.path.splitext(os.path.basename(video_path))[0] + "_dithered.mp4"
            output_video_path = os.path.join(output_dir, output_video_name)
            pattern = os.path.join(temp_dir, "frame_%06d.png")

            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(int(fps)),
                "-i", pattern,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                output_video_path
            ]
            subprocess.run(cmd, check=True)

            # Cleanup
            for f in glob.glob(os.path.join(temp_dir, "*.png")):
                os.remove(f)
            os.rmdir(temp_dir)

            self.update_progress(100)
            self.update_status(f"Status: Completed! Saved as {output_video_name}")
            messagebox.showinfo("Success", f"Dithered video saved as {output_video_name}")

        except Exception as e:
            self.update_status("Status: Error occurred.")
            messagebox.showerror("Error", f"An error occurred: {e}")
        finally:
            self.set_ui_state(True)

    def set_ui_state(self, enabled=True):
        """
        Enable or disable interactive widgets to prevent user interaction during processing.
        Only certain widgets support the 'state' attribute.
        """
        state = tk.NORMAL if enabled else tk.DISABLED
        # Define widgets that support 'state'
        widgets_to_configure = []
        # Left Frame
        for child in self.leftFrame.winfo_children():
            if isinstance(child, (tk.Button, ttk.Combobox, tk.Entry, tk.Scale, tk.Checkbutton)):
                widgets_to_configure.append(child)
            elif isinstance(child, tk.Frame):
                for grandchild in child.winfo_children():
                    if isinstance(grandchild, (tk.Button, tk.Entry, tk.Scale, tk.Checkbutton)):
                        widgets_to_configure.append(grandchild)
        # Right Frame
        for child in self.rightFrame.winfo_children():
            if isinstance(child, (tk.Button, ttk.Combobox, tk.Entry, tk.Scale, tk.Checkbutton)):
                widgets_to_configure.append(child)
            elif isinstance(child, tk.Frame):
                for grandchild in child.winfo_children():
                    if isinstance(grandchild, (tk.Button, tk.Entry, tk.Scale, tk.Checkbutton)):
                        widgets_to_configure.append(grandchild)
        # Configure widgets
        for widget in widgets_to_configure:
            try:
                widget.config(state=state)
            except tk.TclError:
                pass  # Some widgets might not support 'state'

    def update_progress(self, value):
        self.progress['value'] = value
        self.master.update_idletasks()

    def update_status(self, message):
        self.status_label.config(text=message)
        self.master.update_idletasks()

    # ------------------ Pan and Zoom Functions ------------------

    def on_left_button_down(self, event):
        self.last_mouse_x = event.x
        self.last_mouse_y = event.y

    def on_mouse_move(self, event):
        if self.last_mouse_x is not None and self.last_mouse_y is not None:
            dx = event.x - self.last_mouse_x
            dy = event.y - self.last_mouse_y
            self.offset_x += dx
            self.offset_y += dy
            self.last_mouse_x = event.x
            self.last_mouse_y = event.y
            self.draw_preview()

    def on_mouse_wheel(self, event):
        # Determine zoom direction
        if hasattr(event, "delta") and event.delta != 0:
            if event.delta > 0:
                zoom_change = 1.1
            else:
                zoom_change = 0.9
        elif event.num == 4:  # Linux scroll up
            zoom_change = 1.1
        elif event.num == 5:  # Linux scroll down
            zoom_change = 0.9
        else:
            zoom_change = 1.0

        # Update zoom factor
        new_zoom = self.zoom_factor * zoom_change
        new_zoom = max(0.1, min(new_zoom, 10.0))
        zoom_change = new_zoom / self.zoom_factor
        self.zoom_factor = new_zoom

        # Get mouse position relative to canvas
        canvas_w = self.preview_canvas.winfo_width()
        canvas_h = self.preview_canvas.winfo_height()
        mouse_x = event.x
        mouse_y = event.y

        # Adjust offset to zoom towards the mouse position
        self.offset_x -= (mouse_x - self.offset_x) * (zoom_change - 1)
        self.offset_y -= (mouse_y - self.offset_y) * (zoom_change - 1)

        self.draw_preview()

    def draw_preview(self):
        """
        Re-draw preview image at current offset_x, offset_y, and zoom_factor.
        Respects the "Lock Zoom" state.
        """
        if self.preview_pil is None:
            return
        # Resize the PIL image based on zoom_factor
        new_w = max(1, int(self.preview_pil.width * self.zoom_factor))
        new_h = max(1, int(self.preview_pil.height * self.zoom_factor))
        resized_pil = self.preview_pil.resize((new_w, new_h), Image.LANCZOS)

        self.preview_image = ImageTk.PhotoImage(resized_pil)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.preview_image)

    # ------------------ Live Preview and Toggle Functions ------------------

    def on_live_toggle(self):
        # If live preview just turned on, refresh once
        if self.live_preview.get():
            self.load_preview()

    def on_live_preview(self, event=None):
        """
        Called whenever a slider or relevant control changes. If live_preview is enabled,
        update preview immediately.
        """
        # Update numeric entries
        self.dither_strength_entry.delete(0, tk.END)
        self.dither_strength_entry.insert(0, str(int(self.dither_strength.get())))
        self.brightness_entry.delete(0, tk.END)
        self.brightness_entry.insert(0, str(int(self.brightness.get())))
        self.contrast_entry.delete(0, tk.END)
        self.contrast_entry.insert(0, f"{self.contrast.get():.1f}")
        self.saturation_entry.delete(0, tk.END)
        self.saturation_entry.insert(0, f"{self.saturation.get():.1f}")
        self.hue_entry.delete(0, tk.END)
        self.hue_entry.insert(0, str(int(self.hue.get())))
        self.res_entry.delete(0, tk.END)
        self.res_entry.insert(0, f"{self.res_scale.get():.1f}")

        if self.live_preview.get():
            self.load_preview()

# ------------------ Main ------------------
def main():
    root = tk.Tk()
    app = DitherApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
