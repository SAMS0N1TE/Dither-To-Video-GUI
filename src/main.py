"""
main.py

This GUI application provides advanced video and image dithering.
Features include:
  • Multiple dithering algorithms (including Retro Shading and others)
  • Advanced image adjustments (brightness, contrast, saturation, hue)
  • Customizable palette extraction via KMeans clustering with fallback
  • A modern dark-themed interface with pan & zoom live preview
  • Menu bar with File, Settings (save/load/reset configuration), and Help options
  • Configuration management via JSON files
  • Optional custom resolution input (e.g., "800x600")
"""

import os
import sys
import glob
import json
import cv2
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from tkinter import ttk
from tkinter import colorchooser
from PIL import Image, ImageTk

os.environ["LOKY_MAX_CPU_COUNT"] = "8"  # Replace 8 with your desired core count

from dithering_backend import (
    DITHERING_TYPES,
    adjust_brightness,
    adjust_contrast,
    adjust_saturation,
    adjust_hue,
    apply_noise_reduction,
    extract_palette_accurate_kmeans,
    extract_palette_fast_opencv,
    floyd_steinberg_dither,
    retro_shading_dithering,
    blend_palette
)

try:
    import numpy as np
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class DitherApp:
    def __init__(self, master):
        self.master = master
        master.title("Advanced Video Dithering App (Enhanced)")
        master.geometry("1600x900")
        master.minsize(1200, 800)

        # ---------------- DARK THEME STYLING ----------------
        self.dark_bg = "#1c1c1c"
        self.dark_fg = "#cae8e0"
        self.accent_color = "#011f38"
        self.entry_bg = "#141417"

        master.configure(bg=self.dark_bg)
        style = ttk.Style(master)
        style.theme_use("clam")

        # ------------------ App Variables ------------------
        self.input_video_path = tk.StringVar()
        self.output_dir_path = tk.StringVar()
        self.selected_dither = tk.StringVar(value="Floyd-Steinberg")
        self.selected_palette = []
        self.brightness = tk.IntVar(value=0)
        self.contrast = tk.DoubleVar(value=1.0)
        self.saturation = tk.DoubleVar(value=1.0)
        self.hue = tk.DoubleVar(value=0.0)
        self.dither_strength = tk.DoubleVar(value=100.0)
        self.live_preview = tk.BooleanVar(value=False)
        self.preserve_original = tk.BooleanVar(value=False)
        self.preserve_threshold = tk.DoubleVar(value=25.0)
        self.res_scale = tk.DoubleVar(value=1.0)
        self.lock_zoom = tk.BooleanVar(value=False)
        self.noise_reduction = tk.BooleanVar(value=False)
        self.palette_disabled = tk.BooleanVar(value=False)
        self.palette_method = tk.StringVar(value="Fast (OpenCV)")
        self.resample_each_frame = tk.BooleanVar(value=True)
        self.resample_interval = tk.IntVar(value=1)
        self.palette_size = tk.IntVar(value=8)
        self.kmeans_algorithm = tk.StringVar(value="elkan")
        self.kmeans_inits = tk.IntVar(value=1)
        self.kmeans_downsample = tk.IntVar(value=128)
        # New custom resolution input (format: WIDTHxHEIGHT)
        self.custom_resolution = tk.StringVar(value="")
        self.current_color_index = None
        self.color_r = tk.IntVar(value=0)
        self.color_g = tk.IntVar(value=0)
        self.color_b = tk.IntVar(value=0)
        self.color_hex = tk.StringVar(value="#000000")
        self.total_frames = 0
        self.current_preview_frame = 0
        self.preview_image = None
        self.preview_pil = None
        self.offset_x = 0
        self.offset_y = 0
        self.zoom_factor = 1.0
        self.last_mouse_x = None
        self.last_mouse_y = None

        # Presets (configuration)
        self.presets = {}
        self.preset_var = tk.StringVar()
        self.load_presets()

        # ------------------ Menu Bar ------------------
        self.menu_bar = tk.Menu(master)
        master.config(menu=self.menu_bar)
        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        file_menu.add_command(label="Open Video...", command=self.browse_input)
        file_menu.add_command(label="Select Output Directory...", command=self.browse_output)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=master.quit)
        self.menu_bar.add_cascade(label="File", menu=file_menu)
        settings_menu = tk.Menu(self.menu_bar, tearoff=0)
        settings_menu.add_command(label="Save Settings", command=self.save_settings)
        settings_menu.add_command(label="Load Settings", command=self.load_settings)
        settings_menu.add_command(label="Reset Settings", command=self.reset_settings)
        self.menu_bar.add_cascade(label="Settings", menu=settings_menu)
        help_menu = tk.Menu(self.menu_bar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        self.menu_bar.add_cascade(label="Help", menu=help_menu)

        # ------------------ Layout Frames ------------------
        self.leftFrame = ttk.Frame(master, width=450)
        self.centerFrame = ttk.Frame(master)
        self.rightFrame = ttk.Frame(master, width=450)
        self.leftFrame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.centerFrame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.rightFrame.grid(row=0, column=2, sticky="nsew", padx=10, pady=10)
        master.grid_columnconfigure(0, weight=0)
        master.grid_columnconfigure(1, weight=1)
        master.grid_columnconfigure(2, weight=0)
        master.grid_rowconfigure(0, weight=1)

        self.build_left_frame()
        self.build_center_frame()
        self.build_right_frame()

    # ------------- Preset (Settings) Functions -------------
    def load_presets(self):
        if os.path.exists("presets.json"):
            with open("presets.json", "r") as f:
                presets = json.load(f)
            # Ensure presets is a dictionary
            if not isinstance(presets, dict):
                presets = {}
            else:
                # Convert any preset stored as a list (old format) into the new dict format
                for key, value in presets.items():
                    if not isinstance(value, dict):
                        presets[key] = {
                            "brightness": 0,
                            "contrast": 1.0,
                            "saturation": 1.0,
                            "hue": 0.0,
                            "dither_strength": 100,
                            "res_scale": 1.0,
                            "palette_size": 8,
                            "selected_dither": "Floyd-Steinberg",
                            "preserve_original": False,
                            "noise_reduction": False,
                            "selected_palette": value  # old list becomes the palette
                        }
            self.presets = presets
        else:
            self.presets = {}

    def save_current_as_preset(self):
        preset_name = self.preset_var.get()
        if preset_name:
            # Save all current settings as a dictionary
            self.presets[preset_name] = {
                "brightness": self.brightness.get(),
                "contrast": self.contrast.get(),
                "saturation": self.saturation.get(),
                "hue": self.hue.get(),
                "dither_strength": self.dither_strength.get(),
                "res_scale": self.res_scale.get(),
                "palette_size": self.palette_size.get(),
                "selected_dither": self.selected_dither.get(),
                "preserve_original": self.preserve_original.get(),
                "noise_reduction": self.noise_reduction.get(),
                "selected_palette": self.selected_palette
            }
            self.save_presets()

    def apply_preset(self, preset_name):
        if preset_name in self.presets:
            settings = self.presets[preset_name]
            # If settings is still a list (old format), convert it now:
            if isinstance(settings, list):
                settings = {
                    "brightness": 0,
                    "contrast": 1.0,
                    "saturation": 1.0,
                    "hue": 0.0,
                    "dither_strength": 100,
                    "res_scale": 1.0,
                    "palette_size": 8,
                    "selected_dither": "Floyd-Steinberg",
                    "preserve_original": False,
                    "noise_reduction": False,
                    "selected_palette": settings
                }
            self.brightness.set(settings.get("brightness", 0))
            self.contrast.set(settings.get("contrast", 1.0))
            self.saturation.set(settings.get("saturation", 1.0))
            self.hue.set(settings.get("hue", 0.0))
            self.dither_strength.set(settings.get("dither_strength", 100))
            self.res_scale.set(settings.get("res_scale", 1.0))
            self.palette_size.set(settings.get("palette_size", 8))
            self.selected_dither.set(settings.get("selected_dither", "Floyd-Steinberg"))
            self.preserve_original.set(settings.get("preserve_original", False))
            self.noise_reduction.set(settings.get("noise_reduction", False))
            self.selected_palette = settings.get("selected_palette", [])
            for i, color in enumerate(self.selected_palette):
                if i < len(self.palette_buttons):
                    self.palette_buttons[i].configure(background=color)
            self.on_live_preview()

    def save_settings(self):
        settings = {
            "brightness": self.brightness.get(),
            "contrast": self.contrast.get(),
            "saturation": self.saturation.get(),
            "hue": self.hue.get(),
            "dither_strength": self.dither_strength.get(),
            "res_scale": self.res_scale.get(),
            "palette_size": self.palette_size.get(),
            "selected_dither": self.selected_dither.get(),
            "preserve_original": self.preserve_original.get(),
            "noise_reduction": self.noise_reduction.get(),
            "selected_palette": self.selected_palette
        }
        file_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, "w") as f:
                json.dump(settings, f)
            messagebox.showinfo("Settings Saved", "Configuration saved successfully.")

    def load_settings(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, "r") as f:
                settings = json.load(f)
            self.brightness.set(settings.get("brightness", 0))
            self.contrast.set(settings.get("contrast", 1.0))
            self.saturation.set(settings.get("saturation", 1.0))
            self.hue.set(settings.get("hue", 0.0))
            self.dither_strength.set(settings.get("dither_strength", 100))
            self.res_scale.set(settings.get("res_scale", 1.0))
            self.palette_size.set(settings.get("palette_size", 8))
            self.selected_dither.set(settings.get("selected_dither", "Floyd-Steinberg"))
            self.preserve_original.set(settings.get("preserve_original", False))
            self.noise_reduction.set(settings.get("noise_reduction", False))
            self.selected_palette = settings.get("selected_palette", [])
            for i, color in enumerate(self.selected_palette):
                if i < len(self.palette_buttons):
                    self.palette_buttons[i].configure(background=color)
            self.on_live_preview()
            messagebox.showinfo("Settings Loaded", "Configuration loaded successfully.")

    def reset_settings(self):
        self.brightness.set(0)
        self.contrast.set(1.0)
        self.saturation.set(1.0)
        self.hue.set(0.0)
        self.dither_strength.set(100)
        self.res_scale.set(1.0)
        self.palette_size.set(8)
        self.selected_dither.set("Floyd-Steinberg")
        self.preserve_original.set(False)
        self.noise_reduction.set(False)
        self.selected_palette = []
        for btn in self.palette_buttons:
            btn.configure(background="SystemButtonFace")
        self.on_live_preview()
        messagebox.showinfo("Reset", "Settings have been reset to default.")

    def show_about(self):
        messagebox.showinfo("About", "Advanced Video Dithering App (Enhanced)\n\nDeveloped by Sir Samuel The Great\n\nFeatures:\n• Multiple dithering algorithms (including Retro Shading and Random Dithering)\n• Advanced image adjustments\n• Customizable palettes with KMeans extraction\n• Live preview with pan & zoom\n• Save/Load/Reset configuration\n• Custom resolution input")

    def on_palette_size_change(self, event=None):
        if self.live_preview.get():
            self.load_preview()

    # ------------------ Build UI Sections ------------------

    def build_left_frame(self):
        header = ttk.Label(self.leftFrame, text="Input / Adjustments", font=("Helvetica", 16, "bold"))
        header.pack(pady=10)
        ttk.Button(self.leftFrame, text="Select Input File", command=self.browse_input).pack(pady=5)
        ttk.Entry(self.leftFrame, textvariable=self.input_video_path, width=40).pack(pady=5)
        ttk.Button(self.leftFrame, text="Select Output Directory", command=self.browse_output).pack(pady=5)
        ttk.Entry(self.leftFrame, textvariable=self.output_dir_path, width=40).pack(pady=5)

        # Preset Controls
        preset_frame = ttk.Frame(self.leftFrame)
        preset_frame.pack(pady=5)
        ttk.Label(preset_frame, text="Presets:").pack(side=tk.LEFT, padx=5)
        preset_dropdown = ttk.Combobox(preset_frame, textvariable=self.preset_var,
                                       values=list(self.presets.keys()), state="readonly", width=18)
        preset_dropdown.pack(side=tk.LEFT, padx=5)
        preset_dropdown.bind("<<ComboboxSelected>>", lambda e: self.apply_preset(self.preset_var.get()))
        ttk.Button(preset_frame, text="Save Preset", command=self.save_current_as_preset).pack(side=tk.LEFT, padx=5)

        # Dithering Options
        sep = ttk.Separator(self.leftFrame, orient=tk.HORIZONTAL)
        sep.pack(fill=tk.X, pady=10)
        ttk.Label(self.leftFrame, text="Dithering Options", font=("Helvetica", 16, "bold")).pack(pady=5)
        dith_dropdown = ttk.Combobox(self.leftFrame, textvariable=self.selected_dither,
                                     values=list(DITHERING_TYPES.keys()) + ["Indexed (Palette-based)"],
                                     state="readonly", width=28)
        dith_dropdown.pack(pady=5)
        dith_dropdown.bind("<<ComboboxSelected>>", self.on_live_preview)

        # Strength: Scale + Spinbox
        strength_frame = ttk.Frame(self.leftFrame)
        strength_frame.pack(pady=10, fill=tk.X)
        ttk.Label(strength_frame, text="Strength (0-100):").grid(row=0, column=0, sticky="w")
        self.strength_scale = tk.Scale(strength_frame, from_=0, to=100, variable=self.dither_strength,
                                       orient=tk.HORIZONTAL, length=200, command=self.on_live_preview,
                                       bg=self.dark_bg, fg=self.dark_fg, troughcolor="#141417")
        self.strength_scale.grid(row=0, column=1, padx=5)
        self.strength_spin = tk.Spinbox(strength_frame, from_=0, to=100, width=5, textvariable=self.dither_strength,
                                       command=lambda: self.on_strength_entry_change(None),
                                       background=self.entry_bg, foreground=self.dark_fg)
        self.strength_spin.grid(row=0, column=2, padx=5)

        ttk.Button(self.leftFrame, text="Start Dithering", command=self.start_dithering).pack(pady=15)
        self.progress = ttk.Progressbar(self.leftFrame, orient=tk.HORIZONTAL, length=300, mode='determinate')
        self.progress.pack(pady=10)
        self.status_label = ttk.Label(self.leftFrame, text="Status: Idle", font=("Helvetica", 12))
        self.status_label.pack(pady=5)

        # Image Adjustment Controls
        adj_frame = ttk.LabelFrame(self.leftFrame, text="Image Adjustments", padding=10)
        adj_frame.pack(pady=10, fill=tk.X)

        # Brightness
        bright_frame = ttk.Frame(adj_frame)
        bright_frame.pack(pady=5, fill=tk.X)
        ttk.Label(bright_frame, text="Brightness (-100 to 100):").grid(row=0, column=0, sticky="w")
        self.bright_scale = tk.Scale(bright_frame, from_=-100, to=100, variable=self.brightness,
                                     orient=tk.HORIZONTAL, length=150,
                                     command=self.on_live_preview,
                                     bg=self.dark_bg, fg=self.dark_fg, troughcolor="#141417")
        self.bright_scale.grid(row=0, column=1, padx=5)
        self.bright_spin = tk.Spinbox(bright_frame, from_=-100, to=100, width=5, textvariable=self.brightness,
                                      command=lambda: self.on_brightness_entry_change(None),
                                      background=self.entry_bg, foreground=self.dark_fg)
        self.bright_spin.grid(row=0, column=2, padx=5)

        # Contrast
        cont_frame = ttk.Frame(adj_frame)
        cont_frame.pack(pady=5, fill=tk.X)
        ttk.Label(cont_frame, text="Contrast (0.5 to 3.0):").grid(row=0, column=0, sticky="w")
        self.cont_scale = tk.Scale(cont_frame, from_=0.5, to=3.0, resolution=0.1, variable=self.contrast,
                                   orient=tk.HORIZONTAL, length=150,
                                   command=self.on_live_preview,
                                   bg=self.dark_bg, fg=self.dark_fg, troughcolor="#141417")
        self.cont_scale.grid(row=0, column=1, padx=5)
        self.cont_spin = tk.Spinbox(cont_frame, from_=0.5, to=3.0, increment=0.1, width=5, textvariable=self.contrast,
                                   command=lambda: self.on_contrast_entry_change(None),
                                   background=self.entry_bg, foreground=self.dark_fg)
        self.cont_spin.grid(row=0, column=2, padx=5)

        # Saturation
        sat_frame = ttk.Frame(adj_frame)
        sat_frame.pack(pady=5, fill=tk.X)
        ttk.Label(sat_frame, text="Saturation (0.0 to 3.0):").grid(row=0, column=0, sticky="w")
        self.sat_scale = tk.Scale(sat_frame, from_=0.0, to=3.0, resolution=0.1, variable=self.saturation,
                                  orient=tk.HORIZONTAL, length=150,
                                  command=self.on_live_preview,
                                  bg=self.dark_bg, fg=self.dark_fg, troughcolor="#141417")
        self.sat_scale.grid(row=0, column=1, padx=5)
        self.sat_spin = tk.Spinbox(sat_frame, from_=0.0, to=3.0, increment=0.1, width=5, textvariable=self.saturation,
                                  command=lambda: self.on_saturation_entry_change(None),
                                  background=self.entry_bg, foreground=self.dark_fg)
        self.sat_spin.grid(row=0, column=2, padx=5)

        # Hue
        hue_frame = ttk.Frame(adj_frame)
        hue_frame.pack(pady=5, fill=tk.X)
        ttk.Label(hue_frame, text="Hue (-180 to 180):").grid(row=0, column=0, sticky="w")
        self.hue_scale = tk.Scale(hue_frame, from_=-180, to=180, resolution=1, variable=self.hue,
                                  orient=tk.HORIZONTAL, length=150,
                                  command=self.on_live_preview,
                                  bg=self.dark_bg, fg=self.dark_fg, troughcolor="#141417")
        self.hue_scale.grid(row=0, column=1, padx=5)
        self.hue_spin = tk.Spinbox(hue_frame, from_=-180, to=180, width=5, textvariable=self.hue,
                                  command=lambda: self.on_hue_entry_change(None),
                                  background=self.entry_bg, foreground=self.dark_fg)
        self.hue_spin.grid(row=0, column=2, padx=5)

        # Resolution Scale and Custom Resolution Input
        res_frame = ttk.Frame(adj_frame)
        res_frame.pack(pady=5, fill=tk.X)
        ttk.Label(res_frame, text="Resolution Scale (0.5 - 2.0):").grid(row=0, column=0, sticky="w")
        self.res_scale_slider = tk.Scale(res_frame, from_=0.5, to=2.0, resolution=0.1, variable=self.res_scale,
                                         orient=tk.HORIZONTAL, length=150,
                                         command=self.on_live_preview,
                                         bg=self.dark_bg, fg=self.dark_fg, troughcolor="#141417")
        self.res_scale_slider.grid(row=0, column=1, padx=5)
        self.res_spin = tk.Spinbox(res_frame, from_=0.5, to=2.0, increment=0.1, width=5, textvariable=self.res_scale,
                                   command=lambda: self.on_res_entry_change(None),
                                   background=self.entry_bg, foreground=self.dark_fg)
        self.res_spin.grid(row=0, column=2, padx=5)
        ttk.Label(res_frame, text="Custom Resolution (e.g., 800x600):").grid(row=1, column=0, sticky="w", pady=(5,0))
        self.custom_res_entry = ttk.Entry(res_frame, textvariable=self.custom_resolution, width=15)
        self.custom_res_entry.grid(row=1, column=1, padx=5, pady=(5,0))

    def build_center_frame(self):
        self.preview_canvas = tk.Canvas(self.centerFrame, bg="#2D2D30")
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        self.preview_canvas.bind("<Button-1>", self.on_left_button_down)
        self.preview_canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.preview_canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.preview_canvas.bind("<Button-4>", self.on_mouse_wheel)
        self.preview_canvas.bind("<Button-5>", self.on_mouse_wheel)
        bottom_frame = ttk.Frame(self.centerFrame)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        ttk.Button(bottom_frame, text="Load Preview Frame", command=self.load_preview).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(bottom_frame, text="Live Preview", variable=self.live_preview,
                        command=self.on_live_toggle).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(bottom_frame, text="Lock Zoom", variable=self.lock_zoom,
                        command=self.on_lock_zoom_toggle).pack(side=tk.LEFT, padx=10)
        self.frame_slider = tk.Scale(bottom_frame, from_=0, to=0, orient=tk.HORIZONTAL, length=400,
                                     command=self.on_frame_slider_change,
                                     bg=self.dark_bg, fg=self.dark_fg, troughcolor="#141417")
        self.frame_slider.pack(side=tk.LEFT, padx=10)

    def build_right_frame(self):
        container = ttk.Frame(self.rightFrame)
        container.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(container, background=self.dark_bg)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")

        def on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        pal_group = ttk.LabelFrame(scroll_frame, text="Palette / Preserve", padding=10)
        pal_group.pack(pady=10, padx=10, fill=tk.X)
        ttk.Checkbutton(pal_group, text="Preserve Original Colors", variable=self.preserve_original,
                        command=self.on_live_preview).pack(pady=5, anchor="w")
        dist_frame = ttk.Frame(pal_group)
        dist_frame.pack(pady=5, fill=tk.X)
        ttk.Label(dist_frame, text="Preserve Threshold (0-255):").pack(side=tk.LEFT)
        self.dist_scale = tk.Scale(dist_frame, from_=0, to=255, variable=self.preserve_threshold,
                                   orient=tk.HORIZONTAL, length=150, command=self.on_live_preview,
                                   bg=self.dark_bg, fg=self.dark_fg, troughcolor="#141417")
        self.dist_scale.pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(pal_group, text="Apply Noise Reduction", variable=self.noise_reduction,
                        command=self.on_live_preview).pack(pady=5, anchor="w")
        meth_group = ttk.LabelFrame(scroll_frame, text="Dynamic Palette & KMeans", padding=10)
        meth_group.pack(pady=10, padx=10, fill=tk.X)
        ttk.Label(meth_group, text="Palette Extraction:").pack(anchor="w", pady=5)
        palette_method_dropdown = ttk.Combobox(meth_group, textvariable=self.palette_method,
                                               values=["Fast (OpenCV)", "Accurate (scikit-learn)"],
                                               state="readonly", width=20)
        palette_method_dropdown.pack(pady=5)
        palette_method_dropdown.bind("<<ComboboxSelected>>", self.on_live_preview)
        ttk.Checkbutton(meth_group, text="Resample Palette Each Frame", variable=self.resample_each_frame,
                        command=self.on_live_preview).pack(pady=5, anchor="w")
        sample_frame = ttk.Frame(meth_group)
        sample_frame.pack(pady=5, fill=tk.X)
        ttk.Label(sample_frame, text="Colors to Sample:").pack(side=tk.LEFT)
        # KMeans Max Iterations
        kmeans_iter_frame = ttk.Frame(meth_group)
        kmeans_iter_frame.pack(pady=5, fill=tk.X)
        ttk.Label(kmeans_iter_frame, text="KMeans Max Iterations:").pack(side=tk.LEFT, padx=5)
        self.kmeans_max_iter = tk.IntVar(value=300)  # Default value (adjust as needed)
        kmeans_iter_spin = tk.Spinbox(kmeans_iter_frame, from_=100, to=1000, increment=50, width=5, textvariable=self.kmeans_max_iter)
        kmeans_iter_spin.pack(side=tk.LEFT, padx=5)

        # KMeans Tolerance
        kmeans_tol_frame = ttk.Frame(meth_group)
        kmeans_tol_frame.pack(pady=5, fill=tk.X)
        ttk.Label(kmeans_tol_frame, text="KMeans Tolerance:").pack(side=tk.LEFT, padx=5)
        self.kmeans_tol = tk.DoubleVar(value=1e-4)  # Default value (adjust as needed)
        kmeans_tol_spin = tk.Spinbox(kmeans_tol_frame, from_=1e-6, to=1e-2, format="%.6f", increment=1e-5, width=8, textvariable=self.kmeans_tol)
        kmeans_tol_spin.pack(side=tk.LEFT, padx=5)

        # Palette Seed (new setting)
        seed_frame = ttk.Frame(meth_group)
        seed_frame.pack(pady=5, fill=tk.X)
        ttk.Label(seed_frame, text="Palette Seed:").pack(side=tk.LEFT, padx=5)
        self.palette_seed = tk.IntVar(value=0)  # 0 means no fixed seed; nonzero fixes the random state
        seed_spin = tk.Spinbox(seed_frame, from_=0, to=100000, width=8, textvariable=self.palette_seed)
        seed_spin.pack(side=tk.LEFT, padx=5)
        self.palette_size_scale = tk.Scale(sample_frame, from_=2, to=256, variable=self.palette_size,
                                           orient=tk.HORIZONTAL, length=150, command=self.on_palette_size_change,
                                           bg=self.dark_bg, fg=self.dark_fg, troughcolor="#141417")
        self.palette_size_scale.pack(side=tk.LEFT, padx=5)
        ttk.Checkbutton(meth_group, text="Disable Custom Palette", variable=self.palette_disabled,
                        command=self.on_live_preview).pack(pady=5, anchor="w")
        ttk.Button(meth_group, text="Reset Palette", command=self.reset_settings).pack(pady=5)
        color_group = ttk.LabelFrame(scroll_frame, text="Select Palette Colors (Up to 8)", padding=10)
        color_group.pack(pady=10, padx=10, fill=tk.X)
        self.palette_buttons = []
        pal_frame = ttk.Frame(color_group)
        pal_frame.pack()
        for i in range(8):
            btn = tk.Button(pal_frame, text=f"Color {i+1}",
                            command=lambda i=i: self.choose_color(i),
                            width=12, font=("Helvetica", 10),
                            background="SystemButtonFace")
            btn.grid(row=i//2, column=i%2, padx=5, pady=5)
            self.palette_buttons.append(btn)

        # Color Editor (remove the pack_forget so it stays visible)
        self.color_editor_frame = ttk.Frame(color_group)
        self.color_editor_frame.pack(pady=10, fill=tk.X)
        ttk.Label(self.color_editor_frame, text="Color Editor", font=("Helvetica", 14, "bold")).pack(pady=5)
        slider_frame = ttk.Frame(self.color_editor_frame)
        slider_frame.pack()
        ttk.Label(slider_frame, text="R:").grid(row=0, column=0, padx=(0,5), sticky="e")
        scale_r = tk.Scale(slider_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                           variable=self.color_r, command=self.on_color_slider_change, length=180,
                           bg=self.dark_bg, fg=self.dark_fg, troughcolor="#141417")
        scale_r.grid(row=0, column=1, sticky="w")
        ttk.Label(slider_frame, text="G:").grid(row=1, column=0, padx=(0,5), sticky="e")
        scale_g = tk.Scale(slider_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                           variable=self.color_g, command=self.on_color_slider_change, length=180,
                           bg=self.dark_bg, fg=self.dark_fg, troughcolor="#141417")
        scale_g.grid(row=1, column=1, sticky="w")
        ttk.Label(slider_frame, text="B:").grid(row=2, column=0, padx=(0,5), sticky="e")
        scale_b = tk.Scale(slider_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                           variable=self.color_b, command=self.on_color_slider_change, length=180,
                           bg=self.dark_bg, fg=self.dark_fg, troughcolor="#141417")
        scale_b.grid(row=2, column=1, sticky="w")
        hex_frame = ttk.Frame(self.color_editor_frame)
        hex_frame.pack(pady=5)
        ttk.Label(hex_frame, text="Hex (#RRGGBB):").pack(side=tk.LEFT, padx=5)
        hex_entry = ttk.Entry(hex_frame, textvariable=self.color_hex, width=10)
        hex_entry.pack(side=tk.LEFT)
        hex_entry.bind("<Return>", self.on_hex_entry_change)
        apply_frame = ttk.Frame(self.color_editor_frame)
        apply_frame.pack(pady=5)
        self.color_preview_label = tk.Label(apply_frame, text="      ", background="#000000", width=10)
        self.color_preview_label.pack(side=tk.LEFT, padx=10)
        ttk.Button(apply_frame, text="Apply", command=self.apply_color).pack(side=tk.LEFT, padx=10)

    # ------------------ Event Handlers ------------------

    def on_strength_entry_change(self, event):
        try:
            val = float(self.strength_spin.get())
            if 0 <= val <= 100:
                self.dither_strength.set(val)
        except ValueError:
            pass
        if self.live_preview.get():
            self.load_preview()

    def on_brightness_entry_change(self, event):
        try:
            val = float(self.bright_spin.get())
            if -100 <= val <= 100:
                self.brightness.set(val)
        except ValueError:
            pass
        if self.live_preview.get():
            self.load_preview()

    def on_contrast_entry_change(self, event):
        try:
            val = float(self.cont_spin.get())
            if 0.5 <= val <= 3.0:
                self.contrast.set(val)
        except ValueError:
            pass
        if self.live_preview.get():
            self.load_preview()

    def on_saturation_entry_change(self, event):
        try:
            val = float(self.sat_spin.get())
            if 0.0 <= val <= 3.0:
                self.saturation.set(val)
        except ValueError:
            pass
        if self.live_preview.get():
            self.load_preview()

    def on_hue_entry_change(self, event):
        try:
            val = float(self.hue_spin.get())
            if -180 <= val <= 180:
                self.hue.set(val)
        except ValueError:
            pass
        if self.live_preview.get():
            self.load_preview()

    def on_res_entry_change(self, event):
        try:
            val = float(self.res_spin.get())
            if 0.5 <= val <= 2.0:
                self.res_scale.set(val)
        except ValueError:
            pass
        if self.live_preview.get():
            self.load_preview()

    def on_lock_zoom_toggle(self):
        if self.lock_zoom.get():
            self.status_label.config(text="Status: Zoom Locked")
        else:
            self.status_label.config(text="Status: Zoom Unlocked")

    def on_live_toggle(self):
        if self.live_preview.get():
            self.load_preview()

    def on_live_preview(self, event=None):
        if self.live_preview.get():
            self.load_preview()

    def on_frame_slider_change(self, value):
        frame_index = int(float(value))
        self.current_preview_frame = frame_index
        if self.live_preview.get():
            self.load_preview(frame_index=frame_index)

    def on_color_slider_change(self, event=None):
        r = self.color_r.get()
        g = self.color_g.get()
        b = self.color_b.get()
        hexval = f"#{r:02X}{g:02X}{b:02X}"
        self.color_hex.set(hexval)
        self.color_preview_label.configure(background=hexval)
        if self.current_color_index is not None:
            while len(self.selected_palette) <= self.current_color_index:
                self.selected_palette.append("#000000")
            self.selected_palette[self.current_color_index] = hexval
            self.palette_buttons[self.current_color_index].configure(background=hexval)
            if self.live_preview.get():
                self.load_preview()

    def on_hex_entry_change(self, event):
        val = self.color_hex.get().strip()
        if val.startswith("#"):
            val = val[1:]
        if len(val) == 6:
            try:
                r = int(val[0:2], 16)
                g = int(val[2:4], 16)
                b = int(val[4:6], 16)
                self.color_r.set(r)
                self.color_g.set(g)
                self.color_b.set(b)
                self.color_preview_label.configure(background=f"#{val.upper()}")
                if self.current_color_index is not None:
                    while len(self.selected_palette) <= self.current_color_index:
                        self.selected_palette.append("#000000")
                    new_hex = f"#{val.upper()}"
                    self.selected_palette[self.current_color_index] = new_hex
                    self.palette_buttons[self.current_color_index].configure(background=new_hex)
                if self.live_preview.get():
                    self.load_preview()
            except ValueError:
                pass

    def apply_color(self):
        if self.current_color_index is None:
            return
        if self.live_preview.get():
            self.load_preview()

    def browse_input(self):
        file_path = filedialog.askopenfilename(
            title="Select Video or Image",
            filetypes=[
                ("Supported files", "*.mp4 *.png *.jpg *.jpeg *.bmp *.gif *.tiff"),
                ("MP4 files", "*.mp4"),
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff")
            ]
        )
        if file_path:
            self.input_video_path.set(file_path)
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            else:
                self.total_frames = 0
            cap.release()
            self.current_preview_frame = 0
            if self.total_frames > 0:
                self.frame_slider.config(to=self.total_frames - 1)
            else:
                self.frame_slider.config(to=0)
            self.frame_slider.set(0)

    def browse_output(self):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir_path.set(directory)

    def choose_color(self, index):
        # Set the current color index so slider callbacks know which palette entry to update.
        self.current_color_index = index
        # If a color is already chosen (and not the default), load its value into the sliders.
        if index < len(self.selected_palette) and self.selected_palette[index] not in ("SystemButtonFace", "", None):
            hex_val = self.selected_palette[index]
            r = int(hex_val[1:3], 16)
            g = int(hex_val[3:5], 16)
            b = int(hex_val[5:7], 16)
            self.color_r.set(r)
            self.color_g.set(g)
            self.color_b.set(b)
            self.color_hex.set(hex_val)
            # Ensure the color editor is visible
            self.color_editor_frame.pack(pady=10, fill=tk.X)
            if self.live_preview.get():
                self.on_live_preview()
        else:
            # No color chosen yet, so open the color chooser
            color_code = colorchooser.askcolor(title=f"Choose Palette Color {index+1}")
            if color_code[1] is not None:
                if index < len(self.selected_palette):
                    self.selected_palette[index] = color_code[1]
                else:
                    self.selected_palette.append(color_code[1])
                self.palette_buttons[index].configure(background=color_code[1])
                hex_val = color_code[1]
                r = int(hex_val[1:3], 16)
                g = int(hex_val[3:5], 16)
                b = int(hex_val[5:7], 16)
                self.color_r.set(r)
                self.color_g.set(g)
                self.color_b.set(b)
                self.color_hex.set(hex_val)
                self.color_editor_frame.pack(pady=10, fill=tk.X)
                if self.live_preview.get():
                    self.on_live_preview()

    def load_preview(self, frame_index=None):
        if frame_index is not None:
            self.current_preview_frame = frame_index
        input_path = self.input_video_path.get()
        if not input_path or not os.path.isfile(input_path):
            self.status_label.config(text="Status: Please select a valid input file.")
            return
        ext = os.path.splitext(input_path)[1].lower()
        IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"]
        VIDEO_EXTS = [".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv"]
        if ext in IMAGE_EXTS:
            frame = cv2.imread(input_path)
            if frame is None:
                self.status_label.config(text="Status: Cannot read image file.")
                return
        elif ext in VIDEO_EXTS:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                self.status_label.config(text=f"Status: Cannot open video file {input_path}")
                return
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_preview_frame)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                self.status_label.config(text="Status: Cannot read frame from video.")
                return
        else:
            self.status_label.config(text=f"Status: Unsupported file extension: {ext}")
            return

        # Apply custom resolution if provided (format: WIDTHxHEIGHT)
        custom_res = self.custom_resolution.get().strip()
        if custom_res:
            try:
                width, height = map(int, custom_res.lower().split('x'))
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            except Exception:
                sf = self.res_scale.get()
                new_w = int(frame.shape[1] * sf)
                new_h = int(frame.shape[0] * sf)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            sf = self.res_scale.get()
            if abs(sf - 1.0) > 1e-3:
                new_w = int(frame.shape[1] * sf)
                new_h = int(frame.shape[0] * sf)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Apply image adjustments (these functions expect BGR input)
        frame = adjust_brightness(frame, self.brightness.get())
        frame = adjust_contrast(frame, self.contrast.get())
        frame = adjust_saturation(frame, self.saturation.get())
        frame = adjust_hue(frame, self.hue.get())
        interval = self.resample_interval.get()

        # Dynamic Palette Extraction
        if not self.palette_disabled.get():
            # Convert frame to RGB for palette extraction
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.palette_method.get() == "Accurate (scikit-learn)":
                palette_func = extract_palette_accurate_kmeans
                palette_params = {
                    "num_colors": self.palette_size.get(),
                    "max_dim": self.kmeans_downsample.get(),
                    "kmeans_algorithm": self.kmeans_algorithm.get(),
                    "kmeans_inits": self.kmeans_inits.get(),
                    "kmeans_max_iter": self.kmeans_max_iter.get(),  # new setting
                    "kmeans_tol": self.kmeans_tol.get()             # new setting
                }
            else:  # "Fast (OpenCV)"
                palette_func = extract_palette_fast_opencv  # Ensure this function is implemented
                palette_params = {
                    "num_colors": self.palette_size.get(),
                    "max_dim": self.kmeans_downsample.get()
                }
            if self.resample_each_frame.get():
                if self.current_preview_frame == 0 or (self.current_preview_frame % interval == 0):
                    rgb_palette = palette_func(frame_rgb, **palette_params)
                    # Convert extracted RGB palette back to BGR for dithering
                    self.cached_preview_palette = [ (b, g, r) for (r, g, b) in rgb_palette ]
            else:
                if self.current_preview_frame == 0 or not hasattr(self, 'cached_preview_palette'):
                    rgb_palette = palette_func(frame_rgb, **palette_params)
                    self.cached_preview_palette = [ (b, g, r) for (r, g, b) in rgb_palette ]
            dynamic_palette = self.cached_preview_palette
        else:
            # Use custom palette if dynamic palette is disabled.
            dynamic_palette = []
            for hexcol in self.selected_palette:
                r = int(hexcol[1:3], 16)
                g = int(hexcol[3:5], 16)
                b = int(hexcol[5:7], 16)
                dynamic_palette.append((r, g, b))
            if not dynamic_palette:
                dynamic_palette = [(0, 0, 0), (255, 255, 255)]

        # Dithering
        method = self.selected_dither.get()
        dith_func = DITHERING_TYPES.get(method, floyd_steinberg_dither)
        try:
            dithered = dith_func(frame.copy(), dynamic_palette, self.dither_strength.get(),
                                 preserve=self.preserve_original.get(),
                                 preserve_threshold=self.preserve_threshold.get())
        except Exception as e:
            self.status_label.config(text=f"Error during dithering: {e}")
            return
        if self.noise_reduction.get():
            dithered = apply_noise_reduction(dithered)
        # Convert result from BGR to RGB for display
        disp_rgb = cv2.cvtColor(dithered, cv2.COLOR_BGR2RGB)
        self.preview_pil = Image.fromarray(disp_rgb)
        if not self.lock_zoom.get():
            self.offset_x = 0
            self.offset_y = 0
            self.zoom_factor = 1.0
        self.draw_preview()
        self.status_label.config(text=f"Status: Preview Loaded (Frame {self.current_preview_frame})")

    def draw_preview(self):
        if self.preview_pil is None:
            return
        new_w = max(1, int(self.preview_pil.width * self.zoom_factor))
        new_h = max(1, int(self.preview_pil.height * self.zoom_factor))
        resized = self.preview_pil.resize((new_w, new_h), Image.LANCZOS)
        self.preview_image = ImageTk.PhotoImage(resized)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.preview_image)

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
        if hasattr(event, "delta") and event.delta != 0:
            zoom_change = 1.1 if event.delta > 0 else 0.9
        elif hasattr(event, "num"):
            if event.num == 4:
                zoom_change = 1.1
            elif event.num == 5:
                zoom_change = 0.9
            else:
                zoom_change = 1.0
        else:
            zoom_change = 1.0
        new_zoom = self.zoom_factor * zoom_change
        new_zoom = max(0.1, min(new_zoom, 10.0))
        zoom_change = new_zoom / self.zoom_factor
        self.zoom_factor = new_zoom
        mx, my = event.x, event.y
        self.offset_x -= (mx - self.offset_x) * (zoom_change - 1)
        self.offset_y -= (my - self.offset_y) * (zoom_change - 1)
        self.draw_preview()

    def start_dithering(self):
        input_video = self.input_video_path.get()
        output_dir = self.output_dir_path.get()
        if not input_video or not os.path.isfile(input_video):
            messagebox.showerror("Error", "Please select a valid input video.")
            return
        if not output_dir or not os.path.isdir(output_dir):
            messagebox.showerror("Error", "Please select a valid output directory.")
            return
        if not self.selected_palette:
            self.selected_palette = ["#000000", "#FFFFFF"]
        self.set_ui_state(False)
        self.progress['value'] = 0
        self.status_label.config(text="Status: Processing...")
        file_ext = os.path.splitext(input_video)[1].lower()
        if file_ext in [".mp4", ".mov", ".avi", ".mkv"]:
            threading.Thread(target=self.process_video, args=(input_video, output_dir), daemon=True).start()
        else:
            threading.Thread(target=self.process_image, args=(input_video, output_dir), daemon=True).start()

    def process_image(self, input_path, output_dir):
        try:
            frame = cv2.imread(input_path)
            if frame is None:
                raise Exception(f"Cannot read image file {input_path}")
            custom_res = self.custom_resolution.get().strip()
            if custom_res:
                try:
                    width, height = map(int, custom_res.lower().split('x'))
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                except Exception:
                    sf = self.res_scale.get()
                    new_w = int(frame.shape[1] * sf)
                    new_h = int(frame.shape[0] * sf)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                sf = self.res_scale.get()
                if abs(sf - 1.0) > 1e-3:
                    new_w = int(frame.shape[1] * sf)
                    new_h = int(frame.shape[0] * sf)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            frame = adjust_brightness(frame, self.brightness.get())
            frame = adjust_contrast(frame, self.contrast.get())
            frame = adjust_saturation(frame, self.saturation.get())
            frame = adjust_hue(frame, self.hue.get())
            if not self.palette_disabled.get():
                new_palette = extract_palette_accurate_kmeans(
                    frame,
                    num_colors=self.palette_size.get(),
                    max_dim=self.kmeans_downsample.get(),
                    kmeans_algorithm=self.kmeans_algorithm.get(),
                    kmeans_inits=self.kmeans_inits.get()
                )
                dynamic_palette = new_palette
            else:
                dynamic_palette = []
                for hexcol in self.selected_palette:
                    r = int(hexcol[1:3], 16)
                    g = int(hexcol[3:5], 16)
                    b = int(hexcol[5:7], 16)
                    dynamic_palette.append((r, g, b))
                if not dynamic_palette:
                    dynamic_palette = [(0, 0, 0), (255, 255, 255)]
            method = self.selected_dither.get()
            dith_func = DITHERING_TYPES.get(method, floyd_steinberg_dither)
            dframe = dith_func(frame.copy(), dynamic_palette, self.dither_strength.get(),
                               preserve=self.preserve_original.get(),
                               preserve_threshold=self.preserve_threshold.get())
            if self.noise_reduction.get():
                dframe = apply_noise_reduction(dframe)
            out_name = os.path.splitext(os.path.basename(input_path))[0] + "_dithered.png"
            out_path = os.path.join(output_dir, out_name)
            cv2.imwrite(out_path, dframe)
            self.update_progress(100)
            self.update_status(f"Status: Completed! Saved as {out_name}")
            messagebox.showinfo("Success", f"Dithered image saved as {out_path}")
        except Exception as e:
            self.update_status("Status: Error occurred.")
            messagebox.showerror("Error", f"An error occurred: {e}")
        finally:
            self.set_ui_state(True)

    def process_video(self, video_path, output_dir):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Cannot open video file {video_path}")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            temp_dir = os.path.join(output_dir, "temp_frames")
            os.makedirs(temp_dir, exist_ok=True)
            interval = self.resample_interval.get()
            self.cached_palette = None
            alpha = 0.3
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                idx += 1
                custom_res = self.custom_resolution.get().strip()
                if custom_res:
                    try:
                        width, height = map(int, custom_res.lower().split('x'))
                        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                    except Exception:
                        sf = self.res_scale.get()
                        new_w = int(frame.shape[1] * sf)
                        new_h = int(frame.shape[0] * sf)
                        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                else:
                    sf = self.res_scale.get()
                    if abs(sf - 1.0) > 1e-3:
                        new_w = int(frame.shape[1] * sf)
                        new_h = int(frame.shape[0] * sf)
                        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                frame = adjust_brightness(frame, self.brightness.get())
                frame = adjust_contrast(frame, self.contrast.get())
                frame = adjust_saturation(frame, self.saturation.get())
                frame = adjust_hue(frame, self.hue.get())
                if not self.palette_disabled.get():
                    if idx == 1 or ((idx - 1) % interval == 0):
                        new_palette = extract_palette_accurate_kmeans(
                            frame,
                            num_colors=self.palette_size.get(),
                            max_dim=self.kmeans_downsample.get(),
                            kmeans_algorithm=self.kmeans_algorithm.get(),
                            kmeans_inits=self.kmeans_inits.get()
                        )
                        if self.cached_palette is None:
                            self.cached_palette = new_palette
                        else:
                            self.cached_palette = blend_palette(self.cached_palette, new_palette, alpha)
                    dynamic_palette = self.cached_palette
                else:
                    dynamic_palette = []
                    for hexcol in self.selected_palette:
                        r = int(hexcol[1:3], 16)
                        g = int(hexcol[3:5], 16)
                        b = int(hexcol[5:7], 16)
                        dynamic_palette.append((r, g, b))
                    if not dynamic_palette:
                        dynamic_palette = [(0, 0, 0), (255, 255, 255)]
                method = self.selected_dither.get()
                dith_func = DITHERING_TYPES.get(method, floyd_steinberg_dither)
                dframe = dith_func(frame.copy(), dynamic_palette, self.dither_strength.get(),
                                   preserve=self.preserve_original.get(),
                                   preserve_threshold=self.preserve_threshold.get())
                if self.noise_reduction.get():
                    dframe = apply_noise_reduction(dframe)
                out_path = os.path.join(temp_dir, f"frame_{idx:06d}.png")
                cv2.imwrite(out_path, dframe)
                self.update_progress((idx / total_frames) * 100)
            cap.release()
            ffmpeg_path = self.get_ffmpeg_path()
            output_name = os.path.splitext(os.path.basename(video_path))[0] + "_dithered.mp4"
            output_path = os.path.join(output_dir, output_name)
            cmd = [
                ffmpeg_path, "-y",
                "-framerate", str(int(fps)),
                "-i", os.path.join(temp_dir, "frame_%06d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                output_path
            ]
            subprocess.run(cmd, check=True)
            for f in glob.glob(os.path.join(temp_dir, "*.png")):
                os.remove(f)
            os.rmdir(temp_dir)
            self.update_progress(100)
            self.update_status(f"Status: Completed! Saved as {output_name}")
            messagebox.showinfo("Success", f"Dithered video saved as {output_name}")
        except subprocess.CalledProcessError:
            self.update_status("Status: FFmpeg processing failed.")
            messagebox.showerror("Error", "An error occurred during FFmpeg processing.")
        except Exception as e:
            self.update_status("Status: Error occurred.")
            messagebox.showerror("Error", f"An error occurred: {e}")
        finally:
            self.set_ui_state(True)

    def set_ui_state(self, enabled=True):
        state = tk.NORMAL if enabled else tk.DISABLED
        widgets_to_configure = []
        for container in [self.leftFrame, self.rightFrame, self.centerFrame]:
            for child in container.winfo_children():
                if isinstance(child, (tk.Button, ttk.Combobox, tk.Entry, tk.Scale, tk.Checkbutton)):
                    widgets_to_configure.append(child)
                elif isinstance(child, (tk.Frame, ttk.Frame)):
                    for gc in child.winfo_children():
                        if isinstance(gc, (tk.Button, ttk.Combobox, tk.Entry, tk.Scale, tk.Checkbutton)):
                            widgets_to_configure.append(gc)
        for widget in widgets_to_configure:
            try:
                widget.config(state=state)
            except tk.TclError:
                pass

    def update_progress(self, value):
        self.progress["value"] = value
        self.master.update_idletasks()

    def update_status(self, msg):
        self.status_label.config(text=msg)
        self.master.update_idletasks()

    def get_ffmpeg_path(self):
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
            return os.path.join(base_path, 'ffmpeg', 'ffmpeg.exe')
        else:
            return 'ffmpeg'

    def show_about(self):
        messagebox.showinfo("About",
                            "Advanced Video Dithering App (Enhanced)\n\nDeveloped by Sir Samuel The Great\n\nFeatures:\n• Multiple dithering algorithms (including Retro Shading and Random Dithering)\n• Advanced image adjustments\n• Customizable palettes with KMeans extraction\n• Live preview with pan & zoom\n• Save/Load/Reset configuration\n• Custom resolution input")

    # ------------------ Pan & Zoom ------------------

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
        if hasattr(event, "delta") and event.delta != 0:
            zoom_change = 1.1 if event.delta > 0 else 0.9
        elif hasattr(event, "num"):
            if event.num == 4:
                zoom_change = 1.1
            elif event.num == 5:
                zoom_change = 0.9
            else:
                zoom_change = 1.0
        else:
            zoom_change = 1.0
        new_zoom = self.zoom_factor * zoom_change
        new_zoom = max(0.1, min(new_zoom, 10.0))
        zoom_change = new_zoom / self.zoom_factor
        self.zoom_factor = new_zoom
        mx, my = event.x, event.y
        self.offset_x -= (mx - self.offset_x) * (zoom_change - 1)
        self.offset_y -= (my - self.offset_y) * (zoom_change - 1)
        self.draw_preview()

    def draw_preview(self):
        if self.preview_pil is None:
            return
        new_w = max(1, int(self.preview_pil.width * self.zoom_factor))
        new_h = max(1, int(self.preview_pil.height * self.zoom_factor))
        resized = self.preview_pil.resize((new_w, new_h), Image.LANCZOS)
        self.preview_image = ImageTk.PhotoImage(resized)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.preview_image)

    # ------------------ Live Preview ------------------

    def on_live_toggle(self):
        if self.live_preview.get():
            self.load_preview()

    def on_live_preview(self, event=None):
        if self.live_preview.get():
            self.load_preview()

    def on_frame_slider_change(self, value):
        frame_index = int(float(value))
        self.current_preview_frame = frame_index
        if self.live_preview.get():
            self.load_preview(frame_index=frame_index)

    # ------------------ Main Preview and Processing ------------------

    def load_preview(self, frame_index=None):
        if frame_index is not None:
            self.current_preview_frame = frame_index
        input_path = self.input_video_path.get()
        if not input_path or not os.path.isfile(input_path):
            self.status_label.config(text="Status: Please select a valid input file.")
            return
        ext = os.path.splitext(input_path)[1].lower()
        IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"]
        VIDEO_EXTS = [".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv"]
        if ext in IMAGE_EXTS:
            frame = cv2.imread(input_path)
            if frame is None:
                self.status_label.config(text="Status: Cannot read image file.")
                return
        elif ext in VIDEO_EXTS:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                self.status_label.config(text=f"Status: Cannot open video file {input_path}")
                return
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_preview_frame)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                self.status_label.config(text="Status: Cannot read frame from video.")
                return
        else:
            self.status_label.config(text=f"Status: Unsupported file extension: {ext}")
            return

        # Custom resolution handling
        custom_res = self.custom_resolution.get().strip()
        if custom_res:
            try:
                width, height = map(int, custom_res.lower().split('x'))
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            except Exception:
                sf = self.res_scale.get()
                new_w = int(frame.shape[1] * sf)
                new_h = int(frame.shape[0] * sf)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            sf = self.res_scale.get()
            if abs(sf - 1.0) > 1e-3:
                new_w = int(frame.shape[1] * sf)
                new_h = int(frame.shape[0] * sf)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        frame = adjust_brightness(frame, self.brightness.get())
        frame = adjust_contrast(frame, self.contrast.get())
        frame = adjust_saturation(frame, self.saturation.get())
        frame = adjust_hue(frame, self.hue.get())
        interval = self.resample_interval.get()
        if not self.palette_disabled.get():
            if self.resample_each_frame.get():
                if self.current_preview_frame == 0 or (self.current_preview_frame % interval == 0):
                    self.cached_preview_palette = extract_palette_accurate_kmeans(
                        frame,
                        num_colors=self.palette_size.get(),
                        max_dim=self.kmeans_downsample.get(),
                        kmeans_algorithm=self.kmeans_algorithm.get(),
                        kmeans_inits=self.kmeans_inits.get()
                    )
            else:
                if (self.current_preview_frame == 0) or not hasattr(self, 'cached_preview_palette'):
                    self.cached_preview_palette = extract_palette_accurate_kmeans(
                        frame,
                        num_colors=self.palette_size.get(),
                        max_dim=self.kmeans_downsample.get(),
                        kmeans_algorithm=self.kmeans_algorithm.get(),
                        kmeans_inits=self.kmeans_inits.get()
                    )
            dynamic_palette = self.cached_preview_palette
        else:
            dynamic_palette = []
            for hexcol in self.selected_palette:
                r = int(hexcol[1:3], 16)
                g = int(hexcol[3:5], 16)
                b = int(hexcol[5:7], 16)
                dynamic_palette.append((r, g, b))
            if not dynamic_palette:
                dynamic_palette = [(0, 0, 0), (255, 255, 255)]
        method = self.selected_dither.get()
        dith_func = DITHERING_TYPES.get(method, floyd_steinberg_dither)
        try:
            dithered = dith_func(frame.copy(), dynamic_palette, self.dither_strength.get(),
                                 preserve=self.preserve_original.get(),
                                 preserve_threshold=self.preserve_threshold.get())
        except Exception as e:
            self.status_label.config(text=f"Error during dithering: {e}")
            return
        if self.noise_reduction.get():
            dithered = apply_noise_reduction(dithered)
        disp_rgb = cv2.cvtColor(dithered, cv2.COLOR_BGR2RGB)
        self.preview_pil = Image.fromarray(disp_rgb)
        if not self.lock_zoom.get():
            self.offset_x = 0
            self.offset_y = 0
            self.zoom_factor = 1.0
        self.draw_preview()
        self.status_label.config(text=f"Status: Preview Loaded (Frame {self.current_preview_frame})")

    def draw_preview(self):
        if self.preview_pil is None:
            return
        new_w = max(1, int(self.preview_pil.width * self.zoom_factor))
        new_h = max(1, int(self.preview_pil.height * self.zoom_factor))
        resized = self.preview_pil.resize((new_w, new_h), Image.LANCZOS)
        self.preview_image = ImageTk.PhotoImage(resized)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.preview_image)

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
        if hasattr(event, "delta") and event.delta != 0:
            zoom_change = 1.1 if event.delta > 0 else 0.9
        elif hasattr(event, "num"):
            if event.num == 4:
                zoom_change = 1.1
            elif event.num == 5:
                zoom_change = 0.9
            else:
                zoom_change = 1.0
        else:
            zoom_change = 1.0
        new_zoom = self.zoom_factor * zoom_change
        new_zoom = max(0.1, min(new_zoom, 10.0))
        zoom_change = new_zoom / self.zoom_factor
        self.zoom_factor = new_zoom
        mx, my = event.x, event.y
        self.offset_x -= (mx - self.offset_x) * (zoom_change - 1)
        self.offset_y -= (my - self.offset_y) * (zoom_change - 1)
        self.draw_preview()

    def start_dithering(self):
        input_video = self.input_video_path.get()
        output_dir = self.output_dir_path.get()
        if not input_video or not os.path.isfile(input_video):
            messagebox.showerror("Error", "Please select a valid input video.")
            return
        if not output_dir or not os.path.isdir(output_dir):
            messagebox.showerror("Error", "Please select a valid output directory.")
            return
        self.set_ui_state(False)
        self.progress['value'] = 0
        self.status_label.config(text="Status: Processing...")
        file_ext = os.path.splitext(input_video)[1].lower()
        if file_ext in [".mp4", ".mov", ".avi", ".mkv"]:
            threading.Thread(target=self.process_video, args=(input_video, output_dir), daemon=True).start()
        else:
            threading.Thread(target=self.process_image, args=(input_video, output_dir), daemon=True).start()

    def process_image(self, input_path, output_dir):
        try:
            frame = cv2.imread(input_path)
            if frame is None:
                raise Exception(f"Cannot read image file {input_path}")
            custom_res = self.custom_resolution.get().strip()
            if custom_res:
                try:
                    width, height = map(int, custom_res.lower().split('x'))
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                except Exception:
                    sf = self.res_scale.get()
                    new_w = int(frame.shape[1] * sf)
                    new_h = int(frame.shape[0] * sf)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                sf = self.res_scale.get()
                if abs(sf - 1.0) > 1e-3:
                    new_w = int(frame.shape[1] * sf)
                    new_h = int(frame.shape[0] * sf)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            frame = adjust_brightness(frame, self.brightness.get())
            frame = adjust_contrast(frame, self.contrast.get())
            frame = adjust_saturation(frame, self.saturation.get())
            frame = adjust_hue(frame, self.hue.get())
            if not self.palette_disabled.get():
                new_palette = extract_palette_accurate_kmeans(
                    frame,
                    num_colors=self.palette_size.get(),
                    max_dim=self.kmeans_downsample.get(),
                    kmeans_algorithm=self.kmeans_algorithm.get(),
                    kmeans_inits=self.kmeans_inits.get()
                )
                dynamic_palette = new_palette
            else:
                dynamic_palette = []
                for hexcol in self.selected_palette:
                    r = int(hexcol[1:3], 16)
                    g = int(hexcol[3:5], 16)
                    b = int(hexcol[5:7], 16)
                    dynamic_palette.append((r, g, b))
                if not dynamic_palette:
                    dynamic_palette = [(0, 0, 0), (255, 255, 255)]
            method = self.selected_dither.get()
            dith_func = DITHERING_TYPES.get(method, floyd_steinberg_dither)
            dframe = dith_func(frame.copy(), dynamic_palette, self.dither_strength.get(),
                               preserve=self.preserve_original.get(),
                               preserve_threshold=self.preserve_threshold.get())
            if self.noise_reduction.get():
                dframe = apply_noise_reduction(dframe)
            out_name = os.path.splitext(os.path.basename(input_path))[0] + "_dithered.png"
            out_path = os.path.join(output_dir, out_name)
            cv2.imwrite(out_path, dframe)
            self.update_progress(100)
            self.update_status(f"Status: Completed! Saved as {out_name}")
            messagebox.showinfo("Success", f"Dithered image saved as {out_path}")
        except Exception as e:
            self.update_status("Status: Error occurred.")
            messagebox.showerror("Error", f"An error occurred: {e}")
        finally:
            self.set_ui_state(True)

    def process_video(self, video_path, output_dir):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Cannot open video file {video_path}")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            temp_dir = os.path.join(output_dir, "temp_frames")
            os.makedirs(temp_dir, exist_ok=True)
            interval = self.resample_interval.get()
            self.cached_palette = None
            alpha = 0.3
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                idx += 1
                custom_res = self.custom_resolution.get().strip()
                if custom_res:
                    try:
                        width, height = map(int, custom_res.lower().split('x'))
                        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                    except Exception:
                        sf = self.res_scale.get()
                        new_w = int(frame.shape[1] * sf)
                        new_h = int(frame.shape[0] * sf)
                        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                else:
                    sf = self.res_scale.get()
                    if abs(sf - 1.0) > 1e-3:
                        new_w = int(frame.shape[1] * sf)
                        new_h = int(frame.shape[0] * sf)
                        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                frame = adjust_brightness(frame, self.brightness.get())
                frame = adjust_contrast(frame, self.contrast.get())
                frame = adjust_saturation(frame, self.saturation.get())
                frame = adjust_hue(frame, self.hue.get())
                if not self.palette_disabled.get():
                    if idx == 1 or ((idx - 1) % interval == 0):
                        new_palette = extract_palette_accurate_kmeans(
                            frame,
                            num_colors=self.palette_size.get(),
                            max_dim=self.kmeans_downsample.get(),
                            kmeans_algorithm=self.kmeans_algorithm.get(),
                            kmeans_inits=self.kmeans_inits.get()
                        )
                        if self.cached_palette is None:
                            self.cached_palette = new_palette
                        else:
                            self.cached_palette = blend_palette(self.cached_palette, new_palette, alpha)
                    dynamic_palette = self.cached_palette
                else:
                    dynamic_palette = []
                    for hexcol in self.selected_palette:
                        r = int(hexcol[1:3], 16)
                        g = int(hexcol[3:5], 16)
                        b = int(hexcol[5:7], 16)
                        dynamic_palette.append((r, g, b))
                    if not dynamic_palette:
                        dynamic_palette = [(0, 0, 0), (255, 255, 255)]
                method = self.selected_dither.get()
                dith_func = DITHERING_TYPES.get(method, floyd_steinberg_dither)
                dframe = dith_func(frame.copy(), dynamic_palette, self.dither_strength.get(),
                                   preserve=self.preserve_original.get(),
                                   preserve_threshold=self.preserve_threshold.get())
                if self.noise_reduction.get():
                    dframe = apply_noise_reduction(dframe)
                out_path = os.path.join(temp_dir, f"frame_{idx:06d}.png")
                cv2.imwrite(out_path, dframe)
                self.update_progress((idx / total_frames) * 100)
            cap.release()
            ffmpeg_path = self.get_ffmpeg_path()
            output_name = os.path.splitext(os.path.basename(video_path))[0] + "_dithered.mp4"
            output_path = os.path.join(output_dir, output_name)
            cmd = [
                ffmpeg_path, "-y",
                "-framerate", str(int(fps)),
                "-i", os.path.join(temp_dir, "frame_%06d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                output_path
            ]
            subprocess.run(cmd, check=True)
            for f in glob.glob(os.path.join(temp_dir, "*.png")):
                os.remove(f)
            os.rmdir(temp_dir)
            self.update_progress(100)
            self.update_status(f"Status: Completed! Saved as {output_name}")
            messagebox.showinfo("Success", f"Dithered video saved as {output_name}")
        except subprocess.CalledProcessError:
            self.update_status("Status: FFmpeg processing failed.")
            messagebox.showerror("Error", "An error occurred during FFmpeg processing.")
        except Exception as e:
            self.update_status("Status: Error occurred.")
            messagebox.showerror("Error", f"An error occurred: {e}")
        finally:
            self.set_ui_state(True)

    def set_ui_state(self, enabled=True):
        state = tk.NORMAL if enabled else tk.DISABLED
        widgets_to_configure = []
        for container in [self.leftFrame, self.rightFrame, self.centerFrame]:
            for child in container.winfo_children():
                if isinstance(child, (tk.Button, ttk.Combobox, tk.Entry, tk.Scale, tk.Checkbutton)):
                    widgets_to_configure.append(child)
                elif isinstance(child, (tk.Frame, ttk.Frame)):
                    for gc in child.winfo_children():
                        if isinstance(gc, (tk.Button, ttk.Combobox, tk.Entry, tk.Scale, tk.Checkbutton)):
                            widgets_to_configure.append(gc)
        for widget in widgets_to_configure:
            try:
                widget.config(state=state)
            except tk.TclError:
                pass

    def update_progress(self, value):
        self.progress["value"] = value
        self.master.update_idletasks()

    def update_status(self, msg):
        self.status_label.config(text=msg)
        self.master.update_idletasks()

    def get_ffmpeg_path(self):
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
            return os.path.join(base_path, 'ffmpeg', 'ffmpeg.exe')
        else:
            return 'ffmpeg'

    def show_about(self):
        messagebox.showinfo("About",
                            "Advanced Video Dithering App (Enhanced)\n\nDeveloped by Sir Samuel The Great\n\nFeatures:\n• Multiple dithering algorithms (including Retro Shading and Random Dithering)\n• Advanced image adjustments\n• Customizable palettes with KMeans extraction\n• Live preview with pan & zoom\n• Save/Load/Reset configuration\n• Custom resolution input")

    # ------------------ Pan & Zoom ------------------

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
        if hasattr(event, "delta") and event.delta != 0:
            zoom_change = 1.1 if event.delta > 0 else 0.9
        elif hasattr(event, "num"):
            if event.num == 4:
                zoom_change = 1.1
            elif event.num == 5:
                zoom_change = 0.9
            else:
                zoom_change = 1.0
        else:
            zoom_change = 1.0
        new_zoom = self.zoom_factor * zoom_change
        new_zoom = max(0.1, min(new_zoom, 10.0))
        zoom_change = new_zoom / self.zoom_factor
        self.zoom_factor = new_zoom
        mx, my = event.x, event.y
        self.offset_x -= (mx - self.offset_x) * (zoom_change - 1)
        self.offset_y -= (my - self.offset_y) * (zoom_change - 1)
        self.draw_preview()

    def draw_preview(self):
        if self.preview_pil is None:
            return
        new_w = max(1, int(self.preview_pil.width * self.zoom_factor))
        new_h = max(1, int(self.preview_pil.height * self.zoom_factor))
        resized = self.preview_pil.resize((new_w, new_h), Image.LANCZOS)
        self.preview_image = ImageTk.PhotoImage(resized)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.preview_image)

    # ------------------ Live Preview ------------------

    def on_live_toggle(self):
        if self.live_preview.get():
            self.load_preview()

    def on_live_preview(self, event=None):
        if self.live_preview.get():
            self.load_preview()

    def on_frame_slider_change(self, value):
        frame_index = int(float(value))
        self.current_preview_frame = frame_index
        if self.live_preview.get():
            self.load_preview(frame_index=frame_index)

    # ------------------ Main Preview and Processing ------------------

    def load_preview(self, frame_index=None):
        if frame_index is not None:
            self.current_preview_frame = frame_index
        input_path = self.input_video_path.get()
        if not input_path or not os.path.isfile(input_path):
            self.status_label.config(text="Status: Please select a valid input file.")
            return
        ext = os.path.splitext(input_path)[1].lower()
        IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"]
        VIDEO_EXTS = [".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv"]
        if ext in IMAGE_EXTS:
            frame = cv2.imread(input_path)
            if frame is None:
                self.status_label.config(text="Status: Cannot read image file.")
                return
        elif ext in VIDEO_EXTS:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                self.status_label.config(text=f"Status: Cannot open video file {input_path}")
                return
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_preview_frame)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                self.status_label.config(text="Status: Cannot read frame from video.")
                return
        else:
            self.status_label.config(text=f"Status: Unsupported file extension: {ext}")
            return

        custom_res = self.custom_resolution.get().strip()
        if custom_res:
            try:
                width, height = map(int, custom_res.lower().split('x'))
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            except Exception:
                sf = self.res_scale.get()
                new_w = int(frame.shape[1] * sf)
                new_h = int(frame.shape[0] * sf)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            sf = self.res_scale.get()
            if abs(sf - 1.0) > 1e-3:
                new_w = int(frame.shape[1] * sf)
                new_h = int(frame.shape[0] * sf)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        frame = adjust_brightness(frame, self.brightness.get())
        frame = adjust_contrast(frame, self.contrast.get())
        frame = adjust_saturation(frame, self.saturation.get())
        frame = adjust_hue(frame, self.hue.get())
        interval = self.resample_interval.get()
        if not self.palette_disabled.get():
            if self.resample_each_frame.get():
                if self.current_preview_frame == 0 or (self.current_preview_frame % interval == 0):
                    self.cached_preview_palette = extract_palette_accurate_kmeans(
                        frame,
                        num_colors=self.palette_size.get(),
                        max_dim=self.kmeans_downsample.get(),
                        kmeans_algorithm=self.kmeans_algorithm.get(),
                        kmeans_inits=self.kmeans_inits.get()
                    )
            else:
                if (self.current_preview_frame == 0) or not hasattr(self, 'cached_preview_palette'):
                    self.cached_preview_palette = extract_palette_accurate_kmeans(
                        frame,
                        num_colors=self.palette_size.get(),
                        max_dim=self.kmeans_downsample.get(),
                        kmeans_algorithm=self.kmeans_algorithm.get(),
                        kmeans_inits=self.kmeans_inits.get()
                    )
            dynamic_palette = self.cached_preview_palette
        else:
            dynamic_palette = []
            for hexcol in self.selected_palette:
                r = int(hexcol[1:3], 16)
                g = int(hexcol[3:5], 16)
                b = int(hexcol[5:7], 16)
                dynamic_palette.append((r, g, b))
            if not dynamic_palette:
                dynamic_palette = [(0, 0, 0), (255, 255, 255)]
        method = self.selected_dither.get()
        dith_func = DITHERING_TYPES.get(method, floyd_steinberg_dither)
        try:
            dithered = dith_func(frame.copy(), dynamic_palette, self.dither_strength.get(),
                                 preserve=self.preserve_original.get(),
                                 preserve_threshold=self.preserve_threshold.get())
        except Exception as e:
            self.status_label.config(text=f"Error during dithering: {e}")
            return
        if self.noise_reduction.get():
            dithered = apply_noise_reduction(dithered)
        disp_rgb = cv2.cvtColor(dithered, cv2.COLOR_BGR2RGB)
        self.preview_pil = Image.fromarray(disp_rgb)
        if not self.lock_zoom.get():
            self.offset_x = 0
            self.offset_y = 0
            self.zoom_factor = 1.0
        self.draw_preview()
        self.status_label.config(text=f"Status: Preview Loaded (Frame {self.current_preview_frame})")

    def draw_preview(self):
        if self.preview_pil is None:
            return
        new_w = max(1, int(self.preview_pil.width * self.zoom_factor))
        new_h = max(1, int(self.preview_pil.height * self.zoom_factor))
        resized = self.preview_pil.resize((new_w, new_h), Image.LANCZOS)
        self.preview_image = ImageTk.PhotoImage(resized)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.preview_image)

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
        if hasattr(event, "delta") and event.delta != 0:
            zoom_change = 1.1 if event.delta > 0 else 0.9
        elif hasattr(event, "num"):
            if event.num == 4:
                zoom_change = 1.1
            elif event.num == 5:
                zoom_change = 0.9
            else:
                zoom_change = 1.0
        else:
            zoom_change = 1.0
        new_zoom = self.zoom_factor * zoom_change
        new_zoom = max(0.1, min(new_zoom, 10.0))
        zoom_change = new_zoom / self.zoom_factor
        self.zoom_factor = new_zoom
        mx, my = event.x, event.y
        self.offset_x -= (mx - self.offset_x) * (zoom_change - 1)
        self.offset_y -= (my - self.offset_y) * (zoom_change - 1)
        self.draw_preview()

    def start_dithering(self):
        input_video = self.input_video_path.get()
        output_dir = self.output_dir_path.get()
        if not input_video or not os.path.isfile(input_video):
            messagebox.showerror("Error", "Please select a valid input video.")
            return
        if not output_dir or not os.path.isdir(output_dir):
            messagebox.showerror("Error", "Please select a valid output directory.")
            return
        self.set_ui_state(False)
        self.progress['value'] = 0
        self.status_label.config(text="Status: Processing...")
        file_ext = os.path.splitext(input_video)[1].lower()
        if file_ext in [".mp4", ".mov", ".avi", ".mkv"]:
            threading.Thread(target=self.process_video, args=(input_video, output_dir), daemon=True).start()
        else:
            threading.Thread(target=self.process_image, args=(input_video, output_dir), daemon=True).start()

    def process_image(self, input_path, output_dir):
        try:
            frame = cv2.imread(input_path)
            if frame is None:
                raise Exception(f"Cannot read image file {input_path}")
            custom_res = self.custom_resolution.get().strip()
            if custom_res:
                try:
                    width, height = map(int, custom_res.lower().split('x'))
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                except Exception:
                    sf = self.res_scale.get()
                    new_w = int(frame.shape[1] * sf)
                    new_h = int(frame.shape[0] * sf)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                sf = self.res_scale.get()
                if abs(sf - 1.0) > 1e-3:
                    new_w = int(frame.shape[1] * sf)
                    new_h = int(frame.shape[0] * sf)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            frame = adjust_brightness(frame, self.brightness.get())
            frame = adjust_contrast(frame, self.contrast.get())
            frame = adjust_saturation(frame, self.saturation.get())
            frame = adjust_hue(frame, self.hue.get())
            if not self.palette_disabled.get():
                new_palette = extract_palette_accurate_kmeans(
                    frame,
                    num_colors=self.palette_size.get(),
                    max_dim=self.kmeans_downsample.get(),
                    kmeans_algorithm=self.kmeans_algorithm.get(),
                    kmeans_inits=self.kmeans_inits.get()
                )
                dynamic_palette = new_palette
            else:
                dynamic_palette = []
                for hexcol in self.selected_palette:
                    r = int(hexcol[1:3], 16)
                    g = int(hexcol[3:5], 16)
                    b = int(hexcol[5:7], 16)
                    dynamic_palette.append((r, g, b))
                if not dynamic_palette:
                    dynamic_palette = [(0, 0, 0), (255, 255, 255)]
            method = self.selected_dither.get()
            dith_func = DITHERING_TYPES.get(method, floyd_steinberg_dither)
            dframe = dith_func(frame.copy(), dynamic_palette, self.dither_strength.get(),
                               preserve=self.preserve_original.get(),
                               preserve_threshold=self.preserve_threshold.get())
            if self.noise_reduction.get():
                dframe = apply_noise_reduction(dframe)
            out_name = os.path.splitext(os.path.basename(input_path))[0] + "_dithered.png"
            out_path = os.path.join(output_dir, out_name)
            cv2.imwrite(out_path, dframe)
            self.update_progress(100)
            self.update_status(f"Status: Completed! Saved as {out_name}")
            messagebox.showinfo("Success", f"Dithered image saved as {out_path}")
        except Exception as e:
            self.update_status("Status: Error occurred.")
            messagebox.showerror("Error", f"An error occurred: {e}")
        finally:
            self.set_ui_state(True)

    def process_video(self, video_path, output_dir):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Cannot open video file {video_path}")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            temp_dir = os.path.join(output_dir, "temp_frames")
            os.makedirs(temp_dir, exist_ok=True)
            interval = self.resample_interval.get()
            self.cached_palette = None
            alpha = 0.3
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                idx += 1
                custom_res = self.custom_resolution.get().strip()
                if custom_res:
                    try:
                        width, height = map(int, custom_res.lower().split('x'))
                        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                    except Exception:
                        sf = self.res_scale.get()
                        new_w = int(frame.shape[1] * sf)
                        new_h = int(frame.shape[0] * sf)
                        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                else:
                    sf = self.res_scale.get()
                    if abs(sf - 1.0) > 1e-3:
                        new_w = int(frame.shape[1] * sf)
                        new_h = int(frame.shape[0] * sf)
                        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                frame = adjust_brightness(frame, self.brightness.get())
                frame = adjust_contrast(frame, self.contrast.get())
                frame = adjust_saturation(frame, self.saturation.get())
                frame = adjust_hue(frame, self.hue.get())
                if not self.palette_disabled.get():
                    if idx == 1 or ((idx - 1) % interval == 0):
                        new_palette = extract_palette_accurate_kmeans(
                            frame,
                            num_colors=self.palette_size.get(),
                            max_dim=self.kmeans_downsample.get(),
                            kmeans_algorithm=self.kmeans_algorithm.get(),
                            kmeans_inits=self.kmeans_inits.get()
                        )
                        if self.cached_palette is None:
                            self.cached_palette = new_palette
                        else:
                            self.cached_palette = blend_palette(self.cached_palette, new_palette, alpha)
                    dynamic_palette = self.cached_palette
                else:
                    dynamic_palette = []
                    for hexcol in self.selected_palette:
                        r = int(hexcol[1:3], 16)
                        g = int(hexcol[3:5], 16)
                        b = int(hexcol[5:7], 16)
                        dynamic_palette.append((r, g, b))
                    if not dynamic_palette:
                        dynamic_palette = [(0, 0, 0), (255, 255, 255)]
                method = self.selected_dither.get()
                dith_func = DITHERING_TYPES.get(method, floyd_steinberg_dither)
                dframe = dith_func(frame.copy(), dynamic_palette, self.dither_strength.get(),
                                   preserve=self.preserve_original.get(),
                                   preserve_threshold=self.preserve_threshold.get())
                if self.noise_reduction.get():
                    dframe = apply_noise_reduction(dframe)
                out_path = os.path.join(temp_dir, f"frame_{idx:06d}.png")
                cv2.imwrite(out_path, dframe)
                self.update_progress((idx / total_frames) * 100)
            cap.release()
            ffmpeg_path = self.get_ffmpeg_path()
            output_name = os.path.splitext(os.path.basename(video_path))[0] + "_dithered.mp4"
            output_path = os.path.join(output_dir, output_name)
            cmd = [
                ffmpeg_path, "-y",
                "-framerate", str(int(fps)),
                "-i", os.path.join(temp_dir, "frame_%06d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                output_path
            ]
            subprocess.run(cmd, check=True)
            for f in glob.glob(os.path.join(temp_dir, "*.png")):
                os.remove(f)
            os.rmdir(temp_dir)
            self.update_progress(100)
            self.update_status(f"Status: Completed! Saved as {output_name}")
            messagebox.showinfo("Success", f"Dithered video saved as {output_name}")
        except subprocess.CalledProcessError:
            self.update_status("Status: FFmpeg processing failed.")
            messagebox.showerror("Error", "An error occurred during FFmpeg processing.")
        except Exception as e:
            self.update_status("Status: Error occurred.")
            messagebox.showerror("Error", f"An error occurred: {e}")
        finally:
            self.set_ui_state(True)

    def set_ui_state(self, enabled=True):
        state = tk.NORMAL if enabled else tk.DISABLED
        widgets_to_configure = []
        for container in [self.leftFrame, self.rightFrame, self.centerFrame]:
            for child in container.winfo_children():
                if isinstance(child, (tk.Button, ttk.Combobox, tk.Entry, tk.Scale, tk.Checkbutton)):
                    widgets_to_configure.append(child)
                elif isinstance(child, (tk.Frame, ttk.Frame)):
                    for gc in child.winfo_children():
                        if isinstance(gc, (tk.Button, ttk.Combobox, tk.Entry, tk.Scale, tk.Checkbutton)):
                            widgets_to_configure.append(gc)
        for widget in widgets_to_configure:
            try:
                widget.config(state=state)
            except tk.TclError:
                pass

    def update_progress(self, value):
        self.progress["value"] = value
        self.master.update_idletasks()

    def update_status(self, msg):
        self.status_label.config(text=msg)
        self.master.update_idletasks()

    def get_ffmpeg_path(self):
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
            return os.path.join(base_path, 'ffmpeg', 'ffmpeg.exe')
        else:
            return 'ffmpeg'

    def show_about(self):
        messagebox.showinfo("About",
                            "Advanced Video Dithering App (Enhanced)\n\nDeveloped by Sir Samuel The Great\n\nFeatures:\n• Multiple dithering algorithms (including Retro Shading and Random Dithering)\n• Advanced image adjustments\n• Customizable palettes with KMeans extraction\n• Live preview with pan & zoom\n• Save/Load/Reset configuration\n• Custom resolution input")

    # ------------------ Pan & Zoom ------------------

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
        if hasattr(event, "delta") and event.delta != 0:
            zoom_change = 1.1 if event.delta > 0 else 0.9
        elif hasattr(event, "num"):
            if event.num == 4:
                zoom_change = 1.1
            elif event.num == 5:
                zoom_change = 0.9
            else:
                zoom_change = 1.0
        else:
            zoom_change = 1.0
        new_zoom = self.zoom_factor * zoom_change
        new_zoom = max(0.1, min(new_zoom, 10.0))
        zoom_change = new_zoom / self.zoom_factor
        self.zoom_factor = new_zoom
        mx, my = event.x, event.y
        self.offset_x -= (mx - self.offset_x) * (zoom_change - 1)
        self.offset_y -= (my - self.offset_y) * (zoom_change - 1)
        self.draw_preview()

    def draw_preview(self):
        if self.preview_pil is None:
            return
        new_w = max(1, int(self.preview_pil.width * self.zoom_factor))
        new_h = max(1, int(self.preview_pil.height * self.zoom_factor))
        resized = self.preview_pil.resize((new_w, new_h), Image.LANCZOS)
        self.preview_image = ImageTk.PhotoImage(resized)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.preview_image)

    # ------------------ Live Preview ------------------

    def on_live_toggle(self):
        if self.live_preview.get():
            self.load_preview()

    def on_live_preview(self, event=None):
        if self.live_preview.get():
            self.load_preview()

    def on_frame_slider_change(self, value):
        frame_index = int(float(value))
        self.current_preview_frame = frame_index
        if self.live_preview.get():
            self.load_preview(frame_index=frame_index)

    # ------------------ Main Preview and Processing ------------------

    def load_preview(self, frame_index=None):
        if frame_index is not None:
            self.current_preview_frame = frame_index
        input_path = self.input_video_path.get()
        if not input_path or not os.path.isfile(input_path):
            self.status_label.config(text="Status: Please select a valid input file.")
            return
        ext = os.path.splitext(input_path)[1].lower()
        IMAGE_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp"]
        VIDEO_EXTS = [".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv"]
        if ext in IMAGE_EXTS:
            frame = cv2.imread(input_path)
            if frame is None:
                self.status_label.config(text="Status: Cannot read image file.")
                return
        elif ext in VIDEO_EXTS:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                self.status_label.config(text=f"Status: Cannot open video file {input_path}")
                return
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_preview_frame)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                self.status_label.config(text="Status: Cannot read frame from video.")
                return
        else:
            self.status_label.config(text=f"Status: Unsupported file extension: {ext}")
            return

        custom_res = self.custom_resolution.get().strip()
        if custom_res:
            try:
                width, height = map(int, custom_res.lower().split('x'))
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            except Exception:
                sf = self.res_scale.get()
                new_w = int(frame.shape[1] * sf)
                new_h = int(frame.shape[0] * sf)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            sf = self.res_scale.get()
            if abs(sf - 1.0) > 1e-3:
                new_w = int(frame.shape[1] * sf)
                new_h = int(frame.shape[0] * sf)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        frame = adjust_brightness(frame, self.brightness.get())
        frame = adjust_contrast(frame, self.contrast.get())
        frame = adjust_saturation(frame, self.saturation.get())
        frame = adjust_hue(frame, self.hue.get())
        interval = self.resample_interval.get()
        if not self.palette_disabled.get():
            # Convert frame to RGB for palette extraction
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.palette_method.get() == "Accurate (scikit-learn)":
                palette_func = extract_palette_accurate_kmeans
                palette_params = {
                    "num_colors": self.palette_size.get(),
                    "max_dim": self.kmeans_downsample.get(),
                    "kmeans_algorithm": self.kmeans_algorithm.get(),
                    "kmeans_inits": self.kmeans_inits.get(),
                    "kmeans_max_iter": self.kmeans_max_iter.get(),  # new setting
                    "kmeans_tol": self.kmeans_tol.get()             # new setting
                }
                seed = self.palette_seed.get()
                if seed != 0:
                    palette_params["random_state"] = seed
            else:  # "Fast (OpenCV)"
                palette_func = extract_palette_fast_opencv  # Make sure this is implemented in your backend
                palette_params = {
                    "num_colors": self.palette_size.get(),
                    "max_dim": self.kmeans_downsample.get()
                }
            if self.resample_each_frame.get():
                if self.current_preview_frame == 0 or (self.current_preview_frame % interval == 0):
                    rgb_palette = palette_func(frame_rgb, **palette_params)
                    # Convert extracted RGB palette back to BGR for dithering
                    self.cached_preview_palette = [(b, g, r) for (r, g, b) in rgb_palette]
            else:
                if self.current_preview_frame == 0 or not hasattr(self, 'cached_preview_palette'):
                    rgb_palette = palette_func(frame_rgb, **palette_params)
                    self.cached_preview_palette = [(b, g, r) for (r, g, b) in rgb_palette]
            dynamic_palette = self.cached_preview_palette
        else:
            # Use custom palette if dynamic palette is disabled.
            dynamic_palette = []
            for hexcol in self.selected_palette:
                r = int(hexcol[1:3], 16)
                g = int(hexcol[3:5], 16)
                b = int(hexcol[5:7], 16)
                dynamic_palette.append((r, g, b))
            if not dynamic_palette:
                dynamic_palette = [(0, 0, 0), (255, 255, 255)]
        method = self.selected_dither.get()
        dith_func = DITHERING_TYPES.get(method, floyd_steinberg_dither)
        try:
            dithered = dith_func(frame.copy(), dynamic_palette, self.dither_strength.get(),
                                 preserve=self.preserve_original.get(),
                                 preserve_threshold=self.preserve_threshold.get())
        except Exception as e:
            self.status_label.config(text=f"Error during dithering: {e}")
            return
        if self.noise_reduction.get():
            dithered = apply_noise_reduction(dithered)
        disp_rgb = cv2.cvtColor(dithered, cv2.COLOR_BGR2RGB)
        self.preview_pil = Image.fromarray(disp_rgb)
        if not self.lock_zoom.get():
            self.offset_x = 0
            self.offset_y = 0
            self.zoom_factor = 1.0
        self.draw_preview()
        self.status_label.config(text=f"Status: Preview Loaded (Frame {self.current_preview_frame})")

    def draw_preview(self):
        if self.preview_pil is None:
            return
        new_w = max(1, int(self.preview_pil.width * self.zoom_factor))
        new_h = max(1, int(self.preview_pil.height * self.zoom_factor))
        resized = self.preview_pil.resize((new_w, new_h), Image.LANCZOS)
        self.preview_image = ImageTk.PhotoImage(resized)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.preview_image)

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
        if hasattr(event, "delta") and event.delta != 0:
            zoom_change = 1.1 if event.delta > 0 else 0.9
        elif hasattr(event, "num"):
            if event.num == 4:
                zoom_change = 1.1
            elif event.num == 5:
                zoom_change = 0.9
            else:
                zoom_change = 1.0
        else:
            zoom_change = 1.0
        new_zoom = self.zoom_factor * zoom_change
        new_zoom = max(0.1, min(new_zoom, 10.0))
        zoom_change = new_zoom / self.zoom_factor
        self.zoom_factor = new_zoom
        mx, my = event.x, event.y
        self.offset_x -= (mx - self.offset_x) * (zoom_change - 1)
        self.offset_y -= (my - self.offset_y) * (zoom_change - 1)
        self.draw_preview()

    def start_dithering(self):
        input_video = self.input_video_path.get()
        output_dir = self.output_dir_path.get()
        if not input_video or not os.path.isfile(input_video):
            messagebox.showerror("Error", "Please select a valid input video.")
            return
        if not output_dir or not os.path.isdir(output_dir):
            messagebox.showerror("Error", "Please select a valid output directory.")
            return
        self.set_ui_state(False)
        self.progress['value'] = 0
        self.status_label.config(text="Status: Processing...")
        file_ext = os.path.splitext(input_video)[1].lower()
        if file_ext in [".mp4", ".mov", ".avi", ".mkv"]:
            threading.Thread(target=self.process_video, args=(input_video, output_dir), daemon=True).start()
        else:
            threading.Thread(target=self.process_image, args=(input_video, output_dir), daemon=True).start()

    def process_image(self, input_path, output_dir):
        try:
            frame = cv2.imread(input_path)
            if frame is None:
                raise Exception(f"Cannot read image file {input_path}")
            custom_res = self.custom_resolution.get().strip()
            if custom_res:
                try:
                    width, height = map(int, custom_res.lower().split('x'))
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                except Exception:
                    sf = self.res_scale.get()
                    new_w = int(frame.shape[1] * sf)
                    new_h = int(frame.shape[0] * sf)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                sf = self.res_scale.get()
                if abs(sf - 1.0) > 1e-3:
                    new_w = int(frame.shape[1] * sf)
                    new_h = int(frame.shape[0] * sf)
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            frame = adjust_brightness(frame, self.brightness.get())
            frame = adjust_contrast(frame, self.contrast.get())
            frame = adjust_saturation(frame, self.saturation.get())
            frame = adjust_hue(frame, self.hue.get())
            if not self.palette_disabled.get():
                new_palette = extract_palette_accurate_kmeans(
                    frame,
                    num_colors=self.palette_size.get(),
                    max_dim=self.kmeans_downsample.get(),
                    kmeans_algorithm=self.kmeans_algorithm.get(),
                    kmeans_inits=self.kmeans_inits.get()
                )
                dynamic_palette = new_palette
            else:
                dynamic_palette = []
                for hexcol in self.selected_palette:
                    r = int(hexcol[1:3], 16)
                    g = int(hexcol[3:5], 16)
                    b = int(hexcol[5:7], 16)
                    dynamic_palette.append((r, g, b))
                if not dynamic_palette:
                    dynamic_palette = [(0, 0, 0), (255, 255, 255)]
            method = self.selected_dither.get()
            dith_func = DITHERING_TYPES.get(method, floyd_steinberg_dither)
            dframe = dith_func(frame.copy(), dynamic_palette, self.dither_strength.get(),
                               preserve=self.preserve_original.get(),
                               preserve_threshold=self.preserve_threshold.get())
            if self.noise_reduction.get():
                dframe = apply_noise_reduction(dframe)
            out_name = os.path.splitext(os.path.basename(input_path))[0] + "_dithered.png"
            out_path = os.path.join(output_dir, out_name)
            cv2.imwrite(out_path, dframe)
            self.update_progress(100)
            self.update_status(f"Status: Completed! Saved as {out_name}")
            messagebox.showinfo("Success", f"Dithered image saved as {out_path}")
        except Exception as e:
            self.update_status("Status: Error occurred.")
            messagebox.showerror("Error", f"An error occurred: {e}")
        finally:
            self.set_ui_state(True)

    def process_video(self, video_path, output_dir):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception(f"Cannot open video file {video_path}")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            temp_dir = os.path.join(output_dir, "temp_frames")
            os.makedirs(temp_dir, exist_ok=True)
            interval = self.resample_interval.get()
            self.cached_palette = None
            alpha = 0.3
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                idx += 1
                custom_res = self.custom_resolution.get().strip()
                if custom_res:
                    try:
                        width, height = map(int, custom_res.lower().split('x'))
                        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                    except Exception:
                        sf = self.res_scale.get()
                        new_w = int(frame.shape[1] * sf)
                        new_h = int(frame.shape[0] * sf)
                        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                else:
                    sf = self.res_scale.get()
                    if abs(sf - 1.0) > 1e-3:
                        new_w = int(frame.shape[1] * sf)
                        new_h = int(frame.shape[0] * sf)
                        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                frame = adjust_brightness(frame, self.brightness.get())
                frame = adjust_contrast(frame, self.contrast.get())
                frame = adjust_saturation(frame, self.saturation.get())
                frame = adjust_hue(frame, self.hue.get())
                if not self.palette_disabled.get():
                    if idx == 1 or ((idx - 1) % interval == 0):
                        new_palette = extract_palette_accurate_kmeans(
                            frame,
                            num_colors=self.palette_size.get(),
                            max_dim=self.kmeans_downsample.get(),
                            kmeans_algorithm=self.kmeans_algorithm.get(),
                            kmeans_inits=self.kmeans_inits.get()
                        )
                        if self.cached_palette is None:
                            self.cached_palette = new_palette
                        else:
                            self.cached_palette = blend_palette(self.cached_palette, new_palette, alpha)
                    dynamic_palette = self.cached_palette
                else:
                    dynamic_palette = []
                    for hexcol in self.selected_palette:
                        r = int(hexcol[1:3], 16)
                        g = int(hexcol[3:5], 16)
                        b = int(hexcol[5:7], 16)
                        dynamic_palette.append((r, g, b))
                    if not dynamic_palette:
                        dynamic_palette = [(0, 0, 0), (255, 255, 255)]
                method = self.selected_dither.get()
                dith_func = DITHERING_TYPES.get(method, floyd_steinberg_dither)
                dframe = dith_func(frame.copy(), dynamic_palette, self.dither_strength.get(),
                                   preserve=self.preserve_original.get(),
                                   preserve_threshold=self.preserve_threshold.get())
                if self.noise_reduction.get():
                    dframe = apply_noise_reduction(dframe)
                out_path = os.path.join(temp_dir, f"frame_{idx:06d}.png")
                cv2.imwrite(out_path, dframe)
                self.update_progress((idx / total_frames) * 100)
            cap.release()
            ffmpeg_path = self.get_ffmpeg_path()
            output_name = os.path.splitext(os.path.basename(video_path))[0] + "_dithered.mp4"
            output_path = os.path.join(output_dir, output_name)
            cmd = [
                ffmpeg_path, "-y",
                "-framerate", str(int(fps)),
                "-i", os.path.join(temp_dir, "frame_%06d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                output_path
            ]
            subprocess.run(cmd, check=True)
            for f in glob.glob(os.path.join(temp_dir, "*.png")):
                os.remove(f)
            os.rmdir(temp_dir)
            self.update_progress(100)
            self.update_status(f"Status: Completed! Saved as {output_name}")
            messagebox.showinfo("Success", f"Dithered video saved as {output_name}")
        except subprocess.CalledProcessError:
            self.update_status("Status: FFmpeg processing failed.")
            messagebox.showerror("Error", "An error occurred during FFmpeg processing.")
        except Exception as e:
            self.update_status("Status: Error occurred.")
            messagebox.showerror("Error", f"An error occurred: {e}")
        finally:
            self.set_ui_state(True)

    def set_ui_state(self, enabled=True):
        state = tk.NORMAL if enabled else tk.DISABLED
        widgets = []
        for container in [self.leftFrame, self.rightFrame, self.centerFrame]:
            for child in container.winfo_children():
                if isinstance(child, (tk.Button, ttk.Combobox, tk.Entry, tk.Scale, tk.Checkbutton)):
                    widgets.append(child)
                elif isinstance(child, (tk.Frame, ttk.Frame)):
                    for gc in child.winfo_children():
                        if isinstance(gc, (tk.Button, ttk.Combobox, tk.Entry, tk.Scale, tk.Checkbutton)):
                            widgets.append(gc)
        for widget in widgets:
            try:
                widget.config(state=state)
            except tk.TclError:
                pass

    def update_progress(self, value):
        self.progress["value"] = value
        self.master.update_idletasks()

    def update_status(self, msg):
        self.status_label.config(text=msg)
        self.master.update_idletasks()

    def get_ffmpeg_path(self):
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
            return os.path.join(base_path, 'ffmpeg', 'ffmpeg.exe')
        else:
            return 'ffmpeg'

def main():
    root = tk.Tk()
    app = DitherApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
