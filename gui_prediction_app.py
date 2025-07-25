__version__ = "2.5.0"
__app_name__ = "Vegetation Prediction App"
__author__ = "Kahlil Amin"
__email__ = "kaamin@rivco.org"
__build_date__ = "2025-06-30"

import os
import sys
import time
import threading
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import webbrowser

import numpy as np
import tensorflow as tf

import rasterio

from generate_prediction import (
    generate_prediction,
    get_tiles,
    PRE_TRAINED_MODELS,
)

# If running as a PyInstaller EXE, include GDAL_PATH, PROJ_LIB environment variables
# and redirect stdout/stderr to log files.
if getattr(sys, "frozen", False):
    exe_dir = Path(sys._MEIPASS)
    gdal_data_path = exe_dir / "gdal_data"
    os.environ["GDAL_DATA"] = str(gdal_data_path)
    os.environ["PROJ_LIB"] = str(exe_dir / "rasterio" / "proj_data")

    # Redirect stdout and stderr to log files for PyInstaller console=False mode
    logs_dir = Path.cwd() / "logs"
    logs_dir.mkdir(exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    stdout_path = logs_dir / f"app_stdout_{timestamp_str}.log"
    stderr_path = logs_dir / f"app_stderr_{timestamp_str}.log"

    sys.stdout = stdout_path.open("w")
    sys.stderr = stderr_path.open("w")


class PredictionApp:
    def __init__(self, master):
        self.master = master
        master.title(f"Vegetation Prediction App v{__version__}")

        self.pre_trained_models = PRE_TRAINED_MODELS

        self.prediction_thread = None

        self.input_file = tk.StringVar()
        self.output_file = tk.StringVar()
        self.batch_size = tk.IntVar(value=4)
        self.reclassify_values = tk.BooleanVar(value=True)  # New checkbox variable

        # Input File
        tk.Label(master, text="Input Raster File:").grid(row=0, column=0, sticky="e")
        tk.Entry(master, textvariable=self.input_file, width=50).grid(row=0, column=1)
        tk.Button(master, text="Browse", command=self.browse_input).grid(
            row=0, column=2
        )

        # Output File
        tk.Label(master, text="Output Prediction File:").grid(
            row=1, column=0, sticky="e"
        )
        tk.Entry(master, textvariable=self.output_file, width=50).grid(row=1, column=1)
        tk.Button(master, text="Browse", command=self.browse_output).grid(
            row=1, column=2
        )

        # Create a frame to hold the horizontal inputs
        input_frame = tk.Frame(master)
        input_frame.grid(row=2, column=0, columnspan=4, pady=5, sticky="ew")

        # Configure grid columns to expand equally
        input_frame.columnconfigure(0, weight=1)
        input_frame.columnconfigure(1, weight=1)
        input_frame.columnconfigure(2, weight=1)

        # Batch Size section
        batch_frame = tk.Frame(input_frame)
        batch_frame.grid(row=0, column=0, sticky="ew")
        tk.Label(batch_frame, text="Batch Size:").pack(side="left")
        tk.Spinbox(
            batch_frame, from_=1, to=16, textvariable=self.batch_size, width=5
        ).pack(side="left")

        # Model Selection Dropdown section
        model_frame = tk.Frame(input_frame)
        model_frame.grid(row=0, column=1, sticky="ew")

        self.model_selection = tk.StringVar()
        self.model_selection.set("Average (Top 3 Models)")

        model_options = ["Average (Top 3 Models)"] + [
            model.trial_name for model in self.pre_trained_models
        ]

        self.model_dropdown = ttk.Combobox(
            model_frame,
            textvariable=self.model_selection,
            values=model_options,
            state="readonly",
            width=25,
        )
        self.model_dropdown.pack(fill="x", padx=5)

        # Reclassify Values Checkbox section
        reclass_frame = tk.Frame(input_frame)
        reclass_frame.grid(row=0, column=2, sticky="ew")
        tk.Checkbutton(
            reclass_frame, text="Reclassify Values", variable=self.reclassify_values
        ).pack(side="left")

        # Progress Bar (moved to row 4)
        self.progress = ttk.Progressbar(
            master, orient="horizontal", length=400, mode="determinate"
        )
        self.progress.grid(row=4, column=0, columnspan=3, pady=10)

        # Button Frame for centering
        button_frame = tk.Frame(master)
        button_frame.grid(row=3, column=0, columnspan=3, pady=10)

        # Run Prediction Button inside frame
        tk.Button(
            button_frame, text="Run Prediction", command=self.run_prediction
        ).pack(side="left", padx=10)

        # About Button inside frame
        tk.Button(button_frame, text="About", command=self.show_about).pack(
            side="left", padx=10
        )

        # Status Label (moved to row 5)
        self.status_label = tk.Label(master, text="Ready")
        self.status_label.grid(row=5, column=0, columnspan=3)

        # Close window
        master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        if (
            hasattr(self, "prediction_thread")
            and self.prediction_thread is not None
            and self.prediction_thread.is_alive()
        ):
            if messagebox.askyesno(
                "Exit", "Prediction is still running. Do you really want to quit?"
            ):
                self.master.destroy()
        else:
            self.master.destroy()

    def browse_input(self):
        filename = filedialog.askopenfilename(filetypes=[("TIFF files", "*.tif")])
        self.input_file.set(filename)

    def browse_output(self):
        filename = filedialog.asksaveasfilename(
            defaultextension=".tif", filetypes=[("TIFF files", "*.tif")]
        )
        self.output_file.set(filename)

    def update_status(self, text):
        self.status_label.config(text=text)
        self.master.update_idletasks()

    def show_about(self):
        about_win = tk.Toplevel(self.master)
        about_win.title("About")
        about_win.resizable(False, False)

        # App info text
        info = (
            f"{__app_name__} v{__version__}\n"
            f"Created by {__author__}\n"
            f"Email: {__email__}\n\n"
            f"Python: {sys.version.split()[0]}\n"
            f"TensorFlow: {tf.__version__}\n"
            f"Numpy: {np.__version__}\n"
            f"Rasterio: {rasterio.__version__}\n"
        )

        tk.Label(about_win, text=info, justify="left").pack(padx=10, pady=10)

        # Email link
        def open_email(event):
            webbrowser.open(f"mailto:{__email__}")

        email_link = tk.Label(about_win, text=__email__, fg="blue", cursor="hand2")
        email_link.pack()
        email_link.bind("<Button-1>", open_email)

        def open_github(event):
            webbrowser.open("https://github.com/kahlilamin/sar_segmentation_2025_app")

        # GitHub link
        github_link = tk.Label(
            about_win, text="GitHub Repository", fg="blue", cursor="hand2"
        )
        github_link.pack()
        github_link.bind("<Button-1>", open_github)

        # Close button
        tk.Button(about_win, text="Close", command=about_win.destroy).pack(pady=10)

    def validate_input_raster(self, filepath):

        try:
            with rasterio.open(filepath) as src:
                # Check for 4 bands
                if src.count < 4:
                    messagebox.showerror(
                        "Invalid Input", "Input raster must have at least 4 bands."
                    )
                    return False

                # Check data type
                if src.dtypes[0] not in ["uint16", "uint8"]:
                    messagebox.showerror(
                        "Invalid Input",
                        f"Unsupported data type: {src.dtypes[0]}. Only 8-bit or 16-bit integer rasters are allowed.",
                    )
                    return False

                # Check CRS is projected and in California State Plane Zone 6
                epsg = src.crs.to_epsg() if src.crs else None
                if epsg not in [2230, 2875]:
                    messagebox.showerror(
                        "Invalid Input",
                        "Input raster must be in California State Plane Zone 6 (EPSG:2230 or EPSG:2875).",
                    )
                    return False

                # Check resolution (pixel size)
                xres, yres = src.res
                expected_res = 0.5  # feet

                # Allow slight tolerance
                tolerance = expected_res * 0.01  # 1% tolerance

                if not (
                    abs(xres - expected_res) < tolerance
                    and abs(yres - expected_res) < tolerance
                ):
                    messagebox.showerror(
                        "Invalid Input",
                        f"Input raster resolution must be {expected_res} feet.\n"
                        f"Found: ({xres:.4f}, {yres:.4f})",
                    )
                    return False

            return True

        except Exception as e:
            messagebox.showerror("Invalid Input", f"Failed to read input raster:\n{e}")
            return False

    def estimate_valid_windows(self, src, tile_size=256, stride=128):

        profile = src.profile.copy()

        valid_windows = []

        for window, _ in get_tiles(src, tile_size, tile_size, stride):
            tile_img = src.read(1, window=window)

            if np.all(tile_img == profile["nodata"]):
                continue

            valid_windows.append(window)

        return valid_windows

    def run_prediction(self):
        # Validate input and output paths
        if not self.input_file.get():
            messagebox.showerror("Error", "Please select an input raster file.")
            return

        if not self.output_file.get():
            messagebox.showerror("Error", "Please specify an output prediction file.")
            return

        def task():
            try:
                start_time = time.time()
                # Show loading models status
                self.master.after(
                    0,
                    lambda: [
                        self.update_status("Loading models..."),
                        self.progress.config(mode="indeterminate"),
                        self.progress.start(),
                    ],
                )

                # Load models
                selected = self.model_selection.get()
                if selected == "Average (Top 3 Models)":
                    models_to_use = self.pre_trained_models
                else:
                    models_to_use = [
                        m for m in self.pre_trained_models if m.trial_name == selected
                    ]
                    if not models_to_use:
                        messagebox.showerror(
                            "Error", f"Selected model '{selected}' not found."
                        )
                        return

                for model in models_to_use:
                    model.load()

                # Start reading file
                self.master.after(
                    0,
                    lambda: [
                        self.update_status("Reading file..."),
                        self.progress.config(mode="indeterminate"),
                        self.progress.start(),
                    ],
                )

                sar_img_tif = Path(self.input_file.get())
                prediction_tif = Path(self.output_file.get())

                # Input validation check
                if not self.validate_input_raster(sar_img_tif):
                    return  # Abort if invalid

                with rasterio.open(sar_img_tif) as src:

                    valid_windows = self.estimate_valid_windows(src)
                    total_tiles = len(valid_windows)

                    # Switch to determinate mode
                    self.master.after(
                        0,
                        lambda: [
                            self.progress.stop(),
                            self.progress.config(
                                mode="determinate", maximum=total_tiles, value=0
                            ),
                            self.update_status("Processing..."),
                        ],
                    )

                    processed_tiles = 0

                    def progress_callback():
                        nonlocal processed_tiles
                        processed_tiles += 1
                        self.master.after(
                            0,
                            lambda: [
                                self.progress.config(value=processed_tiles),
                                self.update_status(
                                    f"Processing... {processed_tiles}/{total_tiles} ({processed_tiles/total_tiles:.2%})"
                                ),
                            ],
                        )

                    profile = src.profile.copy()

                    generate_prediction(
                        src,
                        profile,
                        prediction_tif,
                        models_to_use,
                        valid_windows,
                        tile_size=256,
                        stride=128,
                        batch_size=self.batch_size.get(),
                        progress_callback=progress_callback,
                        reclassify_values=self.reclassify_values.get(),
                    )

                elapsed_time = time.time() - start_time
                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

                self.master.after(
                    0,
                    lambda: [
                        self.update_status("Completed"),
                        self.progress.config(value=total_tiles),
                        messagebox.showinfo(
                            "Success",
                            f"Prediction completed successfully! {total_tiles} tiles completed in: {elapsed_str}",
                        ),
                    ],
                )

            except Exception as e:
                self.master.after(
                    0,
                    lambda e=e: [
                        self.progress.stop(),
                        self.progress.config(mode="determinate", value=0),
                        self.update_status("Error"),
                        messagebox.showerror("Error", str(e)),
                    ],
                )

        self.prediction_thread = threading.Thread(target=task)
        self.prediction_thread.start()


if __name__ == "__main__":
    root = tk.Tk()
    app = PredictionApp(root)
    root.mainloop()
