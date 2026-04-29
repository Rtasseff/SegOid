"""
Main Tkinter application for SegOid GUI.

Provides a graphical interface for running spheroid segmentation inference
on folders of images using the bundled or custom ONNX model.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Optional

from src.gui.widgets import FolderPicker, FilePicker, CheckboxOption, ScrolledText
from src.gui.logging_handler import GUILogHandler
from src.gui.jobs import InferenceJob


def get_bundled_model_path() -> Optional[Path]:
    """
    Get path to bundled ONNX model.

    Handles both development (source) and PyInstaller bundle contexts.

    Returns:
        Path to bundled model, or None if not found
    """
    # When running as PyInstaller bundle
    if getattr(sys, 'frozen', False):
        base_path = Path(sys._MEIPASS)
    else:
        # Development: look relative to project root
        base_path = Path(__file__).parent.parent.parent

    model_path = base_path / "assets" / "segoid_model.onnx"

    if model_path.exists():
        return model_path

    return None


class SegOidApp:
    """
    Main application window for SegOid inference GUI.
    """

    def __init__(self, root: tk.Tk):
        """
        Initialize the application.

        Args:
            root: Root Tkinter window
        """
        self.root = root
        self.root.title("SegOid Inference")
        self.root.geometry("800x600")
        self.root.minsize(700, 500)

        # State
        self.current_job: Optional[InferenceJob] = None
        self.use_custom_model = tk.BooleanVar(value=False)

        # Build UI
        self._create_widgets()

        # Start log polling
        self.log_handler.start_polling()

        # Check for bundled model
        self._check_bundled_model()

    def _create_widgets(self):
        """Create all UI widgets."""
        # Main container
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title = ttk.Label(
            main_frame,
            text="SegOid Spheroid Segmentation",
            font=("TkDefaultFont", 14, "bold"),
        )
        title.pack(pady=(0, 10))

        # Model selection section
        model_frame = ttk.LabelFrame(main_frame, text="Model", padding=10)
        model_frame.pack(fill=tk.X, pady=5)

        self.model_status = ttk.Label(model_frame, text="Bundled model: checking...")
        self.model_status.pack(anchor=tk.W)

        custom_row = ttk.Frame(model_frame)
        custom_row.pack(fill=tk.X, pady=(5, 0))

        self.custom_model_check = ttk.Checkbutton(
            custom_row,
            text="Use custom model",
            variable=self.use_custom_model,
            command=self._on_custom_model_toggle,
        )
        self.custom_model_check.pack(anchor=tk.W)

        self.model_picker = FilePicker(
            model_frame,
            label="Custom model:",
            filetypes=[("ONNX models", "*.onnx"), ("All files", "*.*")],
        )
        self.model_picker.pack(fill=tk.X, pady=(5, 0))
        self.model_picker.entry.config(state=tk.DISABLED)
        self.model_picker.browse_btn.config(state=tk.DISABLED)

        # Input/Output section
        io_frame = ttk.LabelFrame(main_frame, text="Input / Output", padding=10)
        io_frame.pack(fill=tk.X, pady=5)

        self.input_picker = FolderPicker(
            io_frame,
            label="Input image folder:",
        )
        self.input_picker.pack(fill=tk.X, pady=2)

        self.output_picker = FolderPicker(
            io_frame,
            label="Output folder:",
        )
        self.output_picker.pack(fill=tk.X, pady=2)

        # Inference Settings section
        inference_frame = ttk.LabelFrame(main_frame, text="Inference Settings", padding=10)
        inference_frame.pack(fill=tk.X, pady=5)

        # Pixel size for multi-resolution inference
        pixel_size_row = ttk.Frame(inference_frame)
        pixel_size_row.pack(fill=tk.X)

        ttk.Label(pixel_size_row, text="Pixel size (µm/pixel):").pack(side=tk.LEFT)
        self.inference_pixel_size_entry = ttk.Entry(pixel_size_row, width=10)
        self.inference_pixel_size_entry.insert(0, "2.76")
        self.inference_pixel_size_entry.pack(side=tk.LEFT, padx=5)

        ttk.Label(
            inference_frame,
            text="Leave at 2.76 for standard resolution images. Enter 1.1 for high-mag images.",
            font=("TkDefaultFont", 8),
            foreground="gray",
        ).pack(anchor=tk.W, pady=(2, 0))

        # Histogram matching option
        histogram_row = ttk.Frame(inference_frame)
        histogram_row.pack(fill=tk.X, pady=(10, 0))

        self.use_histogram_match = tk.BooleanVar(value=False)
        self.histogram_check = ttk.Checkbutton(
            histogram_row,
            text="Apply histogram matching",
            variable=self.use_histogram_match,
            command=self._on_histogram_toggle,
        )
        self.histogram_check.pack(anchor=tk.W)

        self.histogram_picker = FilePicker(
            inference_frame,
            label="Reference image:",
            filetypes=[("TIFF images", "*.tif *.tiff"), ("All images", "*.png *.jpg *.tif *.tiff"), ("All files", "*.*")],
        )
        self.histogram_picker.pack(fill=tk.X, pady=(5, 0))
        self.histogram_picker.entry.config(state=tk.DISABLED)
        self.histogram_picker.browse_btn.config(state=tk.DISABLED)

        ttk.Label(
            inference_frame,
            text="Use when images have different intensity than training data (e.g., different microscopy settings).",
            font=("TkDefaultFont", 8),
            foreground="gray",
        ).pack(anchor=tk.W, pady=(2, 0))

        # Options section
        options_frame = ttk.LabelFrame(main_frame, text="Options", padding=10)
        options_frame.pack(fill=tk.X, pady=5)

        self.compute_metrics = CheckboxOption(
            options_frame,
            label="Compute morphology metrics (area, circularity, etc.)",
            initial_value=True,
        )
        self.compute_metrics.pack(anchor=tk.W)

        self.export_video = CheckboxOption(
            options_frame,
            label="Export prediction review video",
            initial_value=False,
        )
        self.export_video.pack(anchor=tk.W)

        # Filename parsing section (collapsible)
        self.parse_filenames = CheckboxOption(
            options_frame,
            label="Parse filename into fields (split by '_')",
            initial_value=False,
            on_change=self._on_parse_toggle,
        )
        self.parse_filenames.pack(anchor=tk.W, pady=(10, 0))

        self.schema_frame = ttk.Frame(options_frame)
        self.schema_frame.pack(fill=tk.X, pady=5)

        self.schema_label = ttk.Label(
            self.schema_frame,
            text="Field labels (comma-separated):",
        )
        self.schema_entry = ttk.Entry(self.schema_frame, width=40)
        self.schema_entry.insert(0, "plate, well, time")

        # Initially hidden
        self.schema_label.pack_forget()
        self.schema_entry.pack_forget()

        # Fluorescence markers section (collapsible)
        self.detect_markers = CheckboxOption(
            options_frame,
            label="Detect fluorescence markers (companion images <base>_<marker>.tif)",
            initial_value=False,
            on_change=self._on_marker_toggle,
        )
        self.detect_markers.pack(anchor=tk.W, pady=(10, 0))

        self.markers_frame = ttk.Frame(options_frame)
        self.markers_frame.pack(fill=tk.X, pady=5)

        self.markers_label = ttk.Label(
            self.markers_frame,
            text="Marker names (comma-separated):",
        )
        self.markers_entry = ttk.Entry(self.markers_frame, width=40)
        self.markers_entry.insert(0, "green, red")

        # Initially hidden
        self.markers_label.pack_forget()
        self.markers_entry.pack_forget()

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)

        self.run_btn = ttk.Button(
            button_frame,
            text="Run Inference",
            command=self._run_inference,
        )
        self.run_btn.pack(side=tk.LEFT, padx=5)

        self.cancel_btn = ttk.Button(
            button_frame,
            text="Cancel",
            command=self._cancel_job,
            state=tk.DISABLED,
        )
        self.cancel_btn.pack(side=tk.LEFT, padx=5)

        self.open_output_btn = ttk.Button(
            button_frame,
            text="Open Output Folder",
            command=self._open_output_folder,
            state=tk.DISABLED,
        )
        self.open_output_btn.pack(side=tk.RIGHT, padx=5)

        # Log area
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.log_text = ScrolledText(log_frame, height=12)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Initialize log handler
        self.log_handler = GUILogHandler(
            text_widget=self.log_text.get_text_widget(),
            root=self.root,
        )

    def _check_bundled_model(self):
        """Check for bundled model and update status."""
        model_path = get_bundled_model_path()

        if model_path:
            self.bundled_model_path = model_path
            self.model_status.config(text=f"Bundled model: Available")
            self.log_handler.log("Found bundled ONNX model", "INFO")
        else:
            self.bundled_model_path = None
            self.model_status.config(text="Bundled model: Not found (select custom)")
            self.use_custom_model.set(True)
            self._on_custom_model_toggle()
            self.log_handler.log("No bundled model found - please select a custom model", "WARNING")

    def _on_custom_model_toggle(self):
        """Handle custom model checkbox toggle."""
        if self.use_custom_model.get():
            self.model_picker.entry.config(state=tk.NORMAL)
            self.model_picker.browse_btn.config(state=tk.NORMAL)
        else:
            self.model_picker.entry.config(state=tk.DISABLED)
            self.model_picker.browse_btn.config(state=tk.DISABLED)

    def _on_histogram_toggle(self):
        """Handle histogram matching checkbox toggle."""
        if self.use_histogram_match.get():
            self.histogram_picker.entry.config(state=tk.NORMAL)
            self.histogram_picker.browse_btn.config(state=tk.NORMAL)
        else:
            self.histogram_picker.entry.config(state=tk.DISABLED)
            self.histogram_picker.browse_btn.config(state=tk.DISABLED)

    def _on_parse_toggle(self, enabled: bool):
        """Handle filename parsing toggle."""
        if enabled:
            self.schema_label.pack(side=tk.LEFT, padx=(20, 5))
            self.schema_entry.pack(side=tk.LEFT)
        else:
            self.schema_label.pack_forget()
            self.schema_entry.pack_forget()

    def _on_marker_toggle(self, enabled: bool):
        """Handle fluorescence marker detection toggle."""
        if enabled:
            self.markers_label.pack(side=tk.LEFT, padx=(20, 5))
            self.markers_entry.pack(side=tk.LEFT)
        else:
            self.markers_label.pack_forget()
            self.markers_entry.pack_forget()

    def _get_model_path(self) -> Optional[Path]:
        """Get the model path to use."""
        if self.use_custom_model.get():
            custom_path = self.model_picker.get()
            if custom_path and Path(custom_path).exists():
                return Path(custom_path)
            return None
        else:
            return self.bundled_model_path

    def _validate_inputs(self) -> bool:
        """Validate all inputs before running."""
        # Check model
        model_path = self._get_model_path()
        if model_path is None:
            messagebox.showerror("Error", "Please select a valid ONNX model file.")
            return False

        # Check input folder
        input_folder = self.input_picker.get()
        if not input_folder or not Path(input_folder).is_dir():
            messagebox.showerror("Error", "Please select a valid input image folder.")
            return False

        # Check output folder (can be created)
        output_folder = self.output_picker.get()
        if not output_folder:
            messagebox.showerror("Error", "Please specify an output folder.")
            return False

        return True

    def _run_inference(self):
        """Start the inference job."""
        if not self._validate_inputs():
            return

        # Disable run button, enable cancel
        self.run_btn.config(state=tk.DISABLED)
        self.cancel_btn.config(state=tk.NORMAL)
        self.open_output_btn.config(state=tk.DISABLED)

        # Clear log
        self.log_handler.clear()

        # Get filename schema if enabled
        filename_schema = None
        if self.parse_filenames.get():
            labels = [l.strip() for l in self.schema_entry.get().split(",")]
            filename_schema = {"labels": labels, "delimiter": "_"}

        # Get pixel size (used for both inference rescaling and metric conversion)
        pixel_size = None
        try:
            pixel_size_str = self.inference_pixel_size_entry.get().strip()
            if pixel_size_str:
                pixel_size = float(pixel_size_str)
        except ValueError:
            messagebox.showerror("Error", "Invalid pixel size. Please enter a number.")
            self.run_btn.config(state=tk.NORMAL)
            self.cancel_btn.config(state=tk.DISABLED)
            return

        # Get histogram reference path if enabled
        histogram_reference_path = None
        if self.use_histogram_match.get():
            ref_path = self.histogram_picker.get()
            if ref_path and Path(ref_path).exists():
                histogram_reference_path = ref_path
            else:
                messagebox.showerror("Error", "Please select a valid reference image for histogram matching.")
                self.run_btn.config(state=tk.NORMAL)
                self.cancel_btn.config(state=tk.DISABLED)
                return

        # Get fluorescence markers if enabled
        markers = None
        if self.detect_markers.get():
            markers = [m.strip() for m in self.markers_entry.get().split(",") if m.strip()]

        # Create and start job
        self.current_job = InferenceJob(
            input_folder=self.input_picker.get(),
            output_folder=self.output_picker.get(),
            model_path=str(self._get_model_path()),
            log_callback=self.log_handler.get_callback(),
            on_complete=self._on_job_complete,
            compute_metrics=self.compute_metrics.get(),
            export_video=self.export_video.get(),
            filename_schema=filename_schema,
            pixel_size=pixel_size,
            histogram_reference_path=histogram_reference_path,
            markers=markers,
        )
        self.current_job.start()

    def _cancel_job(self):
        """Cancel the running job."""
        if self.current_job and self.current_job.is_running():
            self.current_job.cancel()

    def _on_job_complete(self, success: bool, message: str):
        """Handle job completion (called from background thread)."""
        # Schedule UI update on main thread
        self.root.after(0, lambda: self._update_ui_after_job(success))

    def _update_ui_after_job(self, success: bool):
        """Update UI after job completes."""
        self.run_btn.config(state=tk.NORMAL)
        self.cancel_btn.config(state=tk.DISABLED)

        if success:
            self.open_output_btn.config(state=tk.NORMAL)

        self.current_job = None

    def _open_output_folder(self):
        """Open the output folder in file explorer."""
        output_folder = self.output_picker.get()
        if output_folder and Path(output_folder).exists():
            # Cross-platform open folder
            if sys.platform == "win32":
                os.startfile(output_folder)
            elif sys.platform == "darwin":
                os.system(f'open "{output_folder}"')
            else:
                os.system(f'xdg-open "{output_folder}"')


def main():
    """Main entry point for the GUI application."""
    root = tk.Tk()

    # Set icon if available
    try:
        if getattr(sys, 'frozen', False):
            icon_path = Path(sys._MEIPASS) / "assets" / "segoid.ico"
        else:
            icon_path = Path(__file__).parent.parent.parent / "assets" / "segoid.ico"

        if icon_path.exists():
            root.iconbitmap(str(icon_path))
    except Exception:
        pass  # Icon is optional

    app = SegOidApp(root)

    # Handle window close
    def on_closing():
        if app.current_job and app.current_job.is_running():
            if messagebox.askokcancel("Quit", "A job is running. Cancel and quit?"):
                app.current_job.cancel()
                root.destroy()
        else:
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)

    root.mainloop()


if __name__ == "__main__":
    main()
