"""
Pipeline orchestration for the SegOid GUI.

Provides job functions that can be run in background threads to perform
inference, metrics extraction, and video export without blocking the UI.
"""

import threading
from pathlib import Path
from typing import Callable, Optional, List

import pandas as pd


# Type alias for log callback
LogCallback = Callable[[str, str], None]


class InferenceJob:
    """
    Background job for running inference on a folder of images.

    Orchestrates the full pipeline:
    1. Build manifest from input folder
    2. Load ONNX model
    3. Run inference on all images
    4. Compute metrics
    5. Optionally export review video
    """

    def __init__(
        self,
        input_folder: str,
        output_folder: str,
        model_path: str,
        log_callback: LogCallback,
        on_complete: Optional[Callable[[bool, str], None]] = None,
        compute_metrics: bool = True,
        export_video: bool = False,
        filename_schema: Optional[dict] = None,
        pixel_size: Optional[float] = None,
        histogram_reference_path: Optional[str] = None,
    ):
        """
        Initialize inference job.

        Args:
            input_folder: Path to folder containing input images
            output_folder: Path to folder for outputs
            model_path: Path to ONNX model file
            log_callback: Callback for logging (message, level) -> None
            on_complete: Callback when job completes (success, message) -> None
            compute_metrics: Whether to compute morphology metrics
            export_video: Whether to export review video
            filename_schema: Optional dict with 'labels' for filename parsing
            pixel_size: Optional pixel size in µm/pixel for scaling metrics
            histogram_reference_path: Optional path to reference image for histogram matching
        """
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.model_path = Path(model_path)
        self.log_callback = log_callback
        self.on_complete = on_complete
        self.compute_metrics = compute_metrics
        self.export_video = export_video
        self.filename_schema = filename_schema
        self.pixel_size = pixel_size
        self.histogram_reference_path = Path(histogram_reference_path) if histogram_reference_path else None

        self._thread: Optional[threading.Thread] = None
        self._cancelled = False

    def start(self):
        """Start the job in a background thread."""
        self._cancelled = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def cancel(self):
        """Request job cancellation."""
        self._cancelled = True
        self.log_callback("Cancellation requested...", "WARNING")

    def is_running(self) -> bool:
        """Check if job is currently running."""
        return self._thread is not None and self._thread.is_alive()

    def _log(self, msg: str, level: str = "INFO"):
        """Log a message."""
        self.log_callback(msg, level)

    def _run(self):
        """Execute the job (runs in background thread)."""
        try:
            self._log("Starting inference job...")
            self._log(f"Input folder: {self.input_folder}")
            self._log(f"Output folder: {self.output_folder}")
            self._log(f"Model: {self.model_path}")

            # Step 1: Build manifest
            self._log("Scanning input folder for images...")
            image_paths = self._scan_images()

            if len(image_paths) == 0:
                self._complete(False, "No .tif/.tiff files found in input folder")
                return

            self._log(f"Found {len(image_paths)} images")

            if self._cancelled:
                self._complete(False, "Job cancelled")
                return

            # Step 2: Load model
            self._log("Loading ONNX model...")
            model = self._load_model()

            if model is None:
                self._complete(False, "Failed to load ONNX model")
                return

            if self._cancelled:
                self._complete(False, "Job cancelled")
                return

            # Step 2b: Load histogram reference if provided
            histogram_reference = None
            if self.histogram_reference_path:
                self._log(f"Loading histogram reference: {self.histogram_reference_path.name}")
                histogram_reference = self._load_histogram_reference()
                if histogram_reference is None:
                    self._complete(False, "Failed to load histogram reference image")
                    return

            if self._cancelled:
                self._complete(False, "Job cancelled")
                return

            # Step 3: Run inference
            self._log("Running inference...")
            successful, failed, output_paths = self._run_inference(image_paths, model, histogram_reference)

            self._log(f"Inference complete: {successful} successful, {failed} failed")

            if self._cancelled:
                self._complete(False, "Job cancelled")
                return

            # Step 4: Compute metrics
            if self.compute_metrics and successful > 0:
                self._log("Computing morphology metrics...")
                self._compute_metrics(output_paths)

            if self._cancelled:
                self._complete(False, "Job cancelled")
                return

            # Step 5: Export video
            if self.export_video and successful > 0:
                self._log("Exporting review video...")
                self._export_video()

            # Complete
            self._complete(True, f"Completed: {successful} images processed")

        except Exception as e:
            self._log(f"Job failed with error: {e}", "ERROR")
            self._complete(False, f"Error: {e}")

    def _scan_images(self) -> List[Path]:
        """Scan input folder for image files."""
        from src.data.manifest import scan_folder_for_images

        return scan_folder_for_images(self.input_folder)

    def _load_model(self):
        """Load the ONNX model."""
        try:
            from src.inference.predict import load_onnx_model

            return load_onnx_model(self.model_path, self.log_callback)
        except Exception as e:
            self._log(f"Failed to load model: {e}", "ERROR")
            return None

    def _load_histogram_reference(self):
        """Load the histogram reference image."""
        try:
            from src.inference.predict import load_histogram_reference

            return load_histogram_reference(self.histogram_reference_path)
        except Exception as e:
            self._log(f"Failed to load histogram reference: {e}", "ERROR")
            return None

    def _run_inference(self, image_paths: List[Path], model, histogram_reference=None) -> tuple:
        """Run inference on all images."""
        from src.inference.predict import run_inference_batch

        return run_inference_batch(
            image_paths=image_paths,
            model=model,
            output_dir=self.output_folder,
            pixel_size=self.pixel_size,  # For multi-resolution inference
            histogram_reference=histogram_reference,
            log_callback=self.log_callback,
        )

    def _compute_metrics(self, mask_paths: List[Path]):
        """Compute morphology metrics from predicted masks."""
        try:
            from src.analysis.quantify import extract_objects, compute_object_properties
            from src.inference.predict import imread  # Use PIL-based imread for Windows compatibility

            if self.pixel_size:
                self._log(f"Scaling metrics to µm (pixel size: {self.pixel_size} µm/px)")

            all_objects = []

            for mask_path in mask_paths:
                if self._cancelled:
                    return

                basename = mask_path.stem.replace("_pred_mask", "")

                try:
                    mask = imread(mask_path)
                    labeled, n_objects = extract_objects(mask, min_area=100)

                    if n_objects > 0:
                        props = compute_object_properties(labeled, pixel_size=self.pixel_size)
                        props["image"] = basename

                        # Add filename metadata if schema provided
                        if self.filename_schema and self.filename_schema.get("labels"):
                            self._add_filename_metadata(props, basename)

                        all_objects.append(props)
                except Exception as e:
                    self._log(f"Metrics failed for {basename}: {e}", "WARNING")

            if all_objects:
                combined = pd.concat(all_objects, ignore_index=True)

                # Reorder columns
                cols = ["image"] + [c for c in combined.columns if c != "image"]
                combined = combined[cols]

                # Save metrics
                metrics_path = self.output_folder / "metrics.csv"
                combined.to_csv(metrics_path, index=False)
                self._log(f"Saved metrics to {metrics_path}")

                # Save summary statistics
                summary_path = self.output_folder / "summary_statistics.csv"
                numeric_cols = combined.select_dtypes(include="number").columns
                summary = combined[numeric_cols].describe()
                summary.to_csv(summary_path)
                self._log(f"Saved summary to {summary_path}")

        except Exception as e:
            self._log(f"Metrics computation failed: {e}", "ERROR")

    def _add_filename_metadata(self, df: pd.DataFrame, basename: str):
        """Add metadata columns from filename parsing."""
        from src.data.filename_schema import FilenameSchema, parse_filename

        labels = self.filename_schema.get("labels", [])
        delimiter = self.filename_schema.get("delimiter", "_")

        schema = FilenameSchema(
            delimiter=delimiter,
            field_names=labels,
            expected_fields=len(labels),
        )

        metadata = parse_filename(basename, schema)

        for key, value in metadata.items():
            df[key] = value

    def _export_video(self):
        """Export review video."""
        try:
            from src.visualization.review import export_review_video

            video_path = self.output_folder / "prediction_review.mp4"

            export_review_video(
                image_dir=str(self.input_folder),
                pred_mask_dir=str(self.output_folder),
                output_video=str(video_path),
                frame_duration=2.0,
            )

            self._log(f"Saved review video to {video_path}")

        except Exception as e:
            self._log(f"Video export failed: {e}", "WARNING")

    def _complete(self, success: bool, message: str):
        """Handle job completion."""
        level = "SUCCESS" if success else "ERROR"
        self._log(message, level)

        if self.on_complete:
            self.on_complete(success, message)


def run_inference_job(
    input_folder: str,
    output_folder: str,
    model_path: str,
    log_callback: LogCallback,
    on_complete: Optional[Callable[[bool, str], None]] = None,
    compute_metrics: bool = True,
    export_video: bool = False,
    filename_schema: Optional[dict] = None,
    pixel_size: Optional[float] = None,
) -> InferenceJob:
    """
    Create and start an inference job.

    Args:
        input_folder: Path to input images
        output_folder: Path for outputs
        model_path: Path to ONNX model
        log_callback: Logging callback
        on_complete: Completion callback
        compute_metrics: Whether to compute metrics
        export_video: Whether to export video
        filename_schema: Optional filename parsing schema
        pixel_size: Optional pixel size in µm/pixel for scaling metrics

    Returns:
        Started InferenceJob instance
    """
    job = InferenceJob(
        input_folder=input_folder,
        output_folder=output_folder,
        model_path=model_path,
        log_callback=log_callback,
        on_complete=on_complete,
        compute_metrics=compute_metrics,
        export_video=export_video,
        filename_schema=filename_schema,
        pixel_size=pixel_size,
    )
    job.start()
    return job
