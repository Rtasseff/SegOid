"""
Thread-safe logging handler for GUI integration.

Provides a queue-based logging system that allows background worker threads
to send log messages to the GUI thread safely.
"""

import logging
import queue
import tkinter as tk
from datetime import datetime
from enum import Enum
from typing import Callable, Optional


class LogLevel(Enum):
    """Log level enumeration with display properties."""

    INFO = ("INFO", "#000000")  # Black
    WARNING = ("WARNING", "#CC6600")  # Orange
    ERROR = ("ERROR", "#CC0000")  # Red
    SUCCESS = ("SUCCESS", "#006600")  # Green

    def __init__(self, label: str, color: str):
        self.label = label
        self.color = color


class QueueHandler(logging.Handler):
    """
    Logging handler that puts log records into a queue.

    This allows log messages from background threads to be safely
    passed to the GUI thread.
    """

    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord):
        """Put log record into queue."""
        try:
            msg = self.format(record)
            level = record.levelname
            self.log_queue.put((msg, level))
        except Exception:
            self.handleError(record)


class GUILogHandler:
    """
    Manages logging from background threads to a Tkinter Text widget.

    Uses a queue to safely pass messages between threads and polls
    the queue periodically to update the GUI.
    """

    def __init__(
        self,
        text_widget: tk.Text,
        root: tk.Tk,
        poll_interval: int = 100,
    ):
        """
        Initialize the GUI log handler.

        Args:
            text_widget: Tkinter Text widget to display logs
            root: Root Tkinter window (for scheduling)
            poll_interval: Milliseconds between queue polls
        """
        self.text_widget = text_widget
        self.root = root
        self.poll_interval = poll_interval
        self.log_queue: queue.Queue = queue.Queue()
        self._polling = False

        # Configure text widget tags for colored output
        self._configure_tags()

    def _configure_tags(self):
        """Configure text tags for different log levels."""
        for level in LogLevel:
            self.text_widget.tag_configure(
                level.name,
                foreground=level.color,
            )
        # Also configure standard logging level names
        self.text_widget.tag_configure("DEBUG", foreground="#666666")
        self.text_widget.tag_configure("CRITICAL", foreground="#CC0000")

    def get_callback(self) -> Callable[[str, str], None]:
        """
        Get a callback function for use with inference functions.

        Returns:
            Callback function (message, level) -> None
        """
        def callback(message: str, level: str = "INFO"):
            self.log_queue.put((message, level))

        return callback

    def log(self, message: str, level: str = "INFO"):
        """
        Add a log message to the queue.

        Args:
            message: Log message text
            level: Log level (INFO, WARNING, ERROR, SUCCESS)
        """
        self.log_queue.put((message, level))

    def start_polling(self):
        """Start polling the queue for new messages."""
        self._polling = True
        self._poll_queue()

    def stop_polling(self):
        """Stop polling the queue."""
        self._polling = False

    def _poll_queue(self):
        """Check queue for new messages and update text widget."""
        if not self._polling:
            return

        # Process all available messages
        while True:
            try:
                message, level = self.log_queue.get_nowait()
                self._append_message(message, level)
            except queue.Empty:
                break

        # Schedule next poll
        self.root.after(self.poll_interval, self._poll_queue)

    def _append_message(self, message: str, level: str):
        """Append a message to the text widget."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] {message}\n"

        # Enable editing, insert, then disable
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.insert(tk.END, formatted, level.upper())
        self.text_widget.see(tk.END)  # Auto-scroll
        self.text_widget.config(state=tk.DISABLED)

    def clear(self):
        """Clear all log messages."""
        self.text_widget.config(state=tk.NORMAL)
        self.text_widget.delete(1.0, tk.END)
        self.text_widget.config(state=tk.DISABLED)


def create_log_callback(log_handler: GUILogHandler) -> Callable[[str, str], None]:
    """
    Create a logging callback for use with inference functions.

    Args:
        log_handler: GUILogHandler instance

    Returns:
        Callback function compatible with inference log_callback parameter
    """
    return log_handler.get_callback()
