"""
Reusable Tkinter widgets for the SegOid GUI.

Provides file/folder picker widgets and other common UI components.
"""

import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path
from typing import Callable, Optional


class FolderPicker(ttk.Frame):
    """
    A folder picker widget with label, entry, and browse button.

    Attributes:
        path: StringVar containing the selected folder path
    """

    def __init__(
        self,
        parent,
        label: str,
        initial_path: str = "",
        on_change: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize folder picker widget.

        Args:
            parent: Parent Tkinter widget
            label: Label text to display
            initial_path: Initial folder path
            on_change: Callback when path changes
        """
        super().__init__(parent)

        self.on_change = on_change
        self.path = tk.StringVar(value=initial_path)
        self.path.trace_add("write", self._on_path_change)

        # Create widgets
        self.label = ttk.Label(self, text=label, width=20, anchor="e")
        self.label.pack(side=tk.LEFT, padx=(0, 5))

        self.entry = ttk.Entry(self, textvariable=self.path, width=50)
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        self.browse_btn = ttk.Button(
            self,
            text="Browse...",
            command=self._browse,
            width=10,
        )
        self.browse_btn.pack(side=tk.LEFT)

    def _browse(self):
        """Open folder selection dialog."""
        initial_dir = self.path.get() if self.path.get() else None
        folder = filedialog.askdirectory(
            title=f"Select {self.label.cget('text').strip(':')}",
            initialdir=initial_dir,
        )
        if folder:
            self.path.set(folder)

    def _on_path_change(self, *args):
        """Handle path change."""
        if self.on_change:
            self.on_change(self.path.get())

    def get(self) -> str:
        """Get current path."""
        return self.path.get()

    def set(self, value: str):
        """Set path value."""
        self.path.set(value)

    def is_valid(self) -> bool:
        """Check if path exists and is a directory."""
        p = Path(self.path.get())
        return p.exists() and p.is_dir()


class FilePicker(ttk.Frame):
    """
    A file picker widget with label, entry, and browse button.

    Attributes:
        path: StringVar containing the selected file path
    """

    def __init__(
        self,
        parent,
        label: str,
        initial_path: str = "",
        filetypes: list = None,
        on_change: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize file picker widget.

        Args:
            parent: Parent Tkinter widget
            label: Label text to display
            initial_path: Initial file path
            filetypes: List of (label, pattern) tuples for file dialog
            on_change: Callback when path changes
        """
        super().__init__(parent)

        self.on_change = on_change
        self.filetypes = filetypes or [("All files", "*.*")]
        self.path = tk.StringVar(value=initial_path)
        self.path.trace_add("write", self._on_path_change)

        # Create widgets
        self.label = ttk.Label(self, text=label, width=20, anchor="e")
        self.label.pack(side=tk.LEFT, padx=(0, 5))

        self.entry = ttk.Entry(self, textvariable=self.path, width=50)
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        self.browse_btn = ttk.Button(
            self,
            text="Browse...",
            command=self._browse,
            width=10,
        )
        self.browse_btn.pack(side=tk.LEFT)

    def _browse(self):
        """Open file selection dialog."""
        initial_dir = None
        initial_file = None

        current = self.path.get()
        if current:
            p = Path(current)
            if p.parent.exists():
                initial_dir = str(p.parent)
            initial_file = p.name

        filepath = filedialog.askopenfilename(
            title=f"Select {self.label.cget('text').strip(':')}",
            initialdir=initial_dir,
            initialfile=initial_file,
            filetypes=self.filetypes,
        )
        if filepath:
            self.path.set(filepath)

    def _on_path_change(self, *args):
        """Handle path change."""
        if self.on_change:
            self.on_change(self.path.get())

    def get(self) -> str:
        """Get current path."""
        return self.path.get()

    def set(self, value: str):
        """Set path value."""
        self.path.set(value)

    def is_valid(self) -> bool:
        """Check if path exists and is a file."""
        p = Path(self.path.get())
        return p.exists() and p.is_file()


class CheckboxOption(ttk.Frame):
    """
    A checkbox with label and optional detail widgets.
    """

    def __init__(
        self,
        parent,
        label: str,
        initial_value: bool = False,
        on_change: Optional[Callable[[bool], None]] = None,
    ):
        """
        Initialize checkbox option.

        Args:
            parent: Parent Tkinter widget
            label: Label text for checkbox
            initial_value: Initial checked state
            on_change: Callback when state changes
        """
        super().__init__(parent)

        self.on_change = on_change
        self.value = tk.BooleanVar(value=initial_value)
        self.value.trace_add("write", self._on_value_change)

        self.checkbox = ttk.Checkbutton(
            self,
            text=label,
            variable=self.value,
        )
        self.checkbox.pack(side=tk.LEFT)

    def _on_value_change(self, *args):
        """Handle value change."""
        if self.on_change:
            self.on_change(self.value.get())

    def get(self) -> bool:
        """Get current value."""
        return self.value.get()

    def set(self, value: bool):
        """Set value."""
        self.value.set(value)


class LabeledEntry(ttk.Frame):
    """
    A labeled entry widget for text input.
    """

    def __init__(
        self,
        parent,
        label: str,
        initial_value: str = "",
        width: int = 20,
        on_change: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize labeled entry.

        Args:
            parent: Parent Tkinter widget
            label: Label text
            initial_value: Initial entry text
            width: Entry width in characters
            on_change: Callback when text changes
        """
        super().__init__(parent)

        self.on_change = on_change
        self.value = tk.StringVar(value=initial_value)
        self.value.trace_add("write", self._on_value_change)

        self.label = ttk.Label(self, text=label)
        self.label.pack(side=tk.LEFT, padx=(0, 5))

        self.entry = ttk.Entry(self, textvariable=self.value, width=width)
        self.entry.pack(side=tk.LEFT)

    def _on_value_change(self, *args):
        """Handle value change."""
        if self.on_change:
            self.on_change(self.value.get())

    def get(self) -> str:
        """Get current value."""
        return self.value.get()

    def set(self, value: str):
        """Set value."""
        self.value.set(value)


class ScrolledText(ttk.Frame):
    """
    A text widget with scrollbar, suitable for log display.
    """

    def __init__(
        self,
        parent,
        height: int = 15,
        width: int = 80,
        readonly: bool = True,
    ):
        """
        Initialize scrolled text widget.

        Args:
            parent: Parent Tkinter widget
            height: Height in lines
            width: Width in characters
            readonly: Whether text is read-only
        """
        super().__init__(parent)

        # Create text widget
        self.text = tk.Text(
            self,
            height=height,
            width=width,
            wrap=tk.WORD,
            state=tk.DISABLED if readonly else tk.NORMAL,
        )
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create scrollbar
        self.scrollbar = ttk.Scrollbar(
            self,
            orient=tk.VERTICAL,
            command=self.text.yview,
        )
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.text.config(yscrollcommand=self.scrollbar.set)

    def get_text_widget(self) -> tk.Text:
        """Get the underlying Text widget."""
        return self.text

    def append(self, text: str, tag: str = None):
        """Append text to the widget."""
        self.text.config(state=tk.NORMAL)
        if tag:
            self.text.insert(tk.END, text, tag)
        else:
            self.text.insert(tk.END, text)
        self.text.see(tk.END)
        self.text.config(state=tk.DISABLED)

    def clear(self):
        """Clear all text."""
        self.text.config(state=tk.NORMAL)
        self.text.delete(1.0, tk.END)
        self.text.config(state=tk.DISABLED)
