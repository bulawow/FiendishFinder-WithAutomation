#!/usr/bin/env python3
"""
Dataset Browser UI for FiendishFinder

PyQt6-based UI for browsing, viewing, and managing dataset entries.
"""

import sys
from pathlib import Path
from typing import Optional, List

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QListWidgetItem, QMessageBox,
    QGroupBox, QFormLayout, QTextEdit, QSplitter, QTabWidget,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QDialog,
    QDialogButtonBox, QLineEdit, QComboBox, QCheckBox, QDoubleSpinBox,
    QScrollArea
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QPainter

from src.models.dataset_models import (
    DatasetManager, MinimapDatasetEntry, ExivaDatasetEntry,
    ExivaRange, ExivaDirection, MonsterDifficulty, FloorIndication
)
import time


class MinimapEntryEditDialog(QDialog):
    """Dialog for editing minimap dataset entries."""

    def __init__(self, entry: MinimapDatasetEntry, parent=None):
        super().__init__(parent)
        self.entry = entry
        self.setWindowTitle(f"Edit Minimap Entry - {entry.entry_id[:8]}...")
        self.setModal(True)
        self.resize(500, 400)

        self.setup_ui()
        self.load_entry_data()

    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        # Form layout for entry fields
        form_group = QGroupBox("Entry Data")
        form_layout = QFormLayout()

        # Entry ID (read-only)
        self.id_label = QLabel()
        form_layout.addRow("Entry ID:", self.id_label)

        # Floor
        self.floor_spinbox = QDoubleSpinBox()
        self.floor_spinbox.setRange(0, 15)
        self.floor_spinbox.setDecimals(0)
        form_layout.addRow("Floor:", self.floor_spinbox)

        # Crosshair X position
        self.x_spinbox = QDoubleSpinBox()
        self.x_spinbox.setRange(0, 10000)
        self.x_spinbox.setDecimals(1)
        form_layout.addRow("Crosshair X:", self.x_spinbox)

        # Crosshair Y position
        self.y_spinbox = QDoubleSpinBox()
        self.y_spinbox.setRange(0, 10000)
        self.y_spinbox.setDecimals(1)
        form_layout.addRow("Crosshair Y:", self.y_spinbox)

        # Image dimensions (read-only)
        self.size_label = QLabel()
        form_layout.addRow("Image Size:", self.size_label)

        # Notes
        self.notes_edit = QTextEdit()
        self.notes_edit.setMaximumHeight(100)
        form_layout.addRow("Notes:", self.notes_edit)

        form_group.setLayout(form_layout)
        layout.addWidget(form_group)

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def load_entry_data(self):
        """Load entry data into the form."""
        self.id_label.setText(self.entry.entry_id)
        self.floor_spinbox.setValue(self.entry.floor)
        self.x_spinbox.setValue(self.entry.crosshair_x)
        self.y_spinbox.setValue(self.entry.crosshair_y)
        self.size_label.setText(f"{self.entry.image_width} x {self.entry.image_height}")
        self.notes_edit.setPlainText(self.entry.notes)

    def get_updated_entry(self) -> MinimapDatasetEntry:
        """Get the updated entry with modified values."""
        return MinimapDatasetEntry(
            entry_id=self.entry.entry_id,
            screenshot_path=self.entry.screenshot_path,
            crosshair_x=self.x_spinbox.value(),
            crosshair_y=self.y_spinbox.value(),
            floor=int(self.floor_spinbox.value()),
            image_width=self.entry.image_width,
            image_height=self.entry.image_height,
            notes=self.notes_edit.toPlainText(),
            created_timestamp=self.entry.created_timestamp,
            modified_timestamp=time.time()
        )


class ExivaEntryEditDialog(QDialog):
    """Dialog for editing Exiva dataset entries."""

    def __init__(self, entry: ExivaDatasetEntry, parent=None):
        super().__init__(parent)
        self.entry = entry
        self.setWindowTitle(f"Edit Exiva Entry - {entry.entry_id[:8]}...")
        self.setModal(True)
        self.resize(500, 500)

        self.setup_ui()
        self.load_entry_data()

    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        # Form layout for entry fields
        form_group = QGroupBox("Entry Data")
        form_layout = QFormLayout()

        # Entry ID (read-only)
        self.id_label = QLabel()
        form_layout.addRow("Entry ID:", self.id_label)

        # Range
        self.range_combo = QComboBox()
        for range_val in ExivaRange:
            self.range_combo.addItem(range_val.value, range_val)
        form_layout.addRow("Range:", self.range_combo)

        # Direction
        self.direction_combo = QComboBox()
        for direction in ExivaDirection:
            self.direction_combo.addItem(direction.value, direction)
        form_layout.addRow("Direction:", self.direction_combo)

        # Difficulty
        self.difficulty_combo = QComboBox()
        for difficulty in MonsterDifficulty:
            self.difficulty_combo.addItem(difficulty.value, difficulty)
        form_layout.addRow("Difficulty:", self.difficulty_combo)

        # Floor indication
        self.floor_combo = QComboBox()
        for floor in FloorIndication:
            self.floor_combo.addItem(floor.value, floor)
        form_layout.addRow("Floor Indication:", self.floor_combo)

        # Raw text
        self.raw_text_edit = QLineEdit()
        form_layout.addRow("Raw Text:", self.raw_text_edit)

        # Image dimensions (read-only)
        self.size_label = QLabel()
        form_layout.addRow("Image Size:", self.size_label)

        # Notes
        self.notes_edit = QTextEdit()
        self.notes_edit.setMaximumHeight(80)
        form_layout.addRow("Notes:", self.notes_edit)

        form_group.setLayout(form_layout)
        layout.addWidget(form_group)

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def load_entry_data(self):
        """Load entry data into the form."""
        self.id_label.setText(self.entry.entry_id)

        # Set combo box values
        for i in range(self.range_combo.count()):
            if self.range_combo.itemData(i).value == self.entry.range:
                self.range_combo.setCurrentIndex(i)
                break

        for i in range(self.direction_combo.count()):
            if self.direction_combo.itemData(i).value == self.entry.direction:
                self.direction_combo.setCurrentIndex(i)
                break

        for i in range(self.difficulty_combo.count()):
            if self.difficulty_combo.itemData(i).value == self.entry.difficulty:
                self.difficulty_combo.setCurrentIndex(i)
                break

        for i in range(self.floor_combo.count()):
            if self.floor_combo.itemData(i).value == self.entry.floor_indication:
                self.floor_combo.setCurrentIndex(i)
                break

        self.raw_text_edit.setText(self.entry.raw_text if self.entry.raw_text else "")
        self.size_label.setText(f"{self.entry.image_width} x {self.entry.image_height}")
        self.notes_edit.setPlainText(self.entry.notes)

    def get_updated_entry(self) -> ExivaDatasetEntry:
        """Get the updated entry with modified values."""
        return ExivaDatasetEntry(
            entry_id=self.entry.entry_id,
            screenshot_path=self.entry.screenshot_path,
            range=self.range_combo.currentData().value,
            direction=self.direction_combo.currentData().value,
            difficulty=self.difficulty_combo.currentData().value,
            floor_indication=self.floor_combo.currentData().value,
            image_width=self.entry.image_width,
            image_height=self.entry.image_height,
            notes=self.notes_edit.toPlainText(),
            raw_text=self.raw_text_edit.text(),
            created_timestamp=self.entry.created_timestamp,
            modified_timestamp=time.time()
        )


class DatasetBrowserWindow(QMainWindow):
    """Main window for browsing and managing datasets."""

    def __init__(self, parent=None, graphics_view=None, set_floor_callback=None):
        super().__init__(parent)
        self.dataset_manager = DatasetManager()
        self.graphics_view = graphics_view  # Reference to main viewer's graphics view
        self.set_floor_callback = set_floor_callback  # Callback to change floor in main viewer

        self.setWindowTitle("Dataset Browser - FiendishFinder")
        self.setGeometry(100, 100, 1200, 700)

        self.setup_ui()
        self.load_datasets()

    def setup_ui(self):
        """Set up the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        
        # Title and stats
        title_layout = QHBoxLayout()
        title = QLabel("Dataset Browser")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        title_layout.addWidget(title)
        
        title_layout.addStretch()
        
        self.stats_label = QLabel("Loading...")
        title_layout.addWidget(self.stats_label)
        
        layout.addLayout(title_layout)
        
        # Tab widget for different dataset types
        self.tab_widget = QTabWidget()

        # Minimap dataset tab
        self.minimap_tab = self.create_minimap_tab()
        self.tab_widget.addTab(self.minimap_tab, "Minimap Dataset")
        
        # Exiva dataset tab
        self.exiva_tab = self.create_exiva_tab()
        self.tab_widget.addTab(self.exiva_tab, "Exiva Dataset")
        
        layout.addWidget(self.tab_widget)
        
        # Status bar
        self.statusBar().showMessage("Ready")

    def create_minimap_tab(self) -> QWidget:
        """Create the minimap dataset tab."""
        tab = QWidget()
        layout = QHBoxLayout(tab)

        # Left side: List of entries
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        left_layout.addWidget(QLabel("Minimap Dataset Entries:"))

        self.minimap_list = QListWidget()
        self.minimap_list.currentItemChanged.connect(self.on_minimap_entry_selected)
        left_layout.addWidget(self.minimap_list)

        # Buttons
        button_layout = QHBoxLayout()

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.load_datasets)
        button_layout.addWidget(refresh_btn)

        edit_btn = QPushButton("Edit")
        edit_btn.clicked.connect(self.edit_minimap_entry)
        button_layout.addWidget(edit_btn)

        delete_btn = QPushButton("Delete Entry")
        delete_btn.clicked.connect(self.delete_minimap_entry)
        button_layout.addWidget(delete_btn)

        left_layout.addLayout(button_layout)

        # Place crosshair button
        place_minimap_btn = QPushButton("Place Crosshair on Map")
        place_minimap_btn.setStyleSheet("font-weight: bold; background-color: #4CAF50; color: white;")
        place_minimap_btn.clicked.connect(self.place_minimap_on_map)
        left_layout.addWidget(place_minimap_btn)

        # Right side: Entry details
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        right_layout.addWidget(QLabel("Entry Details:"))

        # Minimap preview
        preview_group = QGroupBox("Minimap")
        preview_layout = QVBoxLayout()
        self.minimap_preview = QGraphicsView()
        self.minimap_preview_scene = QGraphicsScene()
        self.minimap_preview.setScene(self.minimap_preview_scene)
        self.minimap_preview.setRenderHint(QPainter.RenderHint.Antialiasing)
        preview_layout.addWidget(self.minimap_preview)
        preview_group.setLayout(preview_layout)
        right_layout.addWidget(preview_group, stretch=1)

        # Details
        details_group = QGroupBox("Details")
        details_layout = QFormLayout()

        self.minimap_id_label = QLabel("-")
        details_layout.addRow("Entry ID:", self.minimap_id_label)

        self.minimap_floor_label = QLabel("-")
        details_layout.addRow("Floor:", self.minimap_floor_label)

        self.minimap_pos_label = QLabel("-")
        details_layout.addRow("Crosshair Position:", self.minimap_pos_label)

        self.minimap_size_label = QLabel("-")
        details_layout.addRow("Image Size:", self.minimap_size_label)

        self.minimap_notes_edit = QTextEdit()
        self.minimap_notes_edit.setMaximumHeight(80)
        self.minimap_notes_edit.setReadOnly(True)
        details_layout.addRow("Notes:", self.minimap_notes_edit)
        
        details_group.setLayout(details_layout)
        right_layout.addWidget(details_group)
        
        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        
        layout.addWidget(splitter)
        
        return tab

    def create_exiva_tab(self) -> QWidget:
        """Create the Exiva dataset tab."""
        tab = QWidget()
        layout = QHBoxLayout(tab)

        # Left side: List of entries
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Statistics section
        stats_group = QGroupBox("Dataset Statistics")
        stats_layout = QVBoxLayout()

        # Create a scrollable area for statistics
        stats_scroll = QScrollArea()
        stats_scroll.setWidgetResizable(True)
        stats_scroll.setMaximumHeight(250)

        stats_content = QWidget()
        stats_content_layout = QVBoxLayout(stats_content)

        self.exiva_stats_label = QLabel("Loading statistics...")
        self.exiva_stats_label.setWordWrap(True)
        self.exiva_stats_label.setTextFormat(Qt.TextFormat.RichText)
        stats_content_layout.addWidget(self.exiva_stats_label)

        stats_scroll.setWidget(stats_content)
        stats_layout.addWidget(stats_scroll)
        stats_group.setLayout(stats_layout)
        left_layout.addWidget(stats_group)

        left_layout.addWidget(QLabel("Exiva Dataset Entries:"))

        self.exiva_list = QListWidget()
        self.exiva_list.currentItemChanged.connect(self.on_exiva_entry_selected)
        left_layout.addWidget(self.exiva_list)
        
        # Buttons
        button_layout = QHBoxLayout()

        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.load_datasets)
        button_layout.addWidget(refresh_btn)

        edit_btn = QPushButton("Edit")
        edit_btn.clicked.connect(self.edit_exiva_entry)
        button_layout.addWidget(edit_btn)

        delete_btn = QPushButton("Delete Entry")
        delete_btn.clicked.connect(self.delete_exiva_entry)
        button_layout.addWidget(delete_btn)

        left_layout.addLayout(button_layout)
        
        # Right side: Entry details
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        right_layout.addWidget(QLabel("Entry Details:"))
        
        # Screenshot preview
        preview_group = QGroupBox("Screenshot")
        preview_layout = QVBoxLayout()
        self.exiva_preview = QGraphicsView()
        self.exiva_preview_scene = QGraphicsScene()
        self.exiva_preview.setScene(self.exiva_preview_scene)
        self.exiva_preview.setRenderHint(QPainter.RenderHint.Antialiasing)
        preview_layout.addWidget(self.exiva_preview)
        preview_group.setLayout(preview_layout)
        right_layout.addWidget(preview_group, stretch=1)
        
        # Details
        details_group = QGroupBox("Exiva Data")
        details_layout = QFormLayout()
        
        self.exiva_id_label = QLabel("-")
        details_layout.addRow("Entry ID:", self.exiva_id_label)
        
        self.exiva_range_label = QLabel("-")
        details_layout.addRow("Range:", self.exiva_range_label)
        
        self.exiva_direction_label = QLabel("-")
        details_layout.addRow("Direction:", self.exiva_direction_label)
        
        self.exiva_difficulty_label = QLabel("-")
        details_layout.addRow("Difficulty:", self.exiva_difficulty_label)
        
        self.exiva_floor_label = QLabel("-")
        details_layout.addRow("Floor Indication:", self.exiva_floor_label)

        self.exiva_raw_text_label = QLabel("-")
        self.exiva_raw_text_label.setWordWrap(True)
        details_layout.addRow("Raw Text:", self.exiva_raw_text_label)
        
        self.exiva_notes_edit = QTextEdit()
        self.exiva_notes_edit.setMaximumHeight(60)
        self.exiva_notes_edit.setReadOnly(True)
        details_layout.addRow("Notes:", self.exiva_notes_edit)
        
        details_group.setLayout(details_layout)
        right_layout.addWidget(details_group)
        
        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        
        layout.addWidget(splitter)
        
        return tab

    def load_datasets(self):
        """Load all datasets and update the UI."""
        # Check minimap dataset size first
        minimap_count = self.dataset_manager.get_minimap_dataset_count()

        # Warn if dataset is very large
        if minimap_count > 10000:
            reply = QMessageBox.warning(
                self,
                "Large Dataset Warning",
                f"The minimap dataset contains {minimap_count:,} entries.\n\n"
                f"Loading all entries in this window may cause lag or freezing.\n\n"
                f"Recommendation: Use the Dataset Browser tab in the main window instead,\n"
                f"which has pagination support for large datasets.\n\n"
                f"Do you want to continue loading all entries here?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.No:
                self.statusBar().showMessage(f"Dataset loading cancelled ({minimap_count:,} entries)")
                return

        self.statusBar().showMessage("Loading datasets...")
        QApplication.processEvents()

        # Load minimap dataset
        minimap_entries = self.dataset_manager.load_minimap_dataset()
        self.minimap_list.clear()
        for entry in minimap_entries:
            item = QListWidgetItem(f"{entry.entry_id[:8]}... - Floor {entry.floor} - ({entry.crosshair_x:.0f}, {entry.crosshair_y:.0f})")
            item.setData(Qt.ItemDataRole.UserRole, entry)
            self.minimap_list.addItem(item)

        # Load Exiva dataset
        exiva_entries = self.dataset_manager.load_exiva_dataset()
        self.exiva_list.clear()
        for entry in exiva_entries:
            item = QListWidgetItem(f"{entry.entry_id[:8]}... - {entry.range} {entry.direction}")
            item.setData(Qt.ItemDataRole.UserRole, entry)
            self.exiva_list.addItem(item)

        # Update Exiva statistics
        self.update_exiva_statistics(exiva_entries)

        # Update stats
        stats = self.dataset_manager.get_dataset_stats()
        self.stats_label.setText(
            f"Minimap: {stats['minimap_count']} | Exiva: {stats['exiva_count']} | Total: {stats['total_count']}"
        )

        self.statusBar().showMessage("Datasets loaded")

    def update_exiva_statistics(self, exiva_entries: List[ExivaDatasetEntry]):
        """Update the Exiva statistics display."""
        from collections import Counter

        if not exiva_entries:
            self.exiva_stats_label.setText("<b>No entries in dataset</b>")
            return

        # Count occurrences of each category
        range_counts = Counter(entry.range for entry in exiva_entries)
        direction_counts = Counter(entry.direction for entry in exiva_entries)
        difficulty_counts = Counter(entry.difficulty for entry in exiva_entries)
        floor_counts = Counter(entry.floor_indication for entry in exiva_entries)

        # Build HTML formatted statistics
        html = f"<b>Total Entries: {len(exiva_entries)}</b><br><br>"

        # Range statistics - show all possible values
        html += "<b>Range:</b><br>"
        range_order = ["none", "to the", "far", "very far"]
        for range_val in range_order:
            count = range_counts.get(range_val, 0)
            html += f"&nbsp;&nbsp;• {range_val}: {count}<br>"
        html += "<br>"

        # Direction statistics - show all possible values
        html += "<b>Direction:</b><br>"
        direction_order = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "none"]
        for direction_val in direction_order:
            count = direction_counts.get(direction_val, 0)
            html += f"&nbsp;&nbsp;• {direction_val}: {count}<br>"
        html += "<br>"

        # Difficulty statistics - show all possible values
        html += "<b>Monster Difficulty:</b><br>"
        difficulty_order = ["none", "unknown", "harmless", "trivial", "easy", "medium", "hard", "challenging"]
        for difficulty_val in difficulty_order:
            count = difficulty_counts.get(difficulty_val, 0)
            html += f"&nbsp;&nbsp;• {difficulty_val}: {count}<br>"
        html += "<br>"

        # Floor indication statistics - show all possible values
        html += "<b>Floor Indication:</b><br>"
        floor_order = ["none", "higher level", "lower level"]
        for floor_val in floor_order:
            count = floor_counts.get(floor_val, 0)
            html += f"&nbsp;&nbsp;• {floor_val}: {count}<br>"

        self.exiva_stats_label.setText(html)

    def on_minimap_entry_selected(self, current: QListWidgetItem, previous: QListWidgetItem):
        """Handle minimap entry selection."""
        if not current:
            return

        entry: MinimapDatasetEntry = current.data(Qt.ItemDataRole.UserRole)

        # Update details
        self.minimap_id_label.setText(entry.entry_id)
        self.minimap_floor_label.setText(str(entry.floor))
        self.minimap_pos_label.setText(f"({entry.crosshair_x:.1f}, {entry.crosshair_y:.1f})")
        self.minimap_size_label.setText(f"{entry.image_width} x {entry.image_height}")
        self.minimap_notes_edit.setPlainText(entry.notes)

        # Load and display screenshot
        screenshot_path = self.dataset_manager.get_screenshot_full_path(entry.screenshot_path)
        if screenshot_path.exists():
            pixmap = QPixmap(str(screenshot_path))
            self.minimap_preview_scene.clear()
            pixmap_item = QGraphicsPixmapItem(pixmap)
            self.minimap_preview_scene.addItem(pixmap_item)
            self.minimap_preview_scene.setSceneRect(pixmap_item.boundingRect())
            self.minimap_preview.fitInView(pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def on_exiva_entry_selected(self, current: QListWidgetItem, previous: QListWidgetItem):
        """Handle Exiva entry selection."""
        if not current:
            return

        entry: ExivaDatasetEntry = current.data(Qt.ItemDataRole.UserRole)

        # Update details
        self.exiva_id_label.setText(entry.entry_id)
        self.exiva_range_label.setText(entry.range)
        self.exiva_direction_label.setText(entry.direction)
        self.exiva_difficulty_label.setText(entry.difficulty)
        self.exiva_floor_label.setText(entry.floor_indication)
        self.exiva_raw_text_label.setText(entry.raw_text if entry.raw_text else "-")
        self.exiva_notes_edit.setPlainText(entry.notes)

        # Load and display screenshot
        screenshot_path = self.dataset_manager.get_screenshot_full_path(entry.screenshot_path)
        if screenshot_path.exists():
            pixmap = QPixmap(str(screenshot_path))
            self.exiva_preview_scene.clear()
            pixmap_item = QGraphicsPixmapItem(pixmap)
            self.exiva_preview_scene.addItem(pixmap_item)
            self.exiva_preview_scene.setSceneRect(pixmap_item.boundingRect())
            self.exiva_preview.fitInView(pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def edit_minimap_entry(self):
        """Edit the selected minimap entry."""
        current_item = self.minimap_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Selection", "Please select an entry to edit.")
            return

        entry: MinimapDatasetEntry = current_item.data(Qt.ItemDataRole.UserRole)

        # Open edit dialog
        dialog = MinimapEntryEditDialog(entry, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            updated_entry = dialog.get_updated_entry()

            # Update in dataset
            if self.dataset_manager.update_minimap_entry(entry.entry_id, updated_entry):
                self.load_datasets()
                self.statusBar().showMessage("Entry updated successfully")

                # Re-select the updated entry
                for i in range(self.minimap_list.count()):
                    item = self.minimap_list.item(i)
                    item_entry = item.data(Qt.ItemDataRole.UserRole)
                    if item_entry.entry_id == entry.entry_id:
                        self.minimap_list.setCurrentItem(item)
                        break
            else:
                QMessageBox.critical(self, "Error", "Failed to update entry.")

    def edit_exiva_entry(self):
        """Edit the selected Exiva entry."""
        current_item = self.exiva_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Selection", "Please select an entry to edit.")
            return

        entry: ExivaDatasetEntry = current_item.data(Qt.ItemDataRole.UserRole)

        # Open edit dialog
        dialog = ExivaEntryEditDialog(entry, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            updated_entry = dialog.get_updated_entry()

            # Update in dataset
            if self.dataset_manager.update_exiva_entry(entry.entry_id, updated_entry):
                self.load_datasets()
                self.statusBar().showMessage("Entry updated successfully")

                # Re-select the updated entry
                for i in range(self.exiva_list.count()):
                    item = self.exiva_list.item(i)
                    item_entry = item.data(Qt.ItemDataRole.UserRole)
                    if item_entry.entry_id == entry.entry_id:
                        self.exiva_list.setCurrentItem(item)
                        break
            else:
                QMessageBox.critical(self, "Error", "Failed to update entry.")

    def delete_minimap_entry(self):
        """Delete the selected minimap entry."""
        current_item = self.minimap_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Selection", "Please select an entry to delete.")
            return

        entry: MinimapDatasetEntry = current_item.data(Qt.ItemDataRole.UserRole)

        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete entry {entry.entry_id[:8]}...?\n\n"
            "This will remove the entry from the dataset AND delete the screenshot file.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            if self.dataset_manager.delete_minimap_entry(entry.entry_id):
                self.load_datasets()
                self.statusBar().showMessage("Entry and screenshot deleted successfully")
            else:
                QMessageBox.critical(self, "Error", "Failed to delete entry.")

    def delete_exiva_entry(self):
        """Delete the selected Exiva entry."""
        current_item = self.exiva_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Selection", "Please select an entry to delete.")
            return

        entry: ExivaDatasetEntry = current_item.data(Qt.ItemDataRole.UserRole)

        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete entry {entry.entry_id[:8]}...?\n\n"
            "This will remove the entry from the dataset AND delete the screenshot file.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            if self.dataset_manager.delete_exiva_entry(entry.entry_id):
                self.load_datasets()
                self.statusBar().showMessage("Entry and screenshot deleted successfully")
            else:
                QMessageBox.critical(self, "Error", "Failed to delete entry.")

    def place_minimap_on_map(self):
        """Place the crosshair on the map using ground truth data from the selected entry."""
        current_item = self.minimap_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "No Selection", "Please select a minimap entry first.")
            return

        # Check if we have access to the graphics view
        if not self.graphics_view:
            QMessageBox.warning(
                self,
                "Not Available",
                "This feature requires the main viewer to be open.\n\n"
                "The dataset browser was opened without a reference to the main viewer."
            )
            return

        entry: MinimapDatasetEntry = current_item.data(Qt.ItemDataRole.UserRole)

        # Import QPointF here to avoid circular imports
        from PyQt6.QtCore import QPointF

        # Set the floor if callback is available
        if self.set_floor_callback:
            self.set_floor_callback(entry.floor)

        # Place the crosshair at the ground truth position
        crosshair_pos = QPointF(entry.crosshair_x, entry.crosshair_y)
        self.graphics_view.place_crosshairs(crosshair_pos)

        # Show success message
        self.statusBar().showMessage(
            f"Crosshair placed at ({entry.crosshair_x:.1f}, {entry.crosshair_y:.1f}) on floor {entry.floor}"
        )

        # Optionally, bring the main window to front
        if self.parent():
            self.parent().raise_()
            self.parent().activateWindow()


def main():
    """Main function for testing the dataset browser UI."""
    app = QApplication(sys.argv)

    window = DatasetBrowserWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

