#!/usr/bin/env python3
"""
Dataset Creator UI for FiendishFinder

PyQt6-based UI components for creating crosshair and Exiva datasets.
"""

import sys
import uuid
import time
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QTextEdit,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsEllipseItem,
    QGraphicsRectItem, QGroupBox, QFormLayout, QComboBox, QCheckBox, QLineEdit, QSpinBox,
    QListWidget, QListWidgetItem, QSplitter, QScrollArea, QColorDialog
)
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt6.QtGui import QPixmap, QPen, QColor, QBrush, QMouseEvent, QPainter, QImage

from src.models.dataset_models import (
    DatasetManager, MinimapDatasetEntry, ExivaDatasetEntry,
    ExivaRange, ExivaDirection, MonsterDifficulty, FloorIndication
)


class MinimapGraphicsView(QGraphicsView):
    """Graphics view for displaying minimap and marking crosshair position."""

    crosshairMarked = pyqtSignal(float, float)  # x, y coordinates

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)

        self.crosshair_marker: Optional[QGraphicsEllipseItem] = None
        self.crosshair_position: Optional[QPointF] = None

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press to mark crosshair position."""
        if event.button() == Qt.MouseButton.LeftButton:
            # Get scene position
            scene_pos = self.mapToScene(event.pos())

            # Mark the crosshair position
            self.mark_crosshair(scene_pos)

            # Emit signal
            self.crosshairMarked.emit(scene_pos.x(), scene_pos.y())
        else:
            super().mousePressEvent(event)

    def mark_crosshair(self, pos: QPointF):
        """Mark the crosshair position with a visual indicator."""
        # Remove existing marker
        if self.crosshair_marker:
            self.scene().removeItem(self.crosshair_marker)

        # Create new marker (red circle)
        radius = 10
        self.crosshair_marker = QGraphicsEllipseItem(
            pos.x() - radius, pos.y() - radius,
            radius * 2, radius * 2
        )
        self.crosshair_marker.setPen(QPen(QColor(255, 0, 0), 2))
        self.crosshair_marker.setBrush(QBrush(QColor(255, 0, 0, 100)))
        self.scene().addItem(self.crosshair_marker)

        self.crosshair_position = pos

    def clear_marker(self):
        """Clear the crosshair marker."""
        if self.crosshair_marker:
            self.scene().removeItem(self.crosshair_marker)
            self.crosshair_marker = None
        self.crosshair_position = None

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming."""
        zoom_factor = 1.15
        if event.angleDelta().y() > 0:
            self.scale(zoom_factor, zoom_factor)
        else:
            self.scale(1 / zoom_factor, 1 / zoom_factor)


class MinimapDatasetCreatorPanel(QWidget):
    """Panel for creating minimap dataset entries - integrated into main viewer."""

    def __init__(self, dataset_manager: DatasetManager, get_crosshair_callback, graphics_view, get_floor_callback=None, parent=None):
        super().__init__(parent)
        self.dataset_manager = dataset_manager
        self.get_crosshair_callback = get_crosshair_callback
        self.graphics_view = graphics_view
        self.get_floor_callback = get_floor_callback

        # Import WalkableDetector for color management
        from src.utils.walkable_detector import WalkableDetector
        self.walkable_detector = WalkableDetector()

        self.setup_ui()

        # Connect to crosshair signal if graphics_view is available
        if self.graphics_view is not None:
            self.graphics_view.crosshairPlaced.connect(self.on_crosshair_placed)

    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        # Title
        title = QLabel("Minimap Dataset Creator")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title)

        # Instructions
        instructions = QLabel(
            "Generate minimap crops from walkable positions on the processed floor map."
        )
        instructions.setStyleSheet("color: gray; font-size: 11px;")
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Walkable feedback section
        feedback_group = QGroupBox("Crosshair Position Feedback")
        feedback_layout = QVBoxLayout()

        # Position info
        self.position_label = QLabel("Position: Not set")
        self.position_label.setStyleSheet("font-size: 11px; color: gray;")
        feedback_layout.addWidget(self.position_label)

        # Walkable status
        self.walkable_status_label = QLabel("Status: Place crosshair to check")
        self.walkable_status_label.setStyleSheet("font-size: 12px; font-weight: bold; padding: 8px; background-color: #f0f0f0; border-radius: 4px;")
        feedback_layout.addWidget(self.walkable_status_label)

        # Color info
        self.color_info_label = QLabel("Color: -")
        self.color_info_label.setStyleSheet("font-size: 10px; color: gray;")
        feedback_layout.addWidget(self.color_info_label)

        feedback_group.setLayout(feedback_layout)
        layout.addWidget(feedback_group)

        # Manual generation from crosshair position
        manual_gen_group = QGroupBox("Generate from Crosshair Position")
        manual_gen_layout = QVBoxLayout()

        # Instructions
        manual_instructions = QLabel(
            "1. Click on the map to place crosshair (player position)\n"
            "2. Click 'Generate' to crop minimap around that position\n"
            "3. Crosshair coordinates = player position on floor map"
        )
        manual_instructions.setStyleSheet("color: gray; font-size: 10px;")
        manual_instructions.setWordWrap(True)
        manual_gen_layout.addWidget(manual_instructions)

        # Generate button
        self.generate_btn = QPushButton("Generate Entry from Crosshair")
        self.generate_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.generate_btn.clicked.connect(self.generate_entry_from_crosshair)
        manual_gen_layout.addWidget(self.generate_btn)

        # Status label
        self.gen_status_label = QLabel("")
        self.gen_status_label.setStyleSheet("color: gray; font-size: 10px;")
        self.gen_status_label.setWordWrap(True)
        manual_gen_layout.addWidget(self.gen_status_label)

        manual_gen_group.setLayout(manual_gen_layout)
        layout.addWidget(manual_gen_group)

        # Auto-generation section
        auto_gen_group = QGroupBox("Automatic Batch Generation")
        auto_gen_layout = QVBoxLayout()

        # Instructions
        auto_instructions = QLabel(
            "Automatically generate multiple entries from random walkable positions."
        )
        auto_instructions.setStyleSheet("color: gray; font-size: 10px;")
        auto_instructions.setWordWrap(True)
        auto_gen_layout.addWidget(auto_instructions)

        # Floor and count selection
        gen_controls_layout = QHBoxLayout()

        gen_controls_layout.addWidget(QLabel("Floor:"))
        self.gen_floor_spin = QSpinBox()
        self.gen_floor_spin.setRange(0, 15)
        self.gen_floor_spin.setValue(7)
        gen_controls_layout.addWidget(self.gen_floor_spin)

        gen_controls_layout.addWidget(QLabel("Count:"))
        self.gen_count_spin = QSpinBox()
        self.gen_count_spin.setRange(1, 50000)
        self.gen_count_spin.setValue(10)
        gen_controls_layout.addWidget(self.gen_count_spin)

        auto_gen_layout.addLayout(gen_controls_layout)

        # Auto generate button
        self.auto_generate_btn = QPushButton("Auto-Generate Entries")
        self.auto_generate_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 10px;")
        self.auto_generate_btn.clicked.connect(self.generate_entries_from_map)
        auto_gen_layout.addWidget(self.auto_generate_btn)

        # Auto status label
        self.auto_gen_status_label = QLabel("")
        self.auto_gen_status_label.setStyleSheet("color: gray; font-size: 10px;")
        self.auto_gen_status_label.setWordWrap(True)
        auto_gen_layout.addWidget(self.auto_gen_status_label)

        auto_gen_group.setLayout(auto_gen_layout)
        layout.addWidget(auto_gen_group)

        # Non-walkable color management
        color_group = QGroupBox("Non-Walkable Colors")
        color_layout = QVBoxLayout()

        # Instructions
        color_instructions = QLabel(
            "Add colors that represent non-walkable areas (water, lava, walls, etc.)"
        )
        color_instructions.setStyleSheet("color: gray; font-size: 10px;")
        color_instructions.setWordWrap(True)
        color_layout.addWidget(color_instructions)

        # Color picker button
        color_picker_layout = QHBoxLayout()
        self.pick_color_btn = QPushButton("Pick Color")
        self.pick_color_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 8px;")
        self.pick_color_btn.clicked.connect(self.pick_color)
        color_picker_layout.addWidget(self.pick_color_btn)

        # Manual RGB input
        self.rgb_input = QLineEdit()
        self.rgb_input.setPlaceholderText("R,G,B (e.g., 255,0,0)")
        self.rgb_input.setMaximumWidth(150)
        color_picker_layout.addWidget(self.rgb_input)

        self.add_rgb_btn = QPushButton("Add")
        self.add_rgb_btn.setStyleSheet("background-color: #4CAF50; color: white; padding: 8px;")
        self.add_rgb_btn.clicked.connect(self.add_rgb_color)
        color_picker_layout.addWidget(self.add_rgb_btn)

        color_layout.addLayout(color_picker_layout)

        # Color list
        color_list_label = QLabel("Current Non-Walkable Colors:")
        color_list_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        color_layout.addWidget(color_list_label)

        self.color_list = QListWidget()
        self.color_list.setMaximumHeight(200)
        self.color_list.setStyleSheet("""
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #ddd;
            }
            QListWidget::item:selected {
                background-color: #e3f2fd;
                color: black;
            }
        """)
        color_layout.addWidget(self.color_list)

        # Remove color button
        self.remove_color_btn = QPushButton("Remove Selected Color")
        self.remove_color_btn.setStyleSheet("background-color: #f44336; color: white; padding: 8px;")
        self.remove_color_btn.clicked.connect(self.remove_selected_color)
        color_layout.addWidget(self.remove_color_btn)

        color_group.setLayout(color_layout)
        layout.addWidget(color_group)

        # Load existing colors
        self.refresh_color_list()

        layout.addStretch()

    def generate_entry_from_crosshair(self):
        """Generate dataset entry from the crosshair position (player position)."""
        from src.utils.minimap_dataset_generator import MinimapDatasetGenerator
        from PyQt6.QtCore import QPointF

        # Check if crosshair is placed
        if self.get_crosshair_callback is None:
            QMessageBox.warning(
                self,
                "No Crosshair Callback",
                "Crosshair callback is not available."
            )
            return

        crosshair_pos = self.get_crosshair_callback()
        if crosshair_pos is None or not isinstance(crosshair_pos, QPointF):
            QMessageBox.warning(
                self,
                "No Crosshair",
                "Please place a crosshair on the map first by clicking on the desired position."
            )
            return

        # Get current floor
        if self.get_floor_callback is None:
            QMessageBox.warning(
                self,
                "No Floor",
                "Floor callback is not available."
            )
            return

        floor = self.get_floor_callback()

        # Get crosshair coordinates (player position on floor map)
        player_x = int(crosshair_pos.x())
        player_y = int(crosshair_pos.y())

        # Update status
        self.gen_status_label.setText(f"Generating entry at player position ({player_x}, {player_y}) on floor {floor}...")
        self.generate_btn.setEnabled(False)

        try:
            # Create generator
            generator = MinimapDatasetGenerator(
                minimap_dir="processed_minimap",
                dataset_manager=self.dataset_manager,
                minimap_size=(106, 106)  # Standard Tibia minimap size
            )

            # Generate entry from player position
            entry = generator.generate_entry_from_position(floor, player_x, player_y)

            if entry is not None:
                # Save the entry
                if self.dataset_manager.add_minimap_entry(entry):
                    self.gen_status_label.setText(
                        f"✓ Successfully generated entry at player position ({player_x}, {player_y}) on floor {floor}!"
                    )
                    self.gen_status_label.setStyleSheet("color: green; font-size: 10px; font-weight: bold;")

                    QMessageBox.information(
                        self,
                        "Success",
                        f"Successfully generated dataset entry!\n\n"
                        f"Player position on floor map: ({player_x}, {player_y})\n"
                        f"Floor: {floor}\n"
                        f"Crosshair in crop: ({entry.crosshair_x}, {entry.crosshair_y})"
                    )
                else:
                    self.gen_status_label.setText(f"✗ Failed to save entry")
                    self.gen_status_label.setStyleSheet("color: red; font-size: 10px; font-weight: bold;")

                    QMessageBox.critical(
                        self,
                        "Error",
                        "Failed to save the dataset entry."
                    )
            else:
                self.gen_status_label.setText(f"✗ Failed to generate entry")
                self.gen_status_label.setStyleSheet("color: red; font-size: 10px; font-weight: bold;")

                QMessageBox.warning(
                    self,
                    "Generation Failed",
                    f"Could not generate entry at position ({player_x}, {player_y}) on floor {floor}.\n\n"
                    f"Possible reasons:\n"
                    f"• The position is NOT WALKABLE (water, lava, wall, etc.)\n"
                    f"• This position already exists in the dataset (duplicate)\n"
                    f"• The position is out of bounds for a full minimap crop\n"
                    f"• The floor image may not exist"
                )

        except Exception as e:
            self.gen_status_label.setText(f"✗ Error: {str(e)}")
            self.gen_status_label.setStyleSheet("color: red; font-size: 10px; font-weight: bold;")

            QMessageBox.critical(
                self,
                "Error",
                f"Failed to generate entry:\n\n{str(e)}"
            )

        finally:
            self.generate_btn.setEnabled(True)

    def generate_entries_from_map(self):
        """Auto-generate multiple dataset entries from random walkable positions."""
        from src.utils.minimap_dataset_generator import MinimapDatasetGenerator

        floor = self.gen_floor_spin.value()
        count = self.gen_count_spin.value()

        # Update status
        self.auto_gen_status_label.setText(f"Generating {count} entries for floor {floor}...")
        self.auto_generate_btn.setEnabled(False)

        # Progress callback to update UI
        def progress_callback(current, total, message):
            self.auto_gen_status_label.setText(f"{message} ({current}/{total})")
            QApplication.processEvents()  # Allow UI to update

        try:
            # Create generator
            generator = MinimapDatasetGenerator(
                minimap_dir="processed_minimap",
                dataset_manager=self.dataset_manager,
                minimap_size=(106, 106)  # Standard Tibia minimap size
            )

            # Generate entries with progress callback
            saved_count = generator.generate_and_save_entries(
                floor, count, sample_rate=2, progress_callback=progress_callback
            )

            # Clear caches to free memory
            generator.clear_caches()

            # Update status
            if saved_count == count:
                self.auto_gen_status_label.setText(
                    f"✓ Successfully generated {saved_count} entries for floor {floor}!"
                )
                self.auto_gen_status_label.setStyleSheet("color: green; font-size: 10px; font-weight: bold;")

                QMessageBox.information(
                    self,
                    "Success",
                    f"Successfully generated {saved_count} dataset entries for floor {floor}!"
                )
            else:
                self.auto_gen_status_label.setText(
                    f"⚠ Generated {saved_count}/{count} entries for floor {floor}"
                )
                self.auto_gen_status_label.setStyleSheet("color: orange; font-size: 10px; font-weight: bold;")

                QMessageBox.warning(
                    self,
                    "Partial Success",
                    f"Generated {saved_count} out of {count} requested entries.\n\n"
                    f"Some positions may not have been walkable or valid."
                )

        except Exception as e:
            self.auto_gen_status_label.setText(f"✗ Error: {str(e)}")
            self.auto_gen_status_label.setStyleSheet("color: red; font-size: 10px; font-weight: bold;")

            QMessageBox.critical(
                self,
                "Error",
                f"Failed to generate entries:\n\n{str(e)}"
            )

        finally:
            self.auto_generate_btn.setEnabled(True)

    def on_crosshair_placed(self, scene_pos):
        """Handle crosshair placement to check if position is walkable."""
        from pathlib import Path
        from PIL import Image

        # Get current floor
        if self.get_floor_callback is None:
            self.walkable_status_label.setText("Status: Floor callback not available")
            self.walkable_status_label.setStyleSheet("font-size: 12px; font-weight: bold; padding: 8px; background-color: #ffeb3b; border-radius: 4px; color: black;")
            return

        floor = self.get_floor_callback()
        x = int(scene_pos.x())
        y = int(scene_pos.y())

        # Update position label
        self.position_label.setText(f"Position: ({x}, {y}) on Floor {floor}")

        # Load floor image
        minimap_dir = Path("processed_minimap")
        floor_file = minimap_dir / f"floor_{floor:02d}.png"

        if not floor_file.exists():
            self.walkable_status_label.setText("Status: Floor image not found")
            self.walkable_status_label.setStyleSheet("font-size: 12px; font-weight: bold; padding: 8px; background-color: #ff9800; border-radius: 4px; color: white;")
            self.color_info_label.setText("Color: -")
            return

        try:
            # Load image
            image = Image.open(floor_file)
            if image.mode != 'RGB' and image.mode != 'RGBA':
                image = image.convert('RGBA')

            # Check if position is within bounds
            if x < 0 or y < 0 or x >= image.width or y >= image.height:
                self.walkable_status_label.setText("Status: OUT OF BOUNDS")
                self.walkable_status_label.setStyleSheet("font-size: 12px; font-weight: bold; padding: 8px; background-color: #9e9e9e; border-radius: 4px; color: white;")
                self.color_info_label.setText("Color: -")
                return

            # Get pixel color
            pixel = image.getpixel((x, y))

            # Handle different image modes
            if isinstance(pixel, int):
                rgb = (pixel, pixel, pixel)
            elif len(pixel) == 3:
                rgb = pixel
            elif len(pixel) == 4:
                r, g, b, a = pixel
                rgb = (r, g, b)
                if a < 128:
                    self.walkable_status_label.setText("Status: ❌ NOT WALKABLE (Transparent)")
                    self.walkable_status_label.setStyleSheet("font-size: 12px; font-weight: bold; padding: 8px; background-color: #f44336; border-radius: 4px; color: white;")
                    self.color_info_label.setText(f"Color: RGBA{pixel}")
                    return
            else:
                rgb = (0, 0, 0)

            # Check if walkable (pass floor for floor-specific rules)
            is_walkable = self.walkable_detector.is_color_walkable(rgb, floor)

            # Update UI
            self.color_info_label.setText(f"Color: RGB{rgb}")

            if is_walkable:
                self.walkable_status_label.setText("Status: ✓ WALKABLE")
                self.walkable_status_label.setStyleSheet("font-size: 12px; font-weight: bold; padding: 8px; background-color: #4CAF50; border-radius: 4px; color: white;")
            else:
                self.walkable_status_label.setText("Status: ❌ NOT WALKABLE")
                self.walkable_status_label.setStyleSheet("font-size: 12px; font-weight: bold; padding: 8px; background-color: #f44336; border-radius: 4px; color: white;")

        except Exception as e:
            self.walkable_status_label.setText(f"Status: Error - {str(e)}")
            self.walkable_status_label.setStyleSheet("font-size: 12px; font-weight: bold; padding: 8px; background-color: #ff9800; border-radius: 4px; color: white;")
            self.color_info_label.setText("Color: -")

    def pick_color(self):
        """Open color picker dialog to select a non-walkable color."""
        color = QColorDialog.getColor()

        if color.isValid():
            rgb = (color.red(), color.green(), color.blue())
            self.walkable_detector.add_non_walkable_color(rgb)
            self.refresh_color_list()

            QMessageBox.information(
                self,
                "Color Added",
                f"Added RGB{rgb} to non-walkable colors list."
            )

    def add_rgb_color(self):
        """Add a color from manual RGB input."""
        rgb_text = self.rgb_input.text().strip()

        if not rgb_text:
            QMessageBox.warning(self, "Invalid Input", "Please enter RGB values (e.g., 255,0,0)")
            return

        try:
            # Parse RGB values
            parts = [p.strip() for p in rgb_text.split(',')]
            if len(parts) != 3:
                raise ValueError("Must provide exactly 3 values (R,G,B)")

            r, g, b = int(parts[0]), int(parts[1]), int(parts[2])

            # Validate range
            if not (0 <= r <= 255 and 0 <= g <= 255 and 0 <= b <= 255):
                raise ValueError("RGB values must be between 0 and 255")

            rgb = (r, g, b)
            self.walkable_detector.add_non_walkable_color(rgb)
            self.refresh_color_list()
            self.rgb_input.clear()

            QMessageBox.information(
                self,
                "Color Added",
                f"Added RGB{rgb} to non-walkable colors list."
            )

        except ValueError as e:
            QMessageBox.warning(
                self,
                "Invalid Input",
                f"Invalid RGB format: {str(e)}\n\nPlease use format: R,G,B (e.g., 255,0,0)"
            )

    def remove_selected_color(self):
        """Remove the selected color from the non-walkable list."""
        current_item = self.color_list.currentItem()

        if not current_item:
            QMessageBox.warning(self, "No Selection", "Please select a color to remove.")
            return

        # Get RGB data stored in the item
        rgb = current_item.data(Qt.ItemDataRole.UserRole)

        if not rgb:
            QMessageBox.warning(self, "Error", "Could not retrieve color data.")
            return

        try:
            self.walkable_detector.remove_non_walkable_color(rgb)
            self.refresh_color_list()

            QMessageBox.information(
                self,
                "Color Removed",
                f"Removed RGB{rgb} from non-walkable colors list."
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to remove color: {str(e)}"
            )

    def refresh_color_list(self):
        """Refresh the display of non-walkable colors."""
        self.color_list.clear()

        colors = sorted(self.walkable_detector.get_non_walkable_colors())

        for rgb in colors:
            r, g, b = rgb

            # Create list item
            item = QListWidgetItem()

            # Create custom widget with label and color preview
            widget = QWidget()
            layout = QHBoxLayout(widget)
            layout.setContentsMargins(5, 2, 5, 2)

            # RGB text label
            label = QLabel(f"RGB({r}, {g}, {b})")
            label.setStyleSheet("font-size: 11px;")
            layout.addWidget(label)

            # Add stretch to push color preview to the right
            layout.addStretch()

            # Color preview square
            color_preview = QLabel()
            color_preview.setFixedSize(40, 20)
            color_preview.setStyleSheet(f"""
                background-color: rgb({r}, {g}, {b});
                border: 1px solid #888;
                border-radius: 3px;
            """)
            layout.addWidget(color_preview)

            # Set the custom widget for this item
            item.setSizeHint(widget.sizeHint())
            self.color_list.addItem(item)
            self.color_list.setItemWidget(item, widget)

            # Store RGB data in item for later retrieval
            item.setData(Qt.ItemDataRole.UserRole, rgb)


class ExivaDatasetCreator(QWidget):
    """Widget for creating Exiva spell message dataset entries."""
    
    def __init__(self, dataset_manager: DatasetManager, parent=None):
        super().__init__(parent)
        self.dataset_manager = dataset_manager
        self.current_screenshot_path: Optional[str] = None
        self.current_pixmap: Optional[QPixmap] = None
        
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Exiva Direction Reading Dataset Creator")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(title)
        
        # Instructions
        instructions = QLabel(
            "1. Load a screenshot containing Exiva spell text\n"
            "2. Fill in the correct parsed data\n"
            "3. Save the entry"
        )
        instructions.setStyleSheet("color: gray;")
        layout.addWidget(instructions)
        
        # Load screenshot button
        load_btn = QPushButton("Load Screenshot")
        load_btn.clicked.connect(self.load_screenshot)
        layout.addWidget(load_btn)
        
        # Screenshot preview (smaller)
        preview_group = QGroupBox("Screenshot Preview")
        preview_layout = QVBoxLayout()
        self.preview_label = QLabel("No screenshot loaded")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setMinimumHeight(200)
        self.preview_label.setMaximumHeight(300)
        self.preview_label.setStyleSheet("border: 1px solid gray;")
        self.preview_label.setScaledContents(False)
        preview_layout.addWidget(self.preview_label)
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        # Form for Exiva data
        self.setup_exiva_form(layout)
        
        # Action buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        save_btn = QPushButton("Save Entry")
        save_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        save_btn.clicked.connect(self.save_entry)
        button_layout.addWidget(save_btn)
        
        layout.addLayout(button_layout)

    def setup_exiva_form(self, parent_layout):
        """Set up the form for Exiva data input."""
        form_group = QGroupBox("Exiva Spell Data")
        form_layout = QFormLayout()

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
        for floor_ind in FloorIndication:
            self.floor_combo.addItem(floor_ind.value, floor_ind)
        form_layout.addRow("Floor Indication:", self.floor_combo)

        # Raw text (optional)
        self.raw_text_edit = QLineEdit()
        self.raw_text_edit.setPlaceholderText("Optional: actual Exiva text from screenshot")
        form_layout.addRow("Raw Text:", self.raw_text_edit)

        # Notes
        self.notes_edit = QTextEdit()
        self.notes_edit.setMaximumHeight(60)
        self.notes_edit.setPlaceholderText("Optional notes...")
        form_layout.addRow("Notes:", self.notes_edit)

        form_group.setLayout(form_layout)
        parent_layout.addWidget(form_group)

    def load_screenshot(self):
        """Load a screenshot image."""
        # Default to Tibia screenshots directory
        default_dir = r"C:\Users\bulaw\AppData\Local\Tibia\packages\Tibia\screenshots"
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Screenshot",
            default_dir,
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if file_path:
            pixmap = QPixmap(file_path)
            if pixmap.isNull():
                QMessageBox.warning(self, "Error", "Failed to load image.")
                return

            self.current_screenshot_path = file_path
            self.current_pixmap = pixmap

            # Display scaled preview
            scaled_pixmap = pixmap.scaled(
                self.preview_label.width() - 10,
                self.preview_label.height() - 10,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.preview_label.setPixmap(scaled_pixmap)

    def save_entry(self):
        """Save the current entry to the dataset."""
        # Validate inputs
        if not self.current_screenshot_path:
            QMessageBox.warning(self, "Error", "Please load a screenshot first.")
            return

        # Load global regions to get exiva region
        regions = self.dataset_manager.load_global_regions()
        exiva_region = regions.get('exiva_region')

        if not exiva_region:
            QMessageBox.warning(
                self,
                "Error",
                "Exiva region not configured. Please set the global exiva region first."
            )
            return

        # Generate entry ID
        entry_id = str(uuid.uuid4())

        # Copy screenshot to dataset with cropping to exiva region
        screenshot_rel_path = self.dataset_manager.copy_screenshot_to_dataset(
            self.current_screenshot_path, entry_id, crop_region=exiva_region
        )

        if not screenshot_rel_path:
            QMessageBox.critical(self, "Error", "Failed to copy screenshot to dataset.")
            return

        # Get the cropped image dimensions
        x, y, width, height = exiva_region

        # Create entry
        entry = ExivaDatasetEntry(
            entry_id=entry_id,
            screenshot_path=screenshot_rel_path,
            range=self.range_combo.currentData().value,
            direction=self.direction_combo.currentData().value,
            difficulty=self.difficulty_combo.currentData().value,
            floor_indication=self.floor_combo.currentData().value,
            image_width=width,  # Use cropped width
            image_height=height,  # Use cropped height
            notes=self.notes_edit.toPlainText(),
            raw_text=self.raw_text_edit.text(),
            created_timestamp=time.time(),
            modified_timestamp=time.time()
        )

        # Save to dataset
        if self.dataset_manager.add_exiva_entry(entry):
            QMessageBox.information(
                self,
                "Success",
                f"Entry saved successfully!\nEntry ID: {entry_id[:8]}...\nCropped to exiva region: {width}x{height}"
            )

            # Clear for next entry
            self.notes_edit.clear()
            self.raw_text_edit.clear()
        else:
            QMessageBox.critical(self, "Error", "Failed to save entry to dataset.")


class DatasetCreatorWindow(QMainWindow):
    """Main window for dataset creation tools."""

    def __init__(self, dataset_type: str = "minimap", parent=None):
        super().__init__(parent)
        self.dataset_manager = DatasetManager()
        self.dataset_type = dataset_type

        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        if self.dataset_type == "minimap":
            self.setWindowTitle("Minimap Dataset Creator - FiendishFinder")
            creator = MinimapDatasetCreatorPanel(self.dataset_manager, None, None, None, self)
        else:
            self.setWindowTitle("Exiva Dataset Creator - FiendishFinder")
            creator = ExivaDatasetCreator(self.dataset_manager)

        self.setCentralWidget(creator)
        self.setGeometry(100, 100, 900, 700)

        # Status bar
        self.statusBar().showMessage("Ready")


def main():
    """Main function for testing the dataset creator UI."""
    import sys

    app = QApplication(sys.argv)

    # Test crosshair creator
    window = DatasetCreatorWindow("crosshair")
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

