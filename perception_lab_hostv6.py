#!/usr/bin/env python3
"""
Antti's Perception Laboratory - Host v6 (Max Compatibility & Bulletproof)
-----------------------------------------------------------------
[V6 CORE] Maximized compatibility with old nodes using the 'import __main__' structure.
[V6 CORE] Fixes the 'AttributeError: NoneType object' crash in the UI layer.
[V5 CORE] Robust NaN/Inf handling throughout the system.
[V5.1 ENHANCEMENT] Integrated v4 visual elements (R/+/- buttons, resize handle).
"""

import sys
import numpy as np
from PyQt6 import QtWidgets, QtGui, QtCore
import cv2
import pyqtgraph as pg
from collections import deque
import os
import inspect
import importlib.util
import json

pg.setConfigOptions(imageAxisOrder='row-major')

# --- HELPER FUNCTION WITH NaN PROTECTION ---
def numpy_to_qimage(array):
    """
    Converts a numpy array (H, W, C) or (H, W) to a QImage
    for high-quality scaling. Now with NaN/Inf protection.
    """
    if array is None:
        return QtGui.QImage()
    
    # [v5 FIX] Handle NaN/Inf in display arrays
    if not np.all(np.isfinite(array)):
        array = np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Ensure array is in 0-1 float range first
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    if array.max() > 1.0:
        # Assuming it's 0-255 uint8, normalize
        array = array / 255.0
    
    array = np.clip(array, 0, 1) * 255
    array = array.astype(np.uint8)
    
    # Make array contiguous in memory
    array = np.ascontiguousarray(array)
    
    if array.ndim == 2: # Grayscale
        h, w = array.shape
        qimage = QtGui.QImage(array.data, w, h, w, QtGui.QImage.Format.Format_Grayscale8)
    elif array.ndim == 3 and array.shape[2] == 3: # RGB
        h, w, c = array.shape
        # Create QImage from 24-bit RGB data
        qimage = QtGui.QImage(array.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888)
    elif array.ndim == 3 and array.shape[2] == 4: # RGBA
        h, w, c = array.shape
        # Create QImage from 32-bit RGBA data
        qimage = QtGui.QImage(array.data, w, h, 4 * w, QtGui.QImage.Format.Format_RGBA8888)
    else:
        return QtGui.QImage() # Unsupported dimensions
        
    # QImage is a view on the numpy array. We must make a copy
    # or the array will be garbage-collected and QImage will crash.
    return qimage.copy()
# --- END HELPER FUNCTION ---


# --- Global PyAudio Instance ---
try:
    import pyaudio
    PA_INSTANCE = pyaudio.PyAudio()
except ImportError:
    print("Warning: pyaudio not installed. Audio nodes will be non-functional.")
    pyaudio = None
    PA_INSTANCE = None

# ==================== BASE NODE SYSTEM ====================

class BaseNode:
    """Base class for all perception nodes"""
    NODE_CATEGORY = "Base"
    NODE_COLOR = QtGui.QColor(80, 80, 80)
    
    def __init__(self):
        self.inputs = {}
        self.outputs = {}
        self.input_data = {}
        self.node_title = "Base Node"
        
    def pre_step(self):
        self.input_data = {name: [] for name in self.inputs}
        
    def set_input(self, port_name, value, port_type='signal', coupling=1.0):
        """[v5 FIX] Added NaN protection in signal input processing"""
        if port_name not in self.input_data:
            return
        if port_type == 'signal':
            if isinstance(value, (np.ndarray, list)):
                value = value[0] if len(value) > 0 else 0.0
            
            # [v5 FIX] Protect against NaN/Inf in signals
            try:
                fval = float(value)
                if not np.isfinite(fval):
                    fval = 0.0
            except (ValueError, TypeError):
                fval = 0.0
                
            self.input_data[port_name].append(fval * coupling)
        else:
            if value is not None:
                self.input_data[port_name].append(value)
                
    def get_blended_input(self, port_name, blend_mode='sum'):
        """[v5 FIX] Added NaN protection in blending operations"""
        values = self.input_data.get(port_name, [])
        if not values:
            return None
            
        if blend_mode == 'sum' and isinstance(values[0], (int, float)):
            result = np.sum(values)
            # [v5 FIX] Safety check
            if not np.isfinite(result):
                return 0.0
            return result
        elif blend_mode == 'mean' and isinstance(values[0], np.ndarray):
            if len(values) > 0:
                # [v5 FIX] Filter out invalid arrays before mean
                valid_arrays = [v.astype(float) for v in values if v is not None and v.size > 0]
                if not valid_arrays:
                    return None
                result = np.mean(valid_arrays, axis=0)
                # [v5 FIX] Clean up result
                if not np.all(np.isfinite(result)):
                    result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
                return result
            return None
        return values[0]
        
    def step(self):
        pass
        
    def get_output(self, port_name):
        return None
        
    def get_display_image(self):
        # Base implementation returns None.
        # Subclasses can return a np.ndarray (float 0-1) or a QImage
        return None
        
    def close(self):
        pass

    def get_config_options(self):
        return []

# ==================== NODE LOADING SYSTEM ====================

def load_nodes_from_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"Nodes folder not found, creating: {folder_path}")
        try:
            os.makedirs(folder_path)
        except Exception as e:
            print(f"Could not create nodes folder: {e}")
            return {}
            
    found_nodes = {}
    
    # Inject BaseNode and PA_INSTANCE into the __main__ module for compatibility
    # This is the key change for backward compatibility with old nodes' imports
    main_module = sys.modules['__main__']
    setattr(main_module, 'BaseNode', BaseNode)
    setattr(main_module, 'PA_INSTANCE', PA_INSTANCE)
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".py") and filename != "__init__.py":
            module_name = filename[:-3]
            file_path = os.path.join(folder_path, filename)
            
            try:
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                
                # V6 Compatibility: Ensure module can resolve __main__ imports
                module.__dict__['__main__'] = main_module
                sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                spec.loader.exec_module(module)
                
                module_nodes = []
                for name, cls in inspect.getmembers(module, inspect.isclass):
                    if issubclass(cls, BaseNode) and cls is not BaseNode:
                        node_key_name = cls.__name__
                        found_nodes[node_key_name] = {
                            "class": cls,
                            "module_name": module_name,
                            "category": cls.NODE_CATEGORY
                        }
                        module_nodes.append(name)
                        
                if module_nodes:
                    print(f"  > Loaded: {', '.join(module_nodes)} from {filename}")

            except Exception as e:
                print(f"Error loading {filename}: {e}")
                
    return found_nodes

print("Loading external nodes from './nodes' folder...")
NODE_TYPES = load_nodes_from_folder('nodes')

PORT_COLORS = {
    'signal': QtGui.QColor(200, 200, 200),
    'image': QtGui.QColor(100, 150, 255),
    'spectrum': QtGui.QColor(255, 150, 100),
    'complex_spectrum': QtGui.QColor(255, 100, 255),
}

# ==================== GRAPHICS ITEMS ====================

PORT_RADIUS = 7
NODE_W, NODE_H = 200, 180  # Increased default height for better text

class PortItem(QtWidgets.QGraphicsEllipseItem):
    def __init__(self, parent, name, port_type, is_output=False):
        super().__init__(-PORT_RADIUS, -PORT_RADIUS, PORT_RADIUS*2, PORT_RADIUS*2, parent)
        self.name = name
        self.port_type = port_type
        self.is_output = is_output
        self.base_color = PORT_COLORS.get(port_type, QtGui.QColor(255, 0, 0))
        self.setBrush(QtGui.QBrush(self.base_color))
        self.setZValue(3)
        self.setAcceptHoverEvents(True)
        
    def hoverEnterEvent(self, ev):
        self.setBrush(QtGui.QBrush(QtGui.QColor(255, 200, 60)))
    def hoverLeaveEvent(self, ev):
        self.setBrush(QtGui.QBrush(self.base_color))

class EdgeItem(QtWidgets.QGraphicsPathItem):
    def __init__(self, src_port, tgt_port=None):
        super().__init__()
        self.src = src_port
        self.tgt = tgt_port
        self.port_type = src_port.port_type
        self.setZValue(1)
        self.effect_val = 0.0
        pen = QtGui.QPen(PORT_COLORS.get(self.port_type, QtGui.QColor(200,200,200)))
        pen.setWidthF(2.0)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        self.setPen(pen)
        
    def update_path(self):
        sp = self.src.scenePos()
        tp = self.tgt.scenePos() if self.tgt else sp
        path = QtGui.QPainterPath()
        path.moveTo(sp)
        dx = (tp.x() - sp.x()) * 0.5
        c1 = QtCore.QPointF(sp.x() + dx, sp.y())
        c2 = QtCore.QPointF(tp.x() - dx, tp.y())
        path.cubicTo(c1, c2, tp)
        self.setPath(path)
        self.update_style()
        
    def update_style(self):
        """[v5 FIX] Bulletproof NaN handling prevents UI crash from bad math"""
        # Protect against NaN/Inf that would crash int() conversion
        if not np.isfinite(self.effect_val):
            self.effect_val = 0.0
            
        val = np.clip(self.effect_val, 0.0, 1.0)
        alpha = int(80 + val * 175)  # Safe now - no NaN possible
        w = 2.0 + val * 4.0
        
        color = PORT_COLORS.get(self.port_type, QtGui.QColor(200,200,200)).lighter(130)
        color.setAlpha(alpha)
        pen = QtGui.QPen(color)
        pen.setWidthF(w)
        self.setPen(pen)

class NodeItem(QtWidgets.QGraphicsItem):
    def __init__(self, sim_node, w=NODE_W, h=NODE_H):
        super().__init__()
        self.setFlags(QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
                      QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
                      QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.sim = sim_node
        self.in_ports = {}
        self.out_ports = {}
        self.min_w = NODE_W
        self.min_h = NODE_H
        self.rect = QtCore.QRectF(0, 0, w, h)
        self.display_pix = None
        
        # --- [V5.1 ENHANCEMENT] Resize and Button setup ---
        self.resize_handle_size = 15
        self.resize_handle = QtCore.QRectF(
            self.rect.width() - self.resize_handle_size,
            self.rect.height() - self.resize_handle_size,
            self.resize_handle_size,
            self.resize_handle_size
        )
        self.is_resizing = False
        self.setAcceptHoverEvents(True)
        
        self.random_btn_rect = None
        self.zoom_in_rect = None 
        self.zoom_out_rect = None 
        
        if hasattr(self.sim, 'randomize'):
            self.random_btn_rect = QtCore.QRectF(self.rect.width() - 18, 4, 14, 14)
        if hasattr(self.sim, 'zoom_factor'):
            self.zoom_in_rect = QtCore.QRectF(self.rect.width() - 38, 4, 14, 14) 
            self.zoom_out_rect = QtCore.QRectF(self.rect.width() - 18, 4, 14, 14) 
        # --- END V5.1 ENHANCEMENT ---
        
        self.init_ports()
        self.setZValue(2)
        self.update_port_positions()
        
    def init_ports(self):
        y_in = 50
        for name, ptype in self.sim.inputs.items():
            self.in_ports[name] = PortItem(self, name, ptype, False)
            self.in_ports[name].setPos(0, y_in)
            y_in += 25
            
        y_out = 50
        for name, ptype in self.sim.outputs.items():
            self.out_ports[name] = PortItem(self, name, ptype, True)
            self.out_ports[name].setPos(self.rect.width(), y_out)
            y_out += 25

    def update_port_positions(self):
        for name, port in self.out_ports.items():
            port.setPos(self.rect.width(), port.y())
            
        # --- [V5.1 ENHANCEMENT] Update button/handle positions ---
        if self.random_btn_rect:
            self.random_btn_rect.moveTopRight(self.rect.topRight() + QtCore.QPointF(-4, 4))
        if self.zoom_in_rect and self.zoom_out_rect:
            self.zoom_in_rect.moveTopRight(self.rect.topRight() + QtCore.QPointF(-24, 4))
            self.zoom_out_rect.moveTopRight(self.rect.topRight() + QtCore.QPointF(-4, 4))
            
        self.resize_handle.moveBottomRight(self.rect.bottomRight())
        # --- END V5.1 ENHANCEMENT ---
        
        if self.scene():
            for edge in self.scene().edges:
                if (edge.src.parentItem() == self) or (edge.tgt.parentItem() == self):
                    edge.update_path()

    def hoverMoveEvent(self, ev):
        if self.resize_handle.contains(ev.pos()):
            self.setCursor(QtCore.Qt.CursorShape.SizeFDiagCursor)
        else:
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        super().hoverMoveEvent(ev)

    def hoverLeaveEvent(self, ev):
        self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        super().hoverLeaveEvent(ev)

    def mousePressEvent(self, ev):
        # --- [V5.1 ENHANCEMENT] Handle resizing via handle ---
        if self.resize_handle.contains(ev.pos()) and ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self.is_resizing = True
            self.resize_start_pos = ev.pos()
            self.resize_start_rect = QtCore.QRectF(self.rect)
            ev.accept()
            return

        # --- [V5.1 ENHANCEMENT] Handle R/Zoom buttons ---
        if self.random_btn_rect and self.random_btn_rect.contains(ev.pos()):
            if hasattr(self.sim, 'randomize'):
                self.sim.randomize()
                self.update_display()
            ev.accept()
            return
        
        if self.zoom_in_rect and self.zoom_in_rect.contains(ev.pos()):
            if hasattr(self.sim, 'zoom_factor'):
                # Zoom In (Smaller factor)
                self.sim.zoom_factor = max(0.1, self.sim.zoom_factor / 1.2) 
                self.update_display()
            ev.accept()
            return
        if self.zoom_out_rect and self.zoom_out_rect.contains(ev.pos()):
            if hasattr(self.sim, 'zoom_factor'):
                # Zoom Out (Larger factor)
                self.sim.zoom_factor = min(5.0, self.sim.zoom_factor * 1.2) 
                self.update_display()
            ev.accept()
            return
        
        super().mousePressEvent(ev)
        
    def mouseMoveEvent(self, ev):
        # --- [V5.1 ENHANCEMENT] Handle resizing via handle ---
        if self.is_resizing:
            delta = ev.pos() - self.resize_start_pos
            new_w = max(self.min_w, self.resize_start_rect.width() + delta.x())
            new_h = max(self.min_h, self.resize_start_rect.height() + delta.y())
            
            self.prepareGeometryChange()
            self.rect.setWidth(new_w)
            self.rect.setHeight(new_h)
            self.update_port_positions()
            
            # Allow nodes to track their own size
            if hasattr(self.sim, 'w') and hasattr(self.sim, 'h'):
                 self.sim.w = int(new_w)
                 self.sim.h = int(new_h)
            
            ev.accept()
            return

        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self.is_resizing:
            self.is_resizing = False
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            ev.accept()
            return
        super().mouseReleaseEvent(ev)
        
    def boundingRect(self):
        return self.rect.adjusted(-8, -8, 8, 8)
        
    def paint(self, painter, option, widget):
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing) # Better text
        
        base_color = self.sim.NODE_COLOR
        if self.isSelected():
            base_color = base_color.lighter(150)
        
        painter.setBrush(QtGui.QBrush(base_color))
        painter.setPen(QtGui.QPen(QtGui.QColor(60, 60, 60), 2))
        painter.drawRoundedRect(self.rect, 10, 10)
        
        # --- [V5.1 ENHANCEMENT] Better title/category rendering ---
        title_rect = QtCore.QRectF(8, 6, self.rect.width() - 24, 22)
        painter.setPen(QtGui.QColor(255, 255, 255))
        font = QtGui.QFont("Segoe UI", 11, QtGui.QFont.Weight.Bold)
        font.setHintingPreference(QtGui.QFont.HintingPreference.PreferFullHinting)
        painter.setFont(font)
        painter.drawText(title_rect, QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter, 
                        self.sim.node_title)
        
        category_rect = QtCore.QRectF(8, 26, self.rect.width() - 16, 16)
        painter.setPen(QtGui.QColor(200, 200, 200))
        category_font = QtGui.QFont("Segoe UI", 8)
        category_font.setHintingPreference(QtGui.QFont.HintingPreference.PreferFullHinting)
        painter.setFont(category_font)
        painter.drawText(category_rect, QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
                        self.sim.NODE_CATEGORY)
        
        # Better port labels
        port_font = QtGui.QFont("Segoe UI", 8)
        port_font.setHintingPreference(QtGui.QFont.HintingPreference.PreferFullHinting)
        painter.setFont(port_font)
        painter.setPen(QtGui.QColor(220, 220, 220))
        
        for name, port in self.in_ports.items():
            painter.drawText(port.pos() + QtCore.QPointF(12, 4), name)
        
        for name, port in self.out_ports.items():
            w_text = painter.fontMetrics().boundingRect(name).width()
            painter.drawText(port.pos() + QtCore.QPointF(-w_text - 12, 4), name)
        # --- END V5.1 ENHANCEMENT ---
        
        if self.display_pix:
            img_h = self.rect.height() - 60
            img_w = self.rect.width() - 16
            
            if img_h >= 10 and img_w >= 10:
                target = QtCore.QRectF(8, 48, img_w, img_h)
                scaled = self.display_pix.scaled(
                    int(img_w), int(img_h), 
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation
                )
                x = 8 + (img_w - scaled.width()) / 2
                y = 48 + (img_h - scaled.height()) / 2
                painter.drawPixmap(
                    QtCore.QRectF(x, y, scaled.width(), scaled.height()),
                    scaled, 
                    QtCore.QRectF(scaled.rect())
                )
        
        # --- [V5.1 ENHANCEMENT] Draw R/Zoom buttons and resize handle ---
        if self.random_btn_rect:
            painter.setBrush(QtGui.QColor(255, 200, 60))
            painter.setPen(QtGui.QColor(40, 40, 40))
            painter.drawEllipse(self.random_btn_rect)
            painter.setPen(QtGui.QColor(40, 40, 40))
            painter.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Weight.Bold))
            painter.drawText(self.random_btn_rect, QtCore.Qt.AlignmentFlag.AlignCenter, "R")
            
        if self.zoom_in_rect and self.zoom_out_rect:
            painter.setBrush(QtGui.QColor(60, 180, 255))
            painter.setPen(QtGui.QColor(40, 40, 40))
            painter.drawEllipse(self.zoom_in_rect)
            painter.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Weight.Bold))
            painter.drawText(self.zoom_in_rect, QtCore.Qt.AlignmentFlag.AlignCenter, "-") # Zoom in
            
            painter.drawEllipse(self.zoom_out_rect)
            painter.drawText(self.zoom_out_rect, QtCore.Qt.AlignmentFlag.AlignCenter, "+") # Zoom out
            
        # Draw resize handle
        p = self.rect.bottomRight()
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 80), 1))
        painter.drawLine(int(p.x() - 12), int(p.y() - 4), int(p.x() - 4), int(p.y() - 12))
        painter.drawLine(int(p.x() - 8), int(p.y() - 4), int(p.x() - 4), int(p.y() - 8))
        # --- END V5.1 ENHANCEMENT ---

    def update_display(self):
        """[v5 FIX] Now uses numpy_to_qimage which has NaN protection"""
        data = self.sim.get_display_image()
        
        if data is None:
            self.display_pix = None
        elif isinstance(data, QtGui.QImage):
            self.display_pix = QtGui.QPixmap.fromImage(data)
        elif isinstance(data, np.ndarray):
            # Use the protected helper
            self.display_pix = QtGui.QPixmap.fromImage(numpy_to_qimage(data))
        else:
            self.display_pix = None
            
        self.update()

# ==================== CONFIG DIALOG ====================

class NodeConfigDialog(QtWidgets.QDialog):
    def __init__(self, node_item, parent=None):
        super().__init__(parent)
        self.node_item = node_item
        self.sim_node = node_item.sim
        self.setWindowTitle(f"Configure: {self.sim_node.node_title}")
        self.setMinimumWidth(400)
        
        self.widgets_map = {}
        
        layout = QtWidgets.QVBoxLayout(self)
        
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        container = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(container)
        
        for item in self.sim_node.get_config_options():
            label_text, attr_name, current_value, widget_type = item
            
            if widget_type == 'file_open':
                # --- [V5.1 ENHANCEMENT] File browser logic ---
                h_layout = QtWidgets.QHBoxLayout()
                line = QtWidgets.QLineEdit(str(current_value))
                browse_btn = QtWidgets.QPushButton("Browse...")
                
                # Connect the browse button to a helper function
                browse_btn.clicked.connect(lambda _, k=attr_name, w=line: self.open_file_dialog(k, w))
                
                h_layout.addWidget(line)
                h_layout.addWidget(browse_btn)
                
                widget = QtWidgets.QWidget()
                widget.setLayout(h_layout)
                form.addRow(label_text + ":", widget)
                self.widgets_map[attr_name] = ('text', line, None) 
                # --- END V5.1 ENHANCEMENT ---
                
            elif isinstance(widget_type, list):
                combo = QtWidgets.QComboBox()
                
                # V6 Compatibility: Handle tuple/list options (name, value) or simple values
                for option in widget_type:
                    if isinstance(option, (list, tuple)) and len(option) == 2:
                        name, value = option
                        combo.addItem(str(name), userData=value)
                    else:
                        name = str(option)
                        value = option
                        combo.addItem(name, userData=value)
                
                try:
                    # Try to find index by current value data
                    current_data_index = combo.findData(current_value)
                    if current_data_index != -1:
                        combo.setCurrentIndex(current_data_index)
                    else:
                         # If value wasn't explicitly defined in the list, add it if it's an ID/number
                         combo.addItem(f"Current ID ({current_value})", userData=current_value)
                         combo.setCurrentIndex(combo.count() - 1)
                        
                except ValueError:
                    pass
                        
                form.addRow(label_text + ":", combo)
                self.widgets_map[attr_name] = ('combo', combo, widget_type)
                
            elif widget_type == 'int':
                spin = QtWidgets.QSpinBox()
                spin.setRange(-999999, 999999)
                spin.setValue(int(current_value))
                form.addRow(label_text + ":", spin)
                self.widgets_map[attr_name] = ('int', spin, None)
                
            elif widget_type == 'float':
                spin = QtWidgets.QDoubleSpinBox()
                spin.setRange(-999999.0, 999999.0)
                spin.setDecimals(4)
                spin.setValue(float(current_value))
                form.addRow(label_text + ":", spin)
                self.widgets_map[attr_name] = ('float', spin, None)
                
            elif widget_type == 'bool':
                check = QtWidgets.QCheckBox()
                check.setChecked(bool(current_value))
                form.addRow(label_text + ":", check)
                self.widgets_map[attr_name] = ('bool', check, None)
                
            elif widget_type == 'text_multi':
                text_edit = QtWidgets.QTextEdit()
                text_edit.setPlainText(str(current_value))
                text_edit.setMaximumHeight(100)
                form.addRow(label_text + ":", text_edit)
                self.widgets_map[attr_name] = ('text_multi', text_edit, None)
            
            elif widget_type == 'text':
                line = QtWidgets.QLineEdit(str(current_value))
                form.addRow(label_text + ":", line)
                self.widgets_map[attr_name] = ('text', line, None)

            else:
                line = QtWidgets.QLineEdit(str(current_value))
                form.addRow(label_text + ":", line)
                self.widgets_map[attr_name] = ('text', line, None)
        
        scroll.setWidget(container)
        layout.addWidget(scroll)
        
        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | 
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)
        
    # --- [V5.1 ENHANCEMENT] Helper method for file dialog ---
    def open_file_dialog(self, key, line_edit_widget):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select File", "", "All Files (*)")
        
        if fileName:
            line_edit_widget.setText(fileName)
            # Immediately update node's config and trigger potential reload
            setattr(self.sim_node, key, fileName)
            if hasattr(self.sim_node, '_load_image'):
                 self.sim_node._load_image()
            if hasattr(self.sim_node, 'setup_source'):
                 self.sim_node.setup_source()
            if hasattr(self.sim_node, 'update'):
                 self.sim_node.update()
    # --- END V5.1 ENHANCEMENT ---
    
    def get_config_dict(self):
        config = {}
        for attr_name, (w_type, widget, options) in self.widgets_map.items():
            if w_type == 'combo':
                # Grab the data stored in the combo box item
                config[attr_name] = widget.currentData()
            elif w_type == 'int':
                config[attr_name] = widget.value()
            elif w_type == 'float':
                config[attr_name] = widget.value()
            elif w_type == 'bool':
                config[attr_name] = widget.isChecked()
            elif w_type == 'text' or w_type == 'text_multi':
                text = widget.text() if w_type == 'text' else widget.toPlainText()
                try:
                    # Attempt to convert to int/float
                    fval = float(text)
                    if fval.is_integer():
                        config[attr_name] = int(fval)
                    else:
                        config[attr_name] = fval
                except ValueError:
                    config[attr_name] = text
        return config

# ==================== MAIN SCENE ====================

class PerceptionScene(QtWidgets.QGraphicsScene):
    def __init__(self):
        super().__init__()
        self.setBackgroundBrush(QtGui.QColor(30, 30, 30))
        self.nodes = []
        self.edges = []
        
        self.temp_edge = None
        self.connection_start_port = None
        
    def add_node(self, node_class, x=0, y=0, w=NODE_W, h=NODE_H):
        sim_node = node_class()
        node_item = NodeItem(sim_node, w, h)
        node_item.setPos(x, y)
        self.addItem(node_item)
        self.nodes.append(node_item)
        node_item.update_display() 
        return node_item
        
    def remove_node(self, node_item):
        if node_item not in self.nodes:
            return
            
        for edge in list(self.edges):
            if edge.src.parentItem() == node_item or edge.tgt.parentItem() == node_item:
                self.removeItem(edge)
                self.edges.remove(edge)
                
        node_item.sim.close()
        self.removeItem(node_item)
        self.nodes.remove(node_item)
        
    def remove_edge(self, edge):
        """Helper for deleting edges"""
        if edge in self.edges:
            self.removeItem(edge)
            self.edges.remove(edge)
    
    def delete_selected_edges(self):
        selected_edges = [i for i in self.selectedItems() if isinstance(i, EdgeItem)]
        for edge in selected_edges:
            self.remove_edge(edge)
    
    def mousePressEvent(self, ev):
        """
        [v6 FIX]: The button/handle interaction logic is entirely handled by
        NodeItem.mousePressEvent, preventing the crash when 'item' is None.
        """
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            super().mousePressEvent(ev)
            return
            
        item = self.itemAt(ev.scenePos(), QtGui.QTransform())
        
        if isinstance(item, PortItem):
            if self.connection_start_port is None:
                # First click - start connection from output port
                if item.is_output:
                    self.connection_start_port = item
                    self.temp_edge = EdgeItem(item)
                    self.addItem(self.temp_edge)
                    # Create animated preview
                    self.temp_edge.setPen(QtGui.QPen(
                        QtGui.QColor(255, 200, 60), 3.0, QtCore.Qt.PenStyle.DashLine
                    ))
                    ev.accept()
                    return
            else:
                # Second click - complete connection to input port
                if not item.is_output and item.parentItem() != self.connection_start_port.parentItem():
                    # Valid connection: output -> input, different nodes
                    if self.connection_start_port.port_type == item.port_type:
                        # Create permanent edge
                        edge = EdgeItem(self.connection_start_port, item)
                        self.addItem(edge)
                        self.edges.append(edge)
                        edge.update_path()
                    
                # Clean up temp edge regardless
                self.cancel_connection()
                ev.accept()
                return
        
        else:
            if self.connection_start_port is not None:
                self.cancel_connection()
            super().mousePressEvent(ev)
    
    def mouseMoveEvent(self, ev):
        """[v5 ENHANCED] Animate preview wire during connection"""
        if self.temp_edge and self.connection_start_port:
            class TempTarget:
                def __init__(self, pos): self._pos = pos
                def scenePos(self): return self._pos
            
            self.temp_edge.tgt = TempTarget(ev.scenePos())
            self.temp_edge.update_path()
            ev.accept()
            return
            
        super().mouseMoveEvent(ev)
    
    def mouseReleaseEvent(self, ev):
        super().mouseReleaseEvent(ev)
    
    def cancel_connection(self):
        """[v5] Clean up connection state"""
        if self.temp_edge:
            self.removeItem(self.temp_edge)
            self.temp_edge = None
        self.connection_start_port = None

# ==================== MAIN WINDOW ====================

class PerceptionLab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Antti's Perception Laboratory v6 - Max Compatibility Edition 🛡️")
        self.resize(1400, 800)
        
        self.scene = PerceptionScene()
        self.view = QtWidgets.QGraphicsView(self.scene)
        self.view.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.view.setDragMode(QtWidgets.QGraphicsView.DragMode.RubberBandDrag)
        
        self.NODE_CLASS_MAP = {name: info for name, info in NODE_TYPES.items()}
        
        self.is_running = False
        self.coupling_strength = 0.7
        
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Top toolbar
        toolbar = QtWidgets.QWidget()
        toolbar.setStyleSheet("background: #2a2a2a; padding: 6px;")
        toolbar_layout = QtWidgets.QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(8, 4, 8, 4)
        
        # Add Node button with menu
        self.btn_add = QtWidgets.QPushButton("➕ Add Node")
        self.btn_add.setStyleSheet("padding: 6px 12px; font-weight: bold; background: #3a5a8a; color: white; border-radius: 5px;")
        self.btn_add.clicked.connect(self.show_add_node_menu)
        toolbar_layout.addWidget(self.btn_add)
        
        toolbar_layout.addSpacing(10)
        
        # Run/Stop button
        self.btn_run = QtWidgets.QPushButton("▶ Start")
        self.btn_run.setStyleSheet("padding: 6px 12px; font-weight: bold; background: #2a5a2a; color: white; border-radius: 5px;")
        self.btn_run.clicked.connect(self.toggle_run)
        toolbar_layout.addWidget(self.btn_run)
        
        toolbar_layout.addSpacing(20)
        
        # Coupling slider
        toolbar_layout.addWidget(QtWidgets.QLabel("Coupling:"))
        self.coupling_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.coupling_slider.setRange(0, 100)
        self.coupling_slider.setValue(70)
        self.coupling_slider.setMaximumWidth(150)
        self.coupling_slider.valueChanged.connect(self.update_coupling)
        toolbar_layout.addWidget(self.coupling_slider)
        
        self.coupling_label = QtWidgets.QLabel("0.70")
        self.coupling_label.setMinimumWidth(40)
        toolbar_layout.addWidget(self.coupling_label)
        
        toolbar_layout.addStretch()
        
        # File operations
        btn_clear = QtWidgets.QPushButton("🗑 Clear All")
        btn_clear.setStyleSheet("padding: 6px 12px; background: #dc2626; color: white; border-radius: 5px;")
        btn_clear.clicked.connect(self.clear_graph)
        toolbar_layout.addWidget(btn_clear)
        
        btn_save = QtWidgets.QPushButton("💾 Save")
        btn_save.setStyleSheet("padding: 6px 12px; background: #2563eb; color: white; border-radius: 5px;")
        btn_save.clicked.connect(self.save_graph)
        toolbar_layout.addWidget(btn_save)
        
        btn_load = QtWidgets.QPushButton("📁 Load")
        btn_load.setStyleSheet("padding: 6px 12px; background: #7c3aed; color: white; border-radius: 5px;")
        btn_load.clicked.connect(self.load_graph)
        toolbar_layout.addWidget(btn_load)
        
        main_layout.addWidget(toolbar)
        main_layout.addWidget(self.view)
        
        # Status bar
        status_bar = QtWidgets.QWidget()
        status_bar.setStyleSheet("background: #2a2a2a; padding: 4px;")
        status_layout = QtWidgets.QHBoxLayout(status_bar)
        status_layout.setContentsMargins(8, 2, 8, 2)
        
        self.status = QtWidgets.QLabel("Ready - Add nodes to begin")
        self.status.setStyleSheet("color: #aaa; font-size: 10px;")
        status_layout.addWidget(self.status)
        status_layout.addStretch()
        
        version_label = QtWidgets.QLabel("v6 - Max Compatibility Edition 🛡️")
        version_label.setStyleSheet("color: #6495ed; font-size: 10px; font-weight: bold;")
        status_layout.addWidget(version_label)
        
        main_layout.addWidget(status_bar)
        
        # Context menu
        self.view.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self.show_context_menu)
        
        # Simulation timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.simulation_step)
        
    def show_add_node_menu(self):
        menu = QtWidgets.QMenu(self)
        
        categories = {}
        for name, info in self.NODE_CLASS_MAP.items():
            cat = info.get('category', 'Uncategorized')
            if cat not in categories:
                categories[cat] = []
            categories[cat].append((name, info))
        
        for cat in sorted(categories.keys()):
            submenu = menu.addMenu(cat)
            for name, info in sorted(categories[cat], key=lambda x: x[0]):
                action = submenu.addAction(name)
                action.triggered.connect(lambda checked, n=name: self.add_node(n))
        
        menu.exec(QtGui.QCursor.pos())
    
    def add_node(self, class_name):
        if class_name not in self.NODE_CLASS_MAP:
            return
            
        node_class = self.NODE_CLASS_MAP[class_name]['class']
        center = self.view.mapToScene(self.view.viewport().rect().center())
        
        # Check if node has w/h attributes to set initial size from config
        temp_node_instance = node_class()
        w = getattr(temp_node_instance, 'w', NODE_W) if hasattr(temp_node_instance, 'w') else NODE_W
        h = getattr(temp_node_instance, 'h', NODE_H) if hasattr(temp_node_instance, 'h') else NODE_H
        del temp_node_instance
        
        node = self.scene.add_node(node_class, center.x() - w/2, center.y() - h/2, w=w, h=h)
        
        if hasattr(node.sim, 'open_stream'):
            node.sim.open_stream()
        if hasattr(node.sim, 'setup_source'):
            node.sim.setup_source()
            
        self.status.setText(f"✓ Added {class_name}")
    
    def show_context_menu(self, pos):
        scene_pos = self.view.mapToScene(pos)
        item = self.scene.itemAt(scene_pos, QtGui.QTransform())
        
        selected_nodes = [i for i in self.scene.selectedItems() if isinstance(i, NodeItem)]
        clicked_node_item = None

        if isinstance(item, NodeItem):
            clicked_node_item = item
        elif isinstance(item, PortItem) and hasattr(item.parentItem(), 'sim'):
            clicked_node_item = item.parentItem()
            
        if clicked_node_item and clicked_node_item not in selected_nodes:
             self.scene.clearSelection()
             clicked_node_item.setSelected(True)
             selected_nodes = [clicked_node_item]
        
        menu = QtWidgets.QMenu(self)
        
        if selected_nodes:
            # Delete Action
            delete_action = menu.addAction(f"🗑️ Delete Selected Node{'s' if len(selected_nodes) > 1 else ''} ({len(selected_nodes)})")
            delete_action.triggered.connect(lambda: self.delete_selected_nodes())
            
            if len(selected_nodes) == 1:
                menu.addSeparator()
                # Configure Action
                config_act = menu.addAction("⚙️ Configure")
                config_act.triggered.connect(lambda: self.configure_node(selected_nodes[0]))
                
        else:
            # Add Node menu
            add_menu = self.show_add_node_menu() # Get the menu structure
            # Re-implementing the add node menu logic to be a submenu instead of recursive action
            categories = {}
            for name, info in self.NODE_CLASS_MAP.items():
                cat = info.get('category', 'Uncategorized')
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append((name, info['class']))
            
            for cat in sorted(categories.keys()):
                submenu = menu.addMenu(cat)
                for name, cls in sorted(categories[cat], key=lambda x: x[0]):
                    action = submenu.addAction(name)
                    action.triggered.connect(lambda checked, nc=cls, sp=scene_pos: self.add_node_at_pos(nc, sp))
            
            
        global_pos = self.view.mapToGlobal(pos)
        menu.exec(global_pos)

    def add_node_at_pos(self, node_class, scene_pos):
        # Check if node has w/h attributes to set initial size from config
        temp_node_instance = node_class()
        w = getattr(temp_node_instance, 'w', NODE_W) if hasattr(temp_node_instance, 'w') else NODE_W
        h = getattr(temp_node_instance, 'h', NODE_H) if hasattr(temp_node_instance, 'h') else NODE_H
        del temp_node_instance
        
        node = self.scene.add_node(node_class, scene_pos.x() - w/2, scene_pos.y() - h/2, w=w, h=h)
        
        if hasattr(node.sim, 'open_stream'):
            node.sim.open_stream()
        if hasattr(node.sim, 'setup_source'):
            node.sim.setup_source()
            
        self.status.setText(f"✓ Added {node.sim.node_title}")
        
    def delete_selected_nodes(self):
        selected_nodes = [i for i in self.scene.selectedItems() if isinstance(i, NodeItem)]
        for node in selected_nodes:
            self.scene.remove_node(node)
        self.status.setText("✓ Deleted selected nodes")
        
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_Delete or event.key() == QtCore.Qt.Key.Key_Backspace:
            self.delete_selected_nodes()
            # Also delete selected edges
            self.scene.delete_selected_edges()
            return
        super().keyPressEvent(event)
        
    def configure_node(self, node_item):
        dialog = NodeConfigDialog(node_item, self)
        if dialog.exec():
            config = dialog.get_config_dict()
            for key, val in config.items():
                setattr(node_item.sim, key, val)
            
            # Re-run setup/open if configuration changes require it
            if hasattr(node_item.sim, 'open_stream'):
                node_item.sim.open_stream()
            if hasattr(node_item.sim, 'setup_source'):
                node_item.sim.setup_source()
            
            node_item.update_display()
            node_item.update_port_positions()
            self.status.setText(f"✓ Configured {node_item.sim.node_title}")
    
    def delete_edge(self, edge):
        self.scene.removeItem(edge)
        self.scene.edges.remove(edge)
        self.status.setText("✓ Connection deleted")
    
    def clear_edges(self):
        for edge in list(self.scene.edges):
            self.scene.removeItem(edge)
        self.scene.edges.clear()
    
    def update_coupling(self, value):
        self.coupling_strength = value / 100.0
        self.coupling_label.setText(f"{self.coupling_strength:.2f}")
    
    def toggle_run(self):
        self.is_running = not self.is_running
        if self.is_running:
            self.timer.start(30)  # ~33 FPS
            self.btn_run.setText("⏸ Stop")
            self.btn_run.setStyleSheet("padding: 6px 12px; font-weight: bold; background: #5a2a2a; color: white; border-radius: 5px;")
            self.status.setText("▶ Simulation running...")
            self.last_time = QtCore.QTime.currentTime()
            self.frame_count = 0
        else:
            self.timer.stop()
            self.btn_run.setText("▶ Start")
            self.btn_run.setStyleSheet("padding: 6px 12px; font-weight: bold; background: #2a5a2a; color: white; border-radius: 5px;")
            self.status.setText("⏸ Simulation paused")
    
    def simulation_step(self):
        """[v5 FIX] Enhanced error handling and NaN protection throughout pipeline"""
        # Step 1: Prepare all nodes
        for node_item in self.scene.nodes:
            node_item.sim.pre_step()
        
        # Step 2: Transfer data through edges
        for edge in self.scene.edges:
            src_node = edge.src.parentItem().sim
            tgt_node = edge.tgt.parentItem().sim
            
            try:
                output_val = src_node.get_output(edge.src.name)
                
                # [v5 FIX] Robust effect calculation with NaN protection
                edge.effect_val = 0.0
                if output_val is not None:
                    tgt_node.set_input(
                        edge.tgt.name, 
                        output_val, 
                        edge.port_type, 
                        self.coupling_strength
                    )
                    
                    # Calculate visual effect safely
                    if isinstance(output_val, (int, float)):
                        if np.isfinite(output_val):
                            edge.effect_val = abs(float(output_val))
                    elif isinstance(output_val, np.ndarray):
                        if output_val.size > 0:
                            mean_val = np.mean(np.abs(output_val))
                            if np.isfinite(mean_val):
                                edge.effect_val = float(mean_val)
                                
                    # Set a base effect value for non-signal types
                    if edge.port_type in ['image', 'spectrum', 'complex_spectrum']:
                        edge.effect_val = max(edge.effect_val * 0.1, 0.5) 
                
                edge.update_path() # Calls update_style
                
            except Exception as e:
                # [v5 FIX] Non-fatal error handling
                print(f"Warning: Edge transfer error ({edge.src.name} -> {edge.tgt.name}): {e}")
                edge.effect_val = 0.0
                edge.update_path()
        
        # Step 3: Execute node logic
        for node_item in self.scene.nodes:
            try:
                node_item.sim.step()
                node_item.update_display()
            except Exception as e:
                # [v5 FIX] Keep simulation running even if one node fails
                print(f"Error in {node_item.sim.node_title}: {e}")
                # Optionally show error state in node
                continue
        
        # Step 4: Update FPS
        if hasattr(self, 'frame_count'):
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                current_time = QtCore.QTime.currentTime()
                elapsed = self.last_time.msecsTo(current_time) / 1000.0
                if elapsed > 0:
                    fps = 30.0 / elapsed
                    self.status.setText(f"▶ Running... | FPS: {fps:.1f}")
                self.last_time = current_time
    
    def clear_graph(self):
        reply = QtWidgets.QMessageBox.question(
            self, 
            "Clear Graph?",
            "This will delete all nodes and connections. Continue?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No
        )
        
        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            if self.is_running:
                self.toggle_run()
            self.clear_edges()
            for node in list(self.scene.nodes):
                self.scene.remove_node(node)
            self.status.setText("✓ Graph cleared - Ready for new workflow")

    def save_graph(self):
        if self.is_running:
            self.toggle_run()

        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Graph", "", "JSON Files (*.json)")
        if not fileName:
            return

        graph_data = {
            "global_settings": {
                "coupling": self.coupling_slider.value()
            },
            "nodes": [],
            "edges": []
        }

        node_to_id = {}
        for i, node_item in enumerate(self.scene.nodes):
            sim_node = node_item.sim
            node_id = i
            node_to_id[node_item] = node_id
            
            config_data = {}
            for key, value in sim_node.__dict__.items():
                if isinstance(value, (int, float, str, bool)):
                    config_data[key] = value
                elif isinstance(value, (list, dict)):
                    try: 
                        json.dumps(value)
                        config_data[key] = value
                    except TypeError:
                        pass
            
            graph_data["nodes"].append({
                "id": node_id,
                "class_name": sim_node.__class__.__name__,
                "pos_x": node_item.pos().x(),
                "pos_y": node_item.pos().y(),
                "width": node_item.rect.width(),
                "height": node_item.rect.height(),
                "config": config_data
            })

        for edge in self.scene.edges:
            src_node_item = edge.src.parentItem()
            tgt_node_item = edge.tgt.parentItem()
            
            if src_node_item in node_to_id and tgt_node_item in node_to_id:
                graph_data["edges"].append({
                    "from_node_id": node_to_id[src_node_item],
                    "from_port": edge.src.name,
                    "to_node_id": node_to_id[tgt_node_item],
                    "to_port": edge.tgt.name
                })

        try:
            with open(fileName, 'w') as f:
                json.dump(graph_data, f, indent=2)
            self.status.setText(f"✓ Graph saved to {os.path.basename(fileName)}")
        except Exception as e:
            self.status.setText(f"❌ Error saving graph: {e}")
            print(f"Error saving graph: {e}")

    def load_graph(self):
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load Graph", "", "JSON Files (*.json)")
        if not fileName:
            return
            
        try:
            with open(fileName, 'r') as f:
                graph_data = json.load(f)
        except Exception as e:
            self.status.setText(f"❌ Error loading file: {e}")
            return

        self.clear_edges()
        for node in list(self.scene.nodes):
            self.scene.remove_node(node)
        
        id_to_node_item = {}

        for node_data in graph_data.get("nodes", []):
            class_name = node_data["class_name"]
            node_info = self.NODE_CLASS_MAP.get(class_name) 
            
            if node_info:
                node_class = node_info["class"]
                pos_x = node_data.get("pos_x", 0)
                pos_y = node_data.get("pos_y", 0)
                width = node_data.get("width", NODE_W)
                height = node_data.get("height", NODE_H)
                config = node_data.get("config", {})
                
                node_item = self.scene.add_node(node_class, x=pos_x, y=pos_y, w=width, h=height)
                
                for key, value in config.items():
                    if hasattr(node_item.sim, key):
                        setattr(node_item.sim, key, value)
                
                if hasattr(node_item.sim, 'open_stream'):
                    node_item.sim.open_stream()
                if hasattr(node_item.sim, 'setup_source'):
                    node_item.sim.setup_source()
                
                node_item.update()
                node_item.update_port_positions()
                id_to_node_item[node_data["id"]] = node_item
            else:
                print(f"Warning: Node class '{class_name}' not found. Skipping.")

        for edge_data in graph_data.get("edges", []):
            try:
                src_node_item = id_to_node_item[edge_data["from_node_id"]]
                tgt_node_item = id_to_node_item[edge_data["to_node_id"]]
                src_port = src_node_item.out_ports[edge_data["from_port"]]
                tgt_port = tgt_node_item.in_ports[edge_data["to_port"]]
                
                edge = EdgeItem(src_port, tgt_port)
                self.scene.addItem(edge)
                self.scene.edges.append(edge)
                edge.update_path()
            except Exception as e:
                print(f"Error loading edge: {e}")

        settings = graph_data.get("global_settings", {})
        self.coupling_slider.setValue(settings.get("coupling", 70))

        self.status.setText(f"✓ Graph loaded from {os.path.basename(fileName)}")
        
    def closeEvent(self, event):
        self.timer.stop()
        for node in self.scene.nodes:
            node.sim.close()
        if PA_INSTANCE:
            try:
                PA_INSTANCE.terminate()
            except Exception as e:
                print(f"Error terminating PyAudio: {e}")
        super().closeEvent(event)

# ==================== APPLICATION ENTRY ====================

def main():
    app = QtWidgets.QApplication(sys.argv)
    
    app.setStyle('Fusion')
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(30, 30, 30))
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor(25, 25, 25))
    palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor(45, 45, 45))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor(45, 45, 45))
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor(220, 220, 220))
    palette.setColor(QtGui.QPalette.ColorRole.BrightText, QtGui.QColor(255, 80, 80))
    palette.setColor(QtGui.QPalette.ColorRole.Link, QtGui.QColor(100, 150, 255))
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor(100, 150, 255))
    palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor(20, 20, 20))
    app.setPalette(palette)
    
    app.setStyleSheet("""
        QWidget {
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        QPushButton {
            background: #4a4a4a;
            border: 1px solid #555;
            color: #ddd;
            border-radius: 4px;
        }
        QPushButton:hover {
            background: #5a5a5a;
        }
        QPushButton:pressed {
            background: #3a3a3a;
        }
        QMenu {
            background: #2a2a2a;
            border: 1px solid #444;
            font-size: 11px;
        }
        QMenu::item {
            padding: 6px 20px;
        }
        QMenu::item:selected {
            background: #3a5a8a;
        }
        QSlider::groove:horizontal {
            height: 4px;
            background: #3a3a3a;
            border-radius: 2px;
        }
        QSlider::handle:horizontal {
            background: #6495ed;
            width: 14px;
            margin: -5px 0;
            border-radius: 7px;
        }
        QSlider::handle:horizontal:hover {
            background: #7ab5ff;
        }
        QLineEdit, QComboBox {
            background: #3a3a3a;
            border: 1px solid #555;
            padding: 4px;
            color: #ddd;
            border-radius: 4px;
            font-size: 11px;
        }
        QTextEdit {
            background: #3a3a3a;
            border: 1px solid #555;
            padding: 4px;
            color: #ddd;
            border-radius: 4px;
            font-size: 11px;
        }
    """)
    
    window = PerceptionLab()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
