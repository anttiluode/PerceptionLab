#!/usr/bin/env python3
"""
Antti's Perception Laboratory - Main Host Application
A professional node-based interface for multi-domain perception experiments.

This is the main host that loads nodes from the 'nodes' folder.
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
        self.inputs = {}   # {'port_name': 'port_type'}
        self.outputs = {}  # {'port_name': 'port_type'}
        self.input_data = {}
        self.node_title = "Base Node"
        
    def pre_step(self):
        """Clear input buffers before propagation"""
        self.input_data = {name: [] for name in self.inputs}
        
    def set_input(self, port_name, value, port_type='signal', coupling=1.0):
        """Receive data from connected edges"""
        if port_name not in self.input_data:
            return
        if port_type == 'signal':
            if isinstance(value, (np.ndarray, list)):
                value = value[0] if len(value) > 0 else 0.0
            self.input_data[port_name].append(float(value) * coupling)
        else:
            if value is not None:
                self.input_data[port_name].append(value)
                
    def get_blended_input(self, port_name, blend_mode='sum'):
        """Get combined input from all connections"""
        values = self.input_data.get(port_name, [])
        if not values:
            return None
            
        if blend_mode == 'sum' and isinstance(values[0], (int, float)):
            return np.sum(values)
        elif blend_mode == 'mean' and isinstance(values[0], np.ndarray):
            if len(values) > 0:
                return np.mean([v.astype(float) for v in values if v is not None and v.size > 0], axis=0)
            return None
        return values[0]
        
    def step(self):
        """Override in subclass - main processing logic"""
        pass
        
    def get_output(self, port_name):
        """Override in subclass - return output data"""
        return None
        
    def get_display_image(self):
        """Override in subclass - return QImage for node preview"""
        return None
        
    def close(self):
        """Cleanup resources. Called when node is deleted."""
        pass

    def get_config_options(self):
        """Returns a list of (display_name, key, current_value, options) for configuration dialog."""
        return []

# ==================== NODE LOADING SYSTEM ====================

def load_nodes_from_folder(folder_path):
    """
    Scans a folder for .py files, imports them, and finds all subclasses
    of BaseNode to add to the registry.
    """
    if not os.path.exists(folder_path):
        print(f"Nodes folder not found, creating: {folder_path}")
        try:
            os.makedirs(folder_path)
        except Exception as e:
            print(f"Could not create nodes folder: {e}")
            return {}
            
    # Key: Class Name (string), Value: {"class": Class Object, "module_name": str}
    found_nodes = {}
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".py") and filename != "__init__.py":
            module_name = filename[:-3]
            file_path = os.path.join(folder_path, filename)
            
            try:
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                
                # Add host's globals (like BaseNode, QtGui) to the module
                module.__dict__['__main__'] = sys.modules['__main__']
                
                sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                spec.loader.exec_module(module)
                
                module_nodes = []
                for name, cls in inspect.getmembers(module, inspect.isclass):
                    # Only register classes that inherit BaseNode and are NOT BaseNode itself
                    if issubclass(cls, BaseNode) and cls is not BaseNode:
                        node_key_name = cls.__name__
                        found_nodes[node_key_name] = {
                            "class": cls,
                            "module_name": module_name,
                            "category": cls.NODE_CATEGORY # Store category for menu
                        }
                        module_nodes.append(name)
                        
                        
                if module_nodes:
                    print(f"  > Loaded external nodes: {', '.join(module_nodes)} from {filename}")

            except Exception as e:
                print(f"Error loading node from {filename}: {e}")
                
    return found_nodes

# Load external nodes ONCE at module level
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
NODE_W, NODE_H = 200, 160 # These are now MINIMUMS

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
        val = np.clip(self.effect_val, 0.0, 1.0)
        alpha = int(80 + val * 175)
        w = 2.0 + val * 4.0
        color = PORT_COLORS.get(self.port_type, QtGui.QColor(200,200,200)).lighter(130)
        color.setAlpha(alpha)
        pen = QtGui.QPen(color)
        pen.setWidthF(w)
        pen.setCapStyle(QtCore.Qt.PenCapStyle.RoundCap)
        self.setPen(pen)

class NodeItem(QtWidgets.QGraphicsItem):
    def __init__(self, sim_node):
        super().__init__()
        self.setFlags(
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QtWidgets.QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.sim = sim_node
        self.in_ports = {}
        self.out_ports = {}
        
        # --- Resizing Logic ---
        self.min_w = NODE_W
        self.min_h = NODE_H
        self.rect = QtCore.QRectF(0, 0, self.min_w, self.min_h)
        
        self.resize_handle_size = 15
        self.resize_handle = QtCore.QRectF(
            self.rect.width() - self.resize_handle_size,
            self.rect.height() - self.resize_handle_size,
            self.resize_handle_size,
            self.resize_handle_size
        )
        self.is_resizing = False
        self.setAcceptHoverEvents(True)
        # --- End Resizing Logic ---

        # Create ports
        for name, ptype in self.sim.inputs.items():
            port = PortItem(self, name, ptype, is_output=False)
            self.in_ports[name] = port
            
        for name, ptype in self.sim.outputs.items():
            port = PortItem(self, name, ptype, is_output=True)
            self.out_ports[name] = port
            
        self.setZValue(2)
        self.display_pix = None
        
        self.random_btn_rect = None
        self.zoom_in_rect = None 
        self.zoom_out_rect = None 
        
        # Check if node has special buttons
        if hasattr(self.sim, 'randomize'):
            self.random_btn_rect = QtCore.QRectF(self.rect.width() - 18, 4, 14, 14)
        if hasattr(self.sim, 'zoom_factor'):
            self.zoom_in_rect = QtCore.QRectF(self.rect.width() - 38, 4, 14, 14) 
            self.zoom_out_rect = QtCore.QRectF(self.rect.width() - 18, 4, 14, 14) 
        
        # Position ports
        self.update_port_positions()
        
    def update_port_positions(self):
        """Helper function to reposition ports and buttons when resizing."""
        y_in = 40
        for port in self.in_ports.values():
            port.setPos(0, y_in)
            y_in += 25
            
        y_out = 40
        for port in self.out_ports.values():
            port.setPos(self.rect.width(), y_out)
            y_out += 25
            
        # Update button positions
        if self.random_btn_rect:
            self.random_btn_rect.moveTopRight(self.rect.topRight() + QtCore.QPointF(-4, 4))
        if self.zoom_in_rect and self.zoom_out_rect:
            self.zoom_in_rect.moveTopRight(self.rect.topRight() + QtCore.QPointF(-24, 4))
            self.zoom_out_rect.moveTopRight(self.rect.topRight() + QtCore.QPointF(-4, 4))
            
        # Update resize handle position
        self.resize_handle.moveBottomRight(self.rect.bottomRight())
        
        # Update connected edges
        if self.scene():
            for edge in self.scene().edges:
                if (edge.src.parentItem() == self) or (edge.tgt.parentItem() == self):
                    edge.update_path()

    def hoverMoveEvent(self, ev):
        """Change cursor when hovering over resize handle."""
        if self.resize_handle.contains(ev.pos()):
            self.setCursor(QtCore.Qt.CursorShape.SizeFDiagCursor)
        else:
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        super().hoverMoveEvent(ev)

    def hoverLeaveEvent(self, ev):
        self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        super().hoverLeaveEvent(ev)

    def mousePressEvent(self, ev):
        # --- Resizing Logic ---
        if self.resize_handle.contains(ev.pos()) and ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self.is_resizing = True
            self.resize_start_pos = ev.pos()
            self.resize_start_rect = QtCore.QRectF(self.rect)
            ev.accept()
            return
        # --- End Resizing Logic ---

        if self.random_btn_rect and self.random_btn_rect.contains(ev.pos()):
            if hasattr(self.sim, 'randomize'):
                self.sim.randomize()
            ev.accept()
            return
        
        if self.zoom_in_rect and self.zoom_in_rect.contains(ev.pos()):
            self.sim.zoom_factor = max(0.1, self.sim.zoom_factor / 1.2) 
            self.update_display()
            ev.accept()
            return
        if self.zoom_out_rect and self.zoom_out_rect.contains(ev.pos()):
            self.sim.zoom_factor = min(5.0, self.sim.zoom_factor * 1.2) 
            self.update_display()
            ev.accept()
            return
            
        super().mousePressEvent(ev)
        
    def mouseMoveEvent(self, ev):
        # --- Resizing Logic ---
        if self.is_resizing:
            delta = ev.pos() - self.resize_start_pos
            
            new_w = max(self.min_w, self.resize_start_rect.width() + delta.x())
            new_h = max(self.min_h, self.resize_start_rect.height() + delta.y())
            
            self.prepareGeometryChange()
            self.rect.setWidth(new_w)
            self.rect.setHeight(new_h)
            self.update_port_positions()
            
            # --- FIX: Update internal size of passive nodes ---
            if hasattr(self.sim, 'w') and hasattr(self.sim, 'h'):
                 self.sim.w = int(new_w)
                 self.sim.h = int(new_h)
            # -------------------------------------------------
            
            ev.accept()
            return
        # --- End Resizing Logic ---

        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        # --- Resizing Logic ---
        if self.is_resizing:
            self.is_resizing = False
            self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            ev.accept()
            return
        # --- End Resizing Logic ---
        super().mouseReleaseEvent(ev)

    def boundingRect(self):
        # Adjust bounding rect to include shadows/glows/handles
        return self.rect.adjusted(-8, -8, 8, 8)
        
    def paint(self, painter, option, widget):
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        
        base = self.sim.NODE_COLOR
        if self.isSelected():
            base = base.lighter(150)
        painter.setBrush(QtGui.QBrush(base))
        painter.setPen(QtGui.QPen(QtGui.QColor(60, 60, 60), 2))
        painter.drawRoundedRect(self.rect, 10, 10)
        
        title_rect = QtCore.QRectF(8, 4, self.rect.width()-24, 20)
        painter.setPen(QtGui.QColor(240, 240, 240))
        font = QtGui.QFont("Arial", 10, QtGui.QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(title_rect, self.sim.node_title)
        
        category_rect = QtCore.QRectF(8, 18, self.rect.width()-16, 12)
        painter.setPen(QtGui.QColor(180, 180, 180))
        painter.setFont(QtGui.QFont("Arial", 7))
        painter.drawText(category_rect, self.sim.NODE_CATEGORY)
        
        painter.setFont(QtGui.QFont("Arial", 7))
        for name, port in self.in_ports.items():
            painter.drawText(port.pos() + QtCore.QPointF(12, 4), name)
        for name, port in self.out_ports.items():
            w = painter.fontMetrics().boundingRect(name).width()
            painter.drawText(port.pos() + QtCore.QPointF(-w - 12, 4), name)
            
        if self.display_pix:
            img_h = self.rect.height() - 50
            img_w = self.rect.width() - 16
            
            if img_h < 10 or img_w < 10: # Don't draw if too small
                return

            target_rect = QtCore.QRectF(8, 38, img_w, img_h)
            scaled = self.display_pix.scaled(
                int(img_w), int(img_h),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.FastTransformation
            )
            x = 8 + (img_w - scaled.width()) / 2
            y = 38 + (img_h - scaled.height()) / 2
            painter.drawPixmap(QtCore.QRectF(x, y, scaled.width(), scaled.height()),
                               scaled, QtCore.QRectF(scaled.rect()))
                                
        if self.random_btn_rect:
            painter.setBrush(QtGui.QColor(255, 200, 60))
            painter.setPen(QtGui.QColor(40, 40, 40))
            painter.drawEllipse(self.random_btn_rect)
            painter.setPen(QtGui.QColor(40, 40, 40))
            painter.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Weight.Bold))
            painter.drawText(self.random_btn_rect, QtCore.Qt.AlignmentFlag.AlignCenter, "R")
            
        if self.zoom_in_rect and self.zoom_in_rect.contains(ev.pos()):
            painter.setBrush(QtGui.QColor(60, 180, 255))
            painter.setPen(QtGui.QColor(40, 40, 40))
            painter.drawEllipse(self.zoom_in_rect)
            painter.setPen(QtGui.QColor(40, 40, 40))
            painter.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Weight.Bold))
            painter.drawText(self.zoom_in_rect, QtCore.Qt.AlignmentFlag.AlignCenter, "-")

            painter.setBrush(QtGui.QColor(60, 180, 255))
            painter.setPen(QtGui.QColor(40, 40, 40))
            painter.drawEllipse(self.zoom_out_rect)
            painter.setPen(QtGui.QColor(40, 40, 40))
            painter.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Weight.Bold))
            painter.drawText(self.zoom_out_rect, QtCore.Qt.AlignmentFlag.AlignCenter, "+")
        
        # --- Draw Resize Handle ---
        p = self.rect.bottomRight()
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 80), 1))
        painter.drawLine(int(p.x() - 12), int(p.y() - 4), int(p.x() - 4), int(p.y() - 12))
        painter.drawLine(int(p.x() - 8), int(p.y() - 4), int(p.x() - 4), int(p.y() - 8))
        # --- End Draw Handle ---
                                
    def update_display(self):
        qimg = self.sim.get_display_image()
        if qimg:
            self.display_pix = QtGui.QPixmap.fromImage(qimg)
        self.update()

class NodeConfigDialog(QtWidgets.QDialog):
    def __init__(self, node_item, parent=None):
        super().__init__(parent)
        self.node = node_item.sim
        self.setWindowTitle(f"Configure: {self.node.node_title}")
        self.setFixedWidth(300)
        
        layout = QtWidgets.QVBoxLayout(self)
        self.inputs = {}

        for display_name, key, current_value, options in self.node.get_config_options():
            h_layout = QtWidgets.QHBoxLayout()
            h_layout.addWidget(QtWidgets.QLabel(display_name + ":"))

            if options:
                combo = QtWidgets.QComboBox()
                for name, value in options:
                    combo.addItem(name, userData=value)
                    if value == current_value:
                        combo.setCurrentIndex(combo.count() - 1)
                
                if isinstance(current_value, (int, float)) and not any(v == current_value for _, v in options):
                    combo.addItem(f"Selected Device ({current_value})", userData=current_value)
                    combo.setCurrentIndex(combo.count() - 1)
                    
                h_layout.addWidget(combo, 1)
                self.inputs[key] = combo
            else:
                # Use QTextEdit for multi-line strings (like DocumentationNode)
                if isinstance(current_value, str) and ('\n' in current_value or len(current_value) > 40):
                     line_edit = QtWidgets.QTextEdit(str(current_value))
                     line_edit.setFixedHeight(100) # Fixed height for documentation
                else:
                     line_edit = QtWidgets.QLineEdit(str(current_value))
                     
                h_layout.addWidget(line_edit, 1)
                self.inputs[key] = line_edit
                
            layout.addLayout(h_layout)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok | 
            QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def get_new_config(self):
        new_config = {}
        for key, widget in self.inputs.items():
            if isinstance(widget, QtWidgets.QComboBox):
                new_config[key] = widget.currentData()
            elif isinstance(widget, QtWidgets.QLineEdit):
                text = widget.text()
                try:
                    # Try to convert to int, then float, then keep as string
                    new_val = float(text)
                    if new_val.is_integer():
                        new_val = int(new_val)
                    new_config[key] = new_val
                except ValueError:
                    new_config[key] = text
            elif isinstance(widget, QtWidgets.QTextEdit):
                new_config[key] = widget.toPlainText()
                
        return new_config

# ==================== MAIN SCENE ====================

class PerceptionScene(QtWidgets.QGraphicsScene):
    def __init__(self):
        super().__init__()
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(25, 25, 25)))
        self.nodes = []
        self.edges = []
        self.temp_edge = None
        self.connecting_src = None
        
    def add_node(self, node_class, x=0, y=0, w=NODE_W, h=NODE_H):
        sim = node_class()
        node = NodeItem(sim)
        
        # Set size if specified (e.g., from save file)
        if w != NODE_W or h != NODE_H:
            node.prepareGeometryChange()
            node.rect.setWidth(w)
            node.rect.setHeight(h)
            node.update_port_positions()

        self.addItem(node)
        node.setPos(x, y)
        self.nodes.append(node)
        node.update_display()
        return node
        
    def remove_node(self, node_item):
        if node_item in self.nodes:
            edges_to_remove = [
                e for e in self.edges 
                if e.src.parentItem() == node_item or e.tgt.parentItem() == node_item
            ]
            for edge in edges_to_remove:
                self.remove_edge(edge)
                
            node_item.sim.close()
            self.removeItem(node_item)
            self.nodes.remove(node_item)
            
    def remove_edge(self, edge):
        if edge in self.edges:
            self.removeItem(edge)
            self.edges.remove(edge)
            
    def start_connection(self, src_port):
        self.connecting_src = src_port
        self.temp_edge = EdgeItem(src_port)
        self.addItem(self.temp_edge)
        self.temp_edge.update_path()
        
    def finish_connection(self, tgt_port):
        if not self.connecting_src:
            return
        if (self.connecting_src.is_output and not tgt_port.is_output and
            self.connecting_src.port_type == tgt_port.port_type):
            
            if self.connecting_src.parentItem() == tgt_port.parentItem():
                self.cancel_connection()
                return

            edge_exists = any(
                e.src == self.connecting_src and e.tgt == tgt_port for e in self.edges
            )
            if edge_exists:
                self.cancel_connection()
                return
            
            edge = EdgeItem(self.connecting_src, tgt_port)
            self.addItem(edge)
            edge.update_path()
            self.edges.append(edge)
        self.cancel_connection()
        
    def cancel_connection(self):
        if self.temp_edge:
            self.removeItem(self.temp_edge)
        self.temp_edge = None
        self.connecting_src = None
        
    def mousePressEvent(self, ev):
        item = self.itemAt(ev.scenePos(), QtGui.QTransform())
        if isinstance(item, PortItem):
            if item.is_output:
                self.start_connection(item)
                return
            elif self.connecting_src:
                self.finish_connection(item)
                return
        super().mousePressEvent(ev)
        
    def mouseMoveEvent(self, ev):
        super().mouseMoveEvent(ev)
        if self.temp_edge and self.connecting_src:
            class FakePort:
                def __init__(self, pos): self._p = pos
                def scenePos(self): return self._p
            self.temp_edge.tgt = FakePort(ev.scenePos())
            self.temp_edge.update_path()
            
    def mouseReleaseEvent(self, ev):
        item = self.itemAt(ev.scenePos(), QtGui.QTransform())
        if isinstance(item, PortItem) and not item.is_output and self.connecting_src:
            self.finish_connection(item)
            return
        if self.connecting_src:
            self.cancel_connection()
        super().mouseReleaseEvent(ev)

    def delete_selected_nodes(self):
        selected_nodes = [i for i in self.selectedItems() if isinstance(i, NodeItem)]
        for node in selected_nodes:
            self.remove_node(node)
            
    def close_all(self):
        for node in self.nodes:
            node.sim.close()

# ==================== MAIN WINDOW ====================

class PerceptionLab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Antti's Perception Laboratory")
        self.resize(1400, 900)
        
        layout = QtWidgets.QVBoxLayout(self)
        
        self.scene = PerceptionScene()
        self.scene.parent = lambda: self 
        
        self.view = QtWidgets.QGraphicsView(self.scene)
        self.view.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing | 
                                QtGui.QPainter.RenderHint.SmoothPixmapTransform)
        self.view.setViewportUpdateMode(
            QtWidgets.QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate)
        
        # --- Context Menu Setup ---
        self.view.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self.on_context_menu)
        # --------------------------
        
        layout.addWidget(self.view, 1)
        
        self.status = QtWidgets.QLabel("Welcome! Add nodes and connect them.")
        self.status.setStyleSheet("color: #aaa; padding: 4px;")
        layout.addWidget(self.status)
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.step)
        self.is_running = False
        
        # Create node class map for loading saved graphs
        self.NODE_CLASS_MAP = NODE_TYPES
        
        # Create toolbar
        toolbar = self._create_toolbar()
        layout.insertLayout(0, toolbar)
        
    def _create_toolbar(self):
        tb = QtWidgets.QHBoxLayout()
        
        add_btn = QtWidgets.QPushButton("➕ Load Node Script")
        add_btn.clicked.connect(self.show_add_node_dialog) 
        tb.addWidget(add_btn)
        
        self.run_btn = QtWidgets.QPushButton("▶ Start")
        self.run_btn.clicked.connect(self.toggle_run)
        self.run_btn.setStyleSheet("background: #16a34a; color: white; padding: 6px 12px; font-weight: bold;")
        tb.addWidget(self.run_btn)
        
        clear_btn = QtWidgets.QPushButton("🗑 Clear Edges")
        clear_btn.clicked.connect(self.clear_edges)
        tb.addWidget(clear_btn)
        
        save_btn = QtWidgets.QPushButton("💾 Save Graph")
        save_btn.clicked.connect(self.save_graph)
        tb.addWidget(save_btn)
        
        load_btn = QtWidgets.QPushButton("📂 Load Graph")
        load_btn.clicked.connect(self.load_graph)
        tb.addWidget(load_btn)
        
        tb.addWidget(QtWidgets.QLabel("  Coupling:"))
        self.coupling_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.coupling_slider.setRange(0, 100)
        self.coupling_slider.setValue(70)
        self.coupling_slider.setMaximumWidth(150)
        tb.addWidget(self.coupling_slider)
        self.coupling_label = QtWidgets.QLabel("70%")
        self.coupling_slider.valueChanged.connect(
            lambda v: self.coupling_label.setText(f"{v}%"))
        tb.addWidget(self.coupling_label)
        
        tb.addStretch()
        
        self.fps_label = QtWidgets.QLabel("FPS: 0")
        self.fps_label.setStyleSheet("color: #666; font-family: monospace;")
        tb.addWidget(self.fps_label)
        
        return tb

    def show_add_node_dialog(self):
        """Opens a file dialog to select a node script directly."""
        nodes_path = os.path.join(os.path.dirname(__file__), 'nodes')
        if not os.path.exists(nodes_path):
            nodes_path = os.path.dirname(__file__) # Fallback if 'nodes' doesn't exist

        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 
            "Select Node Script", 
            nodes_path, 
            "Python Scripts (*.py)"
        )

        if fileName:
            module_name = os.path.basename(fileName).replace(".py", "")
            self.add_node_from_file(module_name)
    
    def add_node_from_file(self, module_name):
        """
        Finds all node classes from the selected module_name and adds them.
        If multiple classes are found (e.g., math_node.py), it asks which one.
        """
        
        # Find all classes loaded from this module
        classes_from_module = []
        for node_key, node_info in self.NODE_CLASS_MAP.items():
            if node_info["module_name"] == module_name:
                classes_from_module.append(node_info["class"])
        
        # Get the position to add the node
        view_center = self.view.mapToScene(self.view.viewport().rect().center())
        
        if not classes_from_module:
            self.status.setText(f"Error: No node classes found from '{module_name}.py'. Did it fail to load? Check dependencies.")
            return

        if len(classes_from_module) == 1:
            self.add_node_at_pos(classes_from_module[0], view_center)

        else:
            menu = QtWidgets.QMenu(self)
            menu.setTitle(f"Select node from {module_name}.py")
            
            for node_class in sorted(classes_from_module, key=lambda c: c.__name__):
                action = menu.addAction(node_class.__name__)
                action.triggered.connect(lambda _, nc=node_class, pos=view_center: self.add_node_at_pos(nc, pos))
            
            menu.exec(QtGui.QCursor.pos())

    def add_node_at_pos(self, node_class, scene_pos):
        """Helper to add a specific node class at a specific scene position."""
        node = self.scene.add_node(node_class, x=scene_pos.x()-100, y=scene_pos.y()-80)
        
        # Auto-configure setup methods if available
        if hasattr(node.sim, 'open_stream'):
            node.sim.open_stream()
        elif hasattr(node.sim, 'setup_source'):
            node.sim.setup_source()
            
        self.status.setText(f"Added {node.sim.node_title}")

    # --- NEW: Context Menu for Adding Nodes ---
    def add_node_context_menu(self, scene_pos):
        """Displays a categorized, scrollable menu for adding nodes."""
        menu = QtWidgets.QMenu(self)
        
        # Group nodes by category
        categories = {}
        for name, info in sorted(self.NODE_CLASS_MAP.items(), key=lambda item: item[1]['category']):
            category = info['category']
            if category not in categories:
                categories[category] = []
            categories[category].append((name, info['class']))
        
        for category, nodes in categories.items():
            submenu = menu.addMenu(category)
            
            for name, cls in nodes:
                action = submenu.addAction(name)
                action.triggered.connect(lambda _, nc=cls: self.add_node_at_pos(nc, scene_pos))
                
        return menu

    def on_context_menu(self, view_pos):
        """Handles right-click on the QGraphicsView and NodeItems."""
        scene_pos = self.view.mapToScene(view_pos)
        item = self.scene.itemAt(scene_pos, QtGui.QTransform())
        
        menu = QtWidgets.QMenu(self)
        
        selected_nodes = [i for i in self.scene.selectedItems() if isinstance(i, NodeItem)]
        clicked_node_item = None
        
        if isinstance(item, NodeItem):
            clicked_node_item = item
        elif isinstance(item, PortItem) or isinstance(item, QtWidgets.QGraphicsTextItem):
             if hasattr(item, 'parentItem') and isinstance(item.parentItem(), NodeItem):
                clicked_node_item = item.parentItem()

        if clicked_node_item and clicked_node_item not in selected_nodes:
             self.scene.clearSelection()
             clicked_node_item.setSelected(True)
             selected_nodes = [clicked_node_item]

        if selected_nodes:
            delete_action = menu.addAction(f"Delete Selected Node{'s' if len(selected_nodes) > 1 else ''} ({len(selected_nodes)})")
            delete_action.triggered.connect(lambda: self.scene.delete_selected_nodes())
            
            if len(selected_nodes) == 1:
                menu.addSeparator()
                config_action = menu.addAction("⚙ Configure Node...")
                config_action.triggered.connect(lambda: self.configure_node(selected_nodes[0]))
        else:
            # If nothing is selected, show the categorized Add Node menu
            add_menu = self.add_node_context_menu(scene_pos)
            menu.addMenu(add_menu)
            
        global_pos = self.view.mapToGlobal(view_pos)
        menu.exec(global_pos)
    # --------------------------------

    def configure_node(self, node_item):
        dialog = NodeConfigDialog(node_item, self)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            new_config = dialog.get_new_config()
            
            for key, value in new_config.items():
                setattr(node_item.sim, key, value)
                
            if hasattr(node_item.sim, 'open_stream'):
                node_item.sim.open_stream()
            elif hasattr(node_item.sim, 'setup_source'):
                node_item.sim.setup_source()
            
            node_item.update()
            node_item.update_port_positions() # Update in case config changed size
            self.status.setText(f"Configured {node_item.sim.node_title}")
        
    def toggle_run(self):
        self.is_running = not self.is_running
        if self.is_running:
            self.run_btn.setText("⏸ Stop")
            self.run_btn.setStyleSheet("background: #dc2626; color: white; padding: 6px 12px; font-weight: bold;")
            self.timer.start(33)
            self.status.setText("Running...")
            self.last_time = QtCore.QTime.currentTime()
            self.frame_count = 0
        else:
            self.run_btn.setText("▶ Start")
            self.run_btn.setStyleSheet("background: #16a34a; color: white; padding: 6px 12px; font-weight: bold;")
            self.timer.stop()
            self.status.setText("Stopped")
            
    def clear_edges(self):
        for edge in list(self.scene.edges):
            self.scene.remove_edge(edge)
        self.status.setText("Cleared all edges")
        
    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key.Key_Delete or event.key() == QtCore.Qt.Key.Key_Backspace:
            self.scene.delete_selected_nodes()
            self.status.setText("Deleted selected nodes.")
            return
        super().keyPressEvent(event)
        
    def step(self):
        for node in self.scene.nodes:
            node.sim.pre_step()
            
        node_map = {n: n for n in self.scene.nodes}
        coupling = self.coupling_slider.value() / 100.0
        
        for edge in self.scene.edges:
            src_node = edge.src.parentItem()
            tgt_node = edge.tgt.parentItem()
            
            if src_node not in node_map or tgt_node not in node_map:
                continue
                
            output = src_node.sim.get_output(edge.src.name)
            if output is None:
                continue
                
            tgt_node.sim.set_input(edge.tgt.name, output, edge.src.port_type, coupling)
            
            if edge.src.port_type == 'signal':
                if isinstance(output, (float, int)):
                    edge.effect_val = abs(float(output) * coupling)
                elif isinstance(output, np.ndarray) and output.size == 1:
                    edge.effect_val = abs(float(output.flat[0]) * coupling)
                else:
                    edge.effect_val = 0.5
            elif edge.src.port_type in ['spectrum', 'image', 'complex_spectrum']:
                edge.effect_val = 0.8
                if isinstance(output, np.ndarray) and output.size > 0:
                     edge.effect_val += np.mean(np.abs(output)) * 0.1
            else:
                edge.effect_val = 0.5
            edge.update_path()
            
        for node in self.scene.nodes:
            node.sim.step()
            node.update_display()
            
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            current_time = QtCore.QTime.currentTime()
            elapsed = self.last_time.msecsTo(current_time) / 1000.0
            if elapsed > 0:
                fps = 30.0 / elapsed
                self.fps_label.setText(f"FPS: {fps:.1f}")
            self.last_time = current_time
            
    def closeEvent(self, event):
        self.timer.stop()
        self.scene.close_all()
        if PA_INSTANCE:
            try:
                PA_INSTANCE.terminate()
                print("PyAudio instance terminated.")
            except Exception as e:
                print(f"Error terminating PyAudio: {e}")
        super().closeEvent(event)

    def clear_all(self):
        """Clears all nodes and edges from the scene."""
        if self.is_running:
            self.toggle_run()
        
        self.clear_edges()
        for node in list(self.scene.nodes):
            self.scene.remove_node(node)
        self.status.setText("Graph cleared")

    def save_graph(self):
        """Saves the current node graph to a JSON file."""
        if self.is_running:
            self.toggle_run()

        fileName, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Graph", "", "JSON Files (*.json)")
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
            self.status.setText(f"Graph saved to {os.path.basename(fileName)}")
        except Exception as e:
            self.status.setText(f"Error saving graph: {e}")
            print(f"Error saving graph: {e}")

    def load_graph(self):
        """Loads a node graph from a JSON file."""
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load Graph", "", "JSON Files (*.json)")
        if not fileName:
            return
            
        try:
            with open(fileName, 'r') as f:
                graph_data = json.load(f)
        except Exception as e:
            self.status.setText(f"Error loading file: {e}")
            return

        self.clear_all()
        
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
                print(f"Warning: Node class '{class_name}' not found in registry. Skipping.")

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
                print(f"Error loading edge: {e}. Skipping.")

        settings = graph_data.get("global_settings", {})
        self.coupling_slider.setValue(settings.get("coupling", 70))

        self.status.setText(f"Graph loaded from {os.path.basename(fileName)}")

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
            border: none;
            border-radius: 4px;
            padding: 6px 12px;
            background: #3a3a3a;
            color: #ddd;
        }
        QPushButton:hover {
            background: #4a4a4a;
        }
        QPushButton:pressed {
            background: #2a2a2a;
        }
        QPushButton::menu-indicator {
            width: 0px;
        }
        QMenu {
            background: #2a2a2a;
            border: 1px solid #444;
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
            padding: 2px;
            color: #ddd;
            border-radius: 4px;
            height: 24px;
        }
        QTextEdit {
            background: #3a3a3a;
            border: 1px solid #555;
            padding: 2px;
            color: #ddd;
            border-radius: 4px;
        }
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 15px;
            border-left-width: 1px;
            border-left-color: #555;
            border-left-style: solid;
        }
    """)
    
    window = PerceptionLab()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
