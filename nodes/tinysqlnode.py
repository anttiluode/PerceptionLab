"""
Tiny SQL Node - A lightweight database for your graph.
Uses Python's built-in sqlite3.

Inputs:
    query: String query to execute (can be parameterized with inputs).
    trigger: Signal (0->1) to execute the query.
    param1..3: Signals/Values to inject into the query parameters (?).

Outputs:
    result_text: JSON string of the fetched data.
    result_signal: The first numeric value of the first row (useful for logic).
    row_count: Number of rows affected or returned.
"""

import sqlite3
import json
import os
import numpy as np
from PyQt6 import QtGui  # ✅ FIXED
import __main__

BaseNode = __main__.BaseNode

class TinySQLNode(BaseNode):
    NODE_CATEGORY = "Data"
    NODE_COLOR = QtGui.QColor(100, 100, 120)  # Database Grey
    
    def __init__(self, db_path="perception_lab.db", default_query="SELECT sqlite_version()"):
        super().__init__()
        self.node_title = "Tiny SQL"
        
        self.inputs = {
            'trigger': 'signal',
            'param_1': 'signal',
            'param_2': 'signal',
            'param_3': 'signal'
        }
        
        self.outputs = {
            'result_text': 'text_multi', # Assuming host supports text output display or similar
            'result_signal': 'signal',
            'row_count': 'signal'
        }
        
        self.db_path = db_path
        self.query = default_query
        
        self.last_trigger = 0.0
        self.conn = None
        self.result_json = "[]"
        self.result_val = 0.0
        self.row_count_val = 0.0
        
        self._connect()

    def _connect(self):
        try:
            # Check if we are in a persistent environment or need a full path
            # For now, local file relative to script execution
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            # Enable column access by name
            self.conn.row_factory = sqlite3.Row
        except Exception as e:
            print(f"TinySQL: Connection failed: {e}")

    def get_config_options(self):
        return [
            ("Database Path", "db_path", self.db_path, None),
            ("SQL Query", "query", self.query, "text_multi"), # Multiline text edit
        ]

    def set_config_options(self, options):
        if "db_path" in options:
            self.db_path = options["db_path"]
            if self.conn: self.conn.close()
            self._connect()
        if "query" in options:
            self.query = options["query"]

    def step(self):
        trigger = self.get_blended_input('trigger', 'sum') or 0.0
        
        # Execute on rising edge
        if trigger > 0.5 and self.last_trigger <= 0.5:
            self.execute_query()
            
        self.last_trigger = trigger
    
    def execute_query(self):
        if not self.conn: return
        
        # Gather parameters
        p1 = self.get_blended_input('param_1', 'sum')
        p2 = self.get_blended_input('param_2', 'sum')
        p3 = self.get_blended_input('param_3', 'sum')
        
        # Filter None values
        params = []
        if p1 is not None: params.append(float(p1))
        if p2 is not None: params.append(float(p2))
        if p3 is not None: params.append(float(p3))
        
        # We only use as many params as the query has '?' placeholders
        needed_params = self.query.count('?')
        params = params[:needed_params]
        
        try:
            cursor = self.conn.cursor()
            cursor.execute(self.query, tuple(params))
            
            if self.query.strip().upper().startswith("SELECT"):
                rows = cursor.fetchall()
                self.row_count_val = float(len(rows))
                
                # Convert to list of dicts
                result_data = [dict(row) for row in rows]
                self.result_json = json.dumps(result_data, indent=2)
                
                # Extract first scalar for signal output
                if len(rows) > 0 and len(rows[0]) > 0:
                    first_val = rows[0][0]
                    if isinstance(first_val, (int, float)):
                        self.result_val = float(first_val)
                    else:
                        self.result_val = 1.0 # Valid result but not a number
                else:
                    self.result_val = 0.0
            else:
                # INSERT/UPDATE/DELETE
                self.conn.commit()
                self.row_count_val = float(cursor.rowcount)
                self.result_json = f'{{"status": "success", "rows_affected": {cursor.rowcount}}}'
                self.result_val = float(cursor.rowcount)
                
        except sqlite3.Error as e:
            self.result_json = f'{{"error": "{str(e)}"}}'
            print(f"TinySQL Error: {e}")
            self.result_val = -1.0

    def get_output(self, port_name):
        if port_name == 'result_text':
            return self.result_json
        elif port_name == 'result_signal':
            return self.result_val
        elif port_name == 'row_count':
            return self.row_count_val
        return None
    
    def get_display_image(self):
        # Simple text display of the result JSON (truncated)
        import cv2
        from PIL import Image, ImageDraw, ImageFont
        
        w, h = 256, 128
        img = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Background color based on success/error
        if "error" in self.result_json:
            img[:, :] = (50, 0, 0) # Dark red
        else:
            img[:, :] = (20, 20, 30) # Dark grey
            
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        
        # Try to load a font
        try:
            font = ImageFont.load_default()
        except:
            font = None
            
        # Draw query snippet
        draw.text((5, 5), f"Q: {self.query[:30]}...", fill=(200, 200, 200), font=font)
        
        # Draw result snippet
        lines = self.result_json.split('\n')
        y = 25
        for line in lines[:6]: # Show first 6 lines
            draw.text((5, y), line[:40], fill=(100, 255, 100), font=font)
            y += 12
            
        img = np.array(img_pil)
        return QtGui.QImage(img.data, w, h, 3*w, QtGui.QImage.Format.Format_RGB888)

    def close(self):
        if self.conn:
            self.conn.close()