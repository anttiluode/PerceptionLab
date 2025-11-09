#!/usr/bin/env python3
"""
Frame Exporter for Infinite Fractal Landscape
Add this to your Perception Lab to export high-quality frame sequences.

Usage:
1. Add this node to your workflow
2. Connect the fractal image output to this node's image input
3. Set export parameters in config
4. Run workflow - frames will be saved to disk

Commercial use: Export sequences for video editing or stock footage sales
"""

import numpy as np
import cv2
import os
from datetime import datetime
from pathlib import Path

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class FrameExporterNode(BaseNode):
    """Exports frames to disk for video production"""
    NODE_CATEGORY = "Output"
    NODE_COLOR = QtGui.QColor(255, 100, 100)
    
    def __init__(self, 
                 export_enabled=False,
                 output_dir="./fractal_export",
                 frame_prefix="fractal",
                 export_format="png",
                 export_every_n_frames=1,
                 max_frames=1000):
        super().__init__()
        self.node_title = "Frame Exporter"
        
        self.inputs = {
            'image': 'image',
            'trigger': 'signal'  # Set to 1.0 to enable export
        }
        self.outputs = {
            'frame_count': 'signal',
            'export_status': 'signal'
        }
        
        # Export settings
        self.export_enabled = bool(export_enabled)
        self.output_dir = str(output_dir)
        self.frame_prefix = str(frame_prefix)
        self.export_format = str(export_format)  # 'png', 'jpg', 'tiff'
        self.export_every_n_frames = int(export_every_n_frames)
        self.max_frames = int(max_frames)
        
        # State
        self.frame_counter = 0
        self.frames_exported = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.last_trigger = 0.0
        
        # Create output directory
        self.setup_output_dir()
        
    def setup_output_dir(self):
        """Create output directory structure"""
        session_dir = os.path.join(self.output_dir, self.session_id)
        Path(session_dir).mkdir(parents=True, exist_ok=True)
        self.session_dir = session_dir
        print(f"FrameExporter: Output directory: {self.session_dir}")
        
    def step(self):
        # Get input
        image = self.get_blended_input('image', 'max')
        trigger = self.get_blended_input('trigger', 'sum') or 0.0
        
        # Check if export should be enabled via trigger
        if trigger > 0.5 and self.last_trigger <= 0.5:
            self.export_enabled = not self.export_enabled
            print(f"FrameExporter: Export {'ENABLED' if self.export_enabled else 'DISABLED'}")
        self.last_trigger = trigger
        
        # Increment frame counter
        self.frame_counter += 1
        
        # Export if enabled and conditions met
        should_export = (
            self.export_enabled 
            and image is not None 
            and self.frame_counter % self.export_every_n_frames == 0
            and self.frames_exported < self.max_frames
        )
        
        if should_export:
            self.export_frame(image)
            
        # Output status
        self.set_output('frame_count', float(self.frame_counter))
        self.set_output('export_status', 1.0 if self.export_enabled else 0.0)
        
    def export_frame(self, image):
        """Save frame to disk"""
        try:
            # Generate filename
            filename = f"{self.frame_prefix}_{self.frames_exported:06d}.{self.export_format}"
            filepath = os.path.join(self.session_dir, filename)
            
            # Convert to uint8 if needed
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = np.clip(image, 0, 255).astype(np.uint8)
            
            # Handle grayscale vs color
            if len(image.shape) == 2:
                # Grayscale - convert to BGR for color output
                image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 3:
                # Assume RGB, convert to BGR for OpenCV
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif image.shape[2] == 4:
                # RGBA, convert to BGR
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            else:
                image_bgr = image
            
            # Set quality based on format
            if self.export_format == 'jpg':
                cv2.imwrite(filepath, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            elif self.export_format == 'png':
                cv2.imwrite(filepath, image_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            elif self.export_format == 'tiff':
                cv2.imwrite(filepath, image_bgr)
            else:
                cv2.imwrite(filepath, image_bgr)
            
            self.frames_exported += 1
            
            # Progress logging
            if self.frames_exported % 100 == 0:
                print(f"FrameExporter: {self.frames_exported} frames exported")
                
        except Exception as e:
            print(f"FrameExporter: Error exporting frame: {e}")


class VideoExporterNode(BaseNode):
    """Exports directly to video file using cv2.VideoWriter"""
    NODE_CATEGORY = "Output"
    NODE_COLOR = QtGui.QColor(255, 80, 80)
    
    def __init__(self,
                 export_enabled=False,
                 output_dir="./fractal_export",
                 filename="fractal_video",
                 fps=30,
                 codec='mp4v',
                 width=1920,
                 height=1080):
        super().__init__()
        self.node_title = "Video Exporter"
        
        self.inputs = {
            'image': 'image',
            'trigger': 'signal'
        }
        self.outputs = {
            'frame_count': 'signal',
            'recording': 'signal'
        }
        
        # Settings
        self.export_enabled = bool(export_enabled)
        self.output_dir = str(output_dir)
        self.filename = str(filename)
        self.fps = int(fps)
        self.codec = str(codec)
        self.width = int(width)
        self.height = int(height)
        
        # State
        self.writer = None
        self.frame_count = 0
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.last_trigger = 0.0
        
        # Setup
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
    def start_recording(self):
        """Initialize video writer"""
        if self.writer is not None:
            self.stop_recording()
            
        output_path = os.path.join(
            self.output_dir,
            f"{self.filename}_{self.session_id}.mp4"
        )
        
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(
            output_path,
            fourcc,
            self.fps,
            (self.width, self.height)
        )
        
        if self.writer.isOpened():
            print(f"VideoExporter: Recording started: {output_path}")
            return True
        else:
            print(f"VideoExporter: Failed to open video writer")
            self.writer = None
            return False
            
    def stop_recording(self):
        """Finalize and close video file"""
        if self.writer is not None:
            self.writer.release()
            print(f"VideoExporter: Recording stopped. {self.frame_count} frames written.")
            self.writer = None
            self.frame_count = 0
            
    def step(self):
        # Get inputs
        image = self.get_blended_input('image', 'max')
        trigger = self.get_blended_input('trigger', 'sum') or 0.0
        
        # Toggle recording on trigger
        if trigger > 0.5 and self.last_trigger <= 0.5:
            if self.writer is None:
                self.start_recording()
            else:
                self.stop_recording()
        self.last_trigger = trigger
        
        # Write frame if recording
        if self.writer is not None and image is not None:
            try:
                # Resize to target resolution
                resized = cv2.resize(image, (self.width, self.height))
                
                # Convert to uint8 BGR
                if resized.dtype != np.uint8:
                    if resized.max() <= 1.0:
                        resized = (resized * 255).astype(np.uint8)
                    else:
                        resized = np.clip(resized, 0, 255).astype(np.uint8)
                        
                if len(resized.shape) == 2:
                    resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
                elif resized.shape[2] == 3:
                    resized = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
                elif resized.shape[2] == 4:
                    resized = cv2.cvtColor(resized, cv2.COLOR_RGBA2BGR)
                    
                self.writer.write(resized)
                self.frame_count += 1
                
            except Exception as e:
                print(f"VideoExporter: Error writing frame: {e}")
                
        # Output status
        self.set_output('frame_count', float(self.frame_count))
        self.set_output('recording', 1.0 if self.writer is not None else 0.0)
        
    def cleanup(self):
        """Ensure video is finalized on node deletion"""
        self.stop_recording()


# Export both node classes
__all__ = ['FrameExporterNode', 'VideoExporterNode']


"""
USAGE EXAMPLES:

1. FRAME SEQUENCE EXPORT (for compositing):
   - Add FrameExporterNode to workflow
   - Connect fractal image -> FrameExporterNode.image
   - Set export_format='png' for lossless
   - Set export_every_n_frames=1 for every frame
   - Set max_frames=3000 for 100 seconds at 30fps
   - Connect trigger signal or manually set export_enabled=True

2. DIRECT VIDEO EXPORT (for quick sharing):
   - Add VideoExporterNode to workflow  
   - Connect fractal image -> VideoExporterNode.image
   - Set fps=60, width=1920, height=1080
   - Toggle recording with trigger signal
   - Video saves automatically when stopped

3. COMMERCIAL STOCK FOOTAGE:
   - Use FrameExporterNode with:
     * export_format='tiff' for maximum quality
     * Resolution set to 3840x2160 (4K)
     * Export 30 seconds = 900 frames at 30fps
   - Import sequence to video editor
   - Apply color grading
   - Export final at high bitrate
   - Upload to stock sites

4. REALTIME STREAMING:
   - Use VideoExporterNode
   - Set up OBS to capture the output folder
   - Stream the live generation process
   - Archive saves automatically

TO ADD TO YOUR PERCEPTION LAB:
1. Save this file as FrameExporterNode.py in your nodes directory
2. Restart Perception Lab
3. Nodes appear in "Output" category
4. Add to any workflow
"""