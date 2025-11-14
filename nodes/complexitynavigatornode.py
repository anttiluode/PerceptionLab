"""
ComplexityNavigatorNode (Simplified)
-------------------------------------
Consciousness navigates toward regions of high alignment.
Gets stuck in damaged/low-complexity areas.
"""

import numpy as np
import cv2
import __main__
BaseNode = __main__.BaseNode
PA_INSTANCE = getattr(__main__, "PA_INSTANCE", None)
QtGui = __main__.QtGui

class ComplexityNavigatorNode(BaseNode):
    NODE_CATEGORY = "Fractal Substrate"
    NODE_COLOR = QtGui.QColor(200, 50, 200)

    def __init__(self, num_particles=5, speed=2.0, attraction=1.0):
        super().__init__()
        self.node_title = "Complexity Navigator"

        self.inputs = {
            'alignment_field': 'image',
            'complexity_map': 'image',
        }

        self.outputs = {
            'navigator_positions': 'image',
            'navigation_trails': 'image',
            'current_complexity': 'signal',
        }

        self.num_particles = int(num_particles)
        self.speed = float(speed)
        self.attraction = float(attraction)
        
        self.field_size = 256
        self.positions = np.random.rand(self.num_particles, 2) * self.field_size
        self.velocities = np.random.randn(self.num_particles, 2) * 0.5
        
        self.trails = [[] for _ in range(self.num_particles)]
        self.trail_length = 50
        
        self.navigator_image = np.zeros((self.field_size, self.field_size), dtype=np.float32)
        self.trails_image = np.zeros((self.field_size, self.field_size), dtype=np.float32)
        self.avg_complexity = 0.0

    def _sample_field(self, field, pos):
        """Sample field value at position"""
        if field is None:
            return 0.5
        
        x, y = int(pos[0]), int(pos[1])
        x = np.clip(x, 0, field.shape[1] - 1)
        y = np.clip(y, 0, field.shape[0] - 1)
        
        return field[y, x]

    def _compute_gradient(self, field, pos):
        """Compute gradient (direction of increase)"""
        if field is None:
            return np.array([0.0, 0.0])
        
        delta = 3.0
        x, y = pos
        
        val_right = self._sample_field(field, [x + delta, y])
        val_left = self._sample_field(field, [x - delta, y])
        val_down = self._sample_field(field, [x, y + delta])
        val_up = self._sample_field(field, [x, y - delta])
        
        grad_x = (val_right - val_left) / (2 * delta)
        grad_y = (val_down - val_up) / (2 * delta)
        
        return np.array([grad_x, grad_y])

    def step(self):
        # Get inputs
        alignment = self.get_blended_input('alignment_field', 'mean')
        complexity = self.get_blended_input('complexity_map', 'mean')
        
        # Resize if needed
        if alignment is not None:
            if alignment.shape[:2] != (self.field_size, self.field_size):
                alignment = cv2.resize(alignment, (self.field_size, self.field_size))
            if alignment.ndim == 3:
                alignment = np.mean(alignment, axis=2)
        
        if complexity is not None:
            if complexity.shape[:2] != (self.field_size, self.field_size):
                complexity = cv2.resize(complexity, (self.field_size, self.field_size))
            if complexity.ndim == 3:
                complexity = np.mean(complexity, axis=2)
        
        # Update each navigator
        complexities = []
        for i in range(self.num_particles):
            pos = self.positions[i]
            vel = self.velocities[i]
            
            # Sense local alignment (attraction to info channels)
            gradient = self._compute_gradient(alignment, pos)
            
            # Sense local complexity
            local_complexity = self._sample_field(complexity, pos)
            complexities.append(local_complexity)
            
            # Forces:
            # 1. Attraction to high alignment
            force = gradient * self.attraction
            
            # 2. Small random exploration
            force += np.random.randn(2) * 0.2
            
            # 3. Damping
            force -= vel * 0.1
            
            # Update
            vel += force * 0.1
            speed_limit = self.speed * (0.5 + 0.5 * local_complexity)  # Slower in low complexity
            vel_magnitude = np.linalg.norm(vel)
            if vel_magnitude > speed_limit:
                vel = vel / vel_magnitude * speed_limit
            
            pos += vel
            
            # Wrap boundaries
            pos[0] = pos[0] % self.field_size
            pos[1] = pos[1] % self.field_size
            
            self.positions[i] = pos
            self.velocities[i] = vel
            
            # Update trail
            self.trails[i].append(pos.copy())
            if len(self.trails[i]) > self.trail_length:
                self.trails[i].pop(0)
        
        self.avg_complexity = np.mean(complexities) if complexities else 0.0
        
        # Generate output images
        self.navigator_image.fill(0)
        self.trails_image.fill(0)
        
        # Draw trails
        for trail in self.trails:
            for j in range(len(trail) - 1):
                p1 = trail[j].astype(int)
                p2 = trail[j + 1].astype(int)
                intensity = (j + 1) / len(trail)
                cv2.line(self.trails_image, tuple(p1), tuple(p2), intensity, 1)
        
        # Draw current positions
        for pos in self.positions:
            x, y = int(pos[0]), int(pos[1])
            cv2.circle(self.navigator_image, (x, y), 3, 1.0, -1)

    def get_output(self, port_name):
        if port_name == 'navigator_positions':
            return self.navigator_image
        elif port_name == 'navigation_trails':
            return self.trails_image
        elif port_name == 'current_complexity':
            return self.avg_complexity
        return None

    def get_display_image(self):
        display_w = 512
        display_h = 512
        
        # Get alignment field for background
        alignment = self.get_blended_input('alignment_field', 'mean')
        if alignment is not None:
            if alignment.shape[:2] != (self.field_size, self.field_size):
                alignment = cv2.resize(alignment, (self.field_size, self.field_size))
            if alignment.ndim == 3:
                alignment = np.mean(alignment, axis=2)
            
            bg_u8 = (alignment * 255).astype(np.uint8)
            bg_color = cv2.applyColorMap(bg_u8, cv2.COLORMAP_OCEAN)
        else:
            bg_color = np.zeros((self.field_size, self.field_size, 3), dtype=np.uint8)
        
        # Overlay trails
        trails_u8 = (self.trails_image * 255).astype(np.uint8)
        trails_color = cv2.applyColorMap(trails_u8, cv2.COLORMAP_HOT)
        
        # Blend
        display = cv2.addWeighted(bg_color, 0.6, trails_color, 0.4, 0)
        
        # Draw current positions
        for pos in self.positions:
            x, y = int(pos[0]), int(pos[1])
            cv2.circle(display, (x, y), 4, (255, 255, 255), -1)
            cv2.circle(display, (x, y), 6, (255, 0, 255), 2)
        
        # Resize
        display = cv2.resize(display, (display_w, display_h))
        
        # Info
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display, 'CONSCIOUSNESS NAVIGATION', (10, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display, f'Avg Complexity: {self.avg_complexity:.3f}', 
                   (10, display_h - 10), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        
        display = np.ascontiguousarray(display)
        return QtGui.QImage(display.data, display_w, display_h, 3*display_w, QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Num Particles", "num_particles", self.num_particles, None),
            ("Speed", "speed", self.speed, None),
            ("Attraction", "attraction", self.attraction, None),
        ]