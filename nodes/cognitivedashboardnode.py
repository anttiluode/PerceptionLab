"""
Cognitive Dashboard Node
========================
Visualizes the full Hopfield/ATP cycle.
Displays the noisy query vs. the reconstructed memory, alongside metabolic state.
"""
import numpy as np
import cv2
import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui

class CognitiveDashboardNode(BaseNode):
    NODE_CATEGORY = "Analysis"
    NODE_COLOR = QtGui.QColor(80, 80, 80)

    def __init__(self):
        super().__init__()
        self.node_title = "Cognitive Dashboard"
        self.inputs = {
            'query_img': 'image',
            'recall_img': 'image',
            'atp_level': 'signal'
        }
        self.outputs = {}
        self.display_img = np.zeros((128, 256, 3), dtype=np.uint8)

    def step(self):
        query = self.get_blended_input('query_img', 'first')
        recall = self.get_blended_input('recall_img', 'first')
        atp = self.get_blended_input('atp_level', 'sum') or 0.0

        img = np.zeros((128, 256, 3), dtype=np.uint8)

        if query is not None:
            q_u8 = (np.clip(query, 0, 1) * 255).astype(np.uint8)
            q_res = cv2.resize(q_u8, (100, 100), interpolation=cv2.INTER_NEAREST)
            img[20:120, 10:110] = q_res
            cv2.putText(img, "Noisy Query", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        if recall is not None:
            r_u8 = (np.clip(recall, 0, 1) * 255).astype(np.uint8)
            r_res = cv2.resize(r_u8, (100, 100), interpolation=cv2.INTER_NEAREST)
            
            # If ATP is very low, visually dim the recall to represent "resting"
            if atp < 0.2:
                r_res = (r_res * 0.3).astype(np.uint8)

            img[20:120, 140:240] = r_res
            cv2.putText(img, "Denoised Recall", (140, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        # Border indicating metabolism
        color = (0, 255, 0) if atp > 0.2 else (0, 0, 255)
        cv2.rectangle(img, (0, 0), (255, 127), color, 2)

        self.display_img = img

    def get_output(self, port_name):
        return None

    def get_display_image(self):
        return QtGui.QImage(self.display_img.data, 256, 128, 256*3, QtGui.QImage.Format.Format_RGB888)