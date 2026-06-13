"""
Token Relay Node (The Synapse Across Machines)
==============================================
Wires a local Mycelial Cortex to a token relay so two PerceptionLab instances on
two machines become one mesh. Broadcasts the cortex's recalled pattern as a
SpikeToken (sparse, gated by confidence) and emits received peer tokens for the
cortex to ingest. No shared weights ever cross the wire — only tokens.

Wiring:
  TEACHER:  Cortex.recall_vec -> Relay.token_in ;  Cortex.confidence -> Relay.confidence
  LISTENER: Relay.token_out   -> Cortex.query_vec  (peer tokens are its only input)

Set host/port in config to the machine running token_relay_server.py.
Networking runs on a background thread; the GUI never blocks.

VERIFIED (federation_proof.py): a pattern taught on peer A was recalled on peer B
through 39 sparse tokens, B's template cos 0.97 to the secret, recall cos 0.77.

PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.
Do not hype. Do not lie. Just show.
"""
import numpy as np
import cv2
import socket
import threading
import queue
import json
import time
from collections import deque

import __main__
BaseNode = __main__.BaseNode
QtGui = __main__.QtGui


class RelayClient:
    """Non-blocking background-thread TCP client for SpikeTokens (inlined, pure stdlib)."""
    def __init__(self, host="127.0.0.1", port=8765, src=0):
        self.host = host; self.port = port; self.src = int(src)
        self.outq = queue.Queue(maxsize=256)
        self.inbox = []; self.lock = threading.Lock()
        self.running = False; self.connected = False
        self.tx = 0; self.rx = 0; self.sock = None

    def start(self):
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def _loop(self):
        while self.running:
            try:
                self.sock = socket.create_connection((self.host, self.port), timeout=3)
                self.connected = True
                threading.Thread(target=self._recv, daemon=True).start()
                while self.running and self.connected:
                    try:
                        tok = self.outq.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    try:
                        self.sock.sendall((json.dumps(tok) + "\n").encode()); self.tx += 1
                    except OSError:
                        self.connected = False
            except OSError:
                self.connected = False; time.sleep(1.0)

    def _recv(self):
        buf = b""
        while self.running and self.connected:
            try:
                data = self.sock.recv(1 << 16)
            except OSError:
                break
            if not data:
                break
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                try:
                    tok = json.loads(line.decode())
                except Exception:
                    continue
                with self.lock:
                    self.inbox.append(tok); self.rx += 1
        self.connected = False

    def send(self, payload, conf=1.0, chi=0):
        try:
            self.outq.put_nowait({"src": self.src, "payload": [float(x) for x in payload],
                                  "conf": float(conf), "chi": int(chi)})
        except queue.Full:
            pass

    def poll(self):
        with self.lock:
            items = self.inbox; self.inbox = []
        return items

    def stop(self):
        self.running = False
        try:
            self.sock.close()
        except (OSError, AttributeError):
            pass


class TokenRelayNode(BaseNode):
    NODE_CATEGORY = "Network"
    NODE_COLOR = QtGui.QColor(60, 140, 200)  # wire blue

    def __init__(self, host="127.0.0.1", port=8765, src_id=1,
                 broadcast_every=3, conf_gate=0.3):
        super().__init__()
        self.node_title = "Token Relay (Synapse)"
        self.inputs = {
            'token_in':   'spectrum',   # local pattern to broadcast (e.g. Cortex.recall_vec)
            'confidence': 'signal',     # gate: only broadcast when confident
            'enable':     'signal',     # 1 = broadcasting on (default on)
        }
        self.outputs = {
            'token_out': 'spectrum',    # a received peer token -> Cortex.query_vec / inject_token
            'peers':     'signal',      # 1 if connected to the relay
            'tx':        'signal',      # tokens sent
            'rx':        'signal',      # tokens received
            'viz':       'image',
        }
        self.host = str(host); self.port = int(port); self.src_id = int(src_id)
        self.broadcast_every = int(broadcast_every); self.conf_gate = float(conf_gate)

        self.client = RelayClient(self.host, self.port, self.src_id).start()
        self.frame = 0
        self.last_sent = None
        self.recv_fifo = deque(maxlen=64)     # received payloads, emitted one per frame
        self.token_out = None
        self.last_recv_vec = None
        self.display_img = np.zeros((96, 256, 3), dtype=np.uint8)

    def step(self):
        self.frame += 1
        tin = self.get_blended_input('token_in', 'first')
        conf = self.get_blended_input('confidence', 'sum')
        en = self.get_blended_input('enable', 'sum')
        enabled = True if en is None else (float(en) > 0.5)
        conf = float(conf) if conf is not None else 1.0

        # --- broadcast (sparse, gated, deduped) ---
        if enabled and self.client.connected and tin is not None \
                and conf >= self.conf_gate and (self.frame % self.broadcast_every == 0):
            v = np.array(tin, dtype=np.float32)
            changed = (self.last_sent is None) or \
                      (float(np.dot(v, self.last_sent) /
                       ((np.linalg.norm(v)+1e-9)*(np.linalg.norm(self.last_sent)+1e-9))) < 0.999)
            if changed:
                self.client.send(v, conf=conf)
                self.last_sent = v.copy()

        # --- receive: queue arrivals, emit one per frame ---
        for tok in self.client.poll():
            self.recv_fifo.append(np.array(tok.get("payload", []), dtype=np.float32))
        if self.recv_fifo:
            self.token_out = self.recv_fifo.popleft()
            self.last_recv_vec = self.token_out
        else:
            self.token_out = None     # nothing this frame -> downstream stays idle (event-driven)

        self._render()

    def get_output(self, port_name):
        if port_name == 'token_out': return self.token_out
        if port_name == 'peers':     return 1.0 if self.client.connected else 0.0
        if port_name == 'tx':        return float(self.client.tx)
        if port_name == 'rx':        return float(self.client.rx)
        if port_name == 'viz':       return self.display_img
        return None

    def _render(self):
        h, w = 96, 256
        img = np.zeros((h, w, 3), dtype=np.uint8)
        ok = self.client.connected
        cv2.circle(img, (14, 16), 6, (90, 220, 120) if ok else (60, 60, 70), -1)
        cv2.putText(img, f"{self.host}:{self.port}  {'LINK' if ok else '...'}",
                    (28, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 210), 1)
        cv2.putText(img, f"tx {self.client.tx}   rx {self.client.rx}   src#{self.src_id}",
                    (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 200, 230), 1)
        # last received token thumbnail (if square)
        if self.last_recv_vec is not None and len(self.last_recv_vec):
            d = len(self.last_recv_vec); side = int(np.sqrt(d))
            if side*side == d:
                g = self.last_recv_vec.reshape(side, side)
                g = (g - g.min())/(np.ptp(g)+1e-9)
                t = (np.clip(g, 0, 1)*255).astype(np.uint8)
                t = cv2.applyColorMap(cv2.resize(t, (40, 40), interpolation=cv2.INTER_NEAREST),
                                      cv2.COLORMAP_OCEAN)
                img[48:88, 200:240] = t
        cv2.putText(img, "spike == token (no weights cross the wire)",
                    (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (120, 140, 160), 1)
        self.display_img = img

    def get_display_image(self):
        return QtGui.QImage(self.display_img.data, 256, 96, 256*3,
                            QtGui.QImage.Format.Format_RGB888)

    def get_config_options(self):
        return [
            ("Relay Host", "host", self.host, "str"),
            ("Relay Port", "port", self.port, None),
            ("Source ID", "src_id", self.src_id, None),
            ("Broadcast Every (frames)", "broadcast_every", self.broadcast_every, None),
            ("Confidence Gate", "conf_gate", self.conf_gate, "float"),
        ]

    def set_config_options(self, options):
        reconnect = False
        if "host" in options and str(options["host"]) != self.host:
            self.host = str(options["host"]); reconnect = True
        if "port" in options and int(options["port"]) != self.port:
            self.port = int(options["port"]); reconnect = True
        if "src_id" in options: self.src_id = int(options["src_id"])
        if "broadcast_every" in options: self.broadcast_every = int(options["broadcast_every"])
        if "conf_gate" in options: self.conf_gate = float(options["conf_gate"])
        if reconnect:
            try: self.client.stop()
            except Exception: pass
            self.client = RelayClient(self.host, self.port, self.src_id).start()
