"""
relay_client.py  —  a non-blocking SpikeToken client
=====================================================
Background-thread TCP client: send() never blocks the caller (it queues), poll()
returns received tokens. Auto-reconnects. Pure stdlib so the PerceptionLab GUI
thread is never stalled by the socket.

Token schema (newline-delimited JSON):
    {"src": int, "payload": [float,...], "conf": float, "chi": int}

PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.
"""
import socket
import threading
import queue
import json
import time


class RelayClient:
    def __init__(self, host="127.0.0.1", port=8765, src=0):
        self.host = host; self.port = port; self.src = int(src)
        self.outq = queue.Queue(maxsize=256)
        self.inbox = []; self.lock = threading.Lock()
        self.running = False; self.connected = False
        self.tx = 0; self.rx = 0
        self.sock = None

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
                        self.sock.sendall((json.dumps(tok) + "\n").encode())
                        self.tx += 1
                    except OSError:
                        self.connected = False
            except OSError:
                self.connected = False
                time.sleep(1.0)

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
        tok = {"src": self.src, "payload": [float(x) for x in payload],
               "conf": float(conf), "chi": int(chi)}
        try:
            self.outq.put_nowait(tok)
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
