"""
token_relay_server.py  —  the field bus, over the wire
=======================================================
A minimal, dependency-free pub/sub hub for SpikeTokens. Every JSON line a peer
sends is rebroadcast to all OTHER peers. This is the shared field made into a
network: peers never address each other, they just broadcast and listen.

Run once on any machine both peers can reach:
    python token_relay_server.py --port 8765

Pure stdlib (socket + threading). Newline-delimited JSON. No deps so it drops
straight into the PerceptionLab environment.

PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.
Do not hype. Do not lie. Just show.
"""
import socket
import threading
import argparse


class Relay:
    def __init__(self, host="0.0.0.0", port=8765):
        self.host = host; self.port = port
        self.clients = []; self.lock = threading.Lock(); self.running = False
        self.sock = None; self.n_msgs = 0

    def start(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((self.host, self.port)); self.sock.listen(32)
        self.running = True
        threading.Thread(target=self._accept, daemon=True).start()
        return self

    def _accept(self):
        while self.running:
            try:
                conn, addr = self.sock.accept()
            except OSError:
                break
            with self.lock:
                self.clients.append(conn)
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    def _handle(self, conn):
        buf = b""
        while self.running:
            try:
                data = conn.recv(1 << 16)
            except OSError:
                break
            if not data:
                break
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                self.n_msgs += 1
                self._broadcast(line + b"\n", conn)
        with self.lock:
            if conn in self.clients:
                self.clients.remove(conn)
        try:
            conn.close()
        except OSError:
            pass

    def _broadcast(self, line, sender):
        with self.lock:
            targets = [c for c in self.clients if c is not sender]
        for c in targets:
            try:
                c.sendall(line)
            except OSError:
                pass

    def stop(self):
        self.running = False
        try:
            self.sock.close()
        except OSError:
            pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    r = Relay(args.host, args.port).start()
    print(f"token relay listening on {args.host}:{args.port}  (Ctrl-C to stop)")
    import time
    try:
        while True:
            time.sleep(2.0)
            print(f"  peers={len(r.clients)}  msgs_relayed={r.n_msgs}")
    except KeyboardInterrupt:
        r.stop()
