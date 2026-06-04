# -*- coding: utf-8 -*-
import argparse
import socket
import threading
import time
import sys

HTTP_OK = b"HTTP/1.1 200 OK\r\nServer: VideoStream\r\nContent-Type: multipart/x-mixed-replace; boundary=jpg\r\n\r\n"
BOUNDARY = b"--jpg\r\nContent-Type: image/jpeg\r\n\r\n"


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print("[%s] [Video] %s" % (ts, msg))


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"


class Streamer:
    def __init__(self, device=0, width=640, height=480, fps=15, port=8080):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.port = port
        self.running = False
        self.capture = None
        self.frame = None
        self.lock = threading.Lock()
        self.server = None

    def start(self):
        import cv2
        log("Opening camera device %d" % self.device)

        # Try different backends
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(self.device, backend)
            if cap.isOpened():
                self.capture = cap
                log("Camera opened with backend %d" % backend)
                break

        if self.capture is None or not self.capture.isOpened():
            log("Failed to open camera device %d" % self.device)
            return False

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.capture.set(cv2.CAP_PROP_FPS, self.fps)

        w = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        f = int(self.capture.get(cv2.CAP_PROP_FPS))
        log("Camera: %dx%d @ %dfps" % (w, h, f))

        self.running = True
        threading.Thread(target=self._capture_loop).start()
        self._start_http()
        return True

    def _capture_loop(self):
        import cv2
        frame_count = 0
        while self.running and self.capture and self.capture.isOpened():
            ok, frame = self.capture.read()
            if ok and frame is not None:
                with self.lock:
                    self.frame = frame
                frame_count += 1
                if frame_count % 100 == 0:
                    log("Captured %d frames, shape=%s" % (frame_count, frame.shape if hasattr(frame, 'shape') else 'no shape'))
            time.sleep(1.0 / self.fps)

    def _start_http(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("0.0.0.0", self.port))
        self.server.listen(5)
        log("HTTP server listening on port %d" % self.port)
        threading.Thread(target=self._accept_loop).start()

    def _accept_loop(self):
        while self.running:
            try:
                client, addr = self.server.accept()
                log("Client connected from %s" % str(addr))
                threading.Thread(target=self._handle, args=(client,)).start()
            except Exception as e:
                if self.running:
                    pass

    def _handle(self, client):
        try:
            # Recv request
            data = b""
            while b"\r\n\r\n" not in data:
                chunk = client.recv(1024)
                if not chunk:
                    log("Client %s sent no data, returning" % str(client.getpeername()))
                    return
                data += chunk

            log("Client %s request received, sending HTTP OK" % str(client.getpeername()))
            client.sendall(HTTP_OK)

            sent_count = 0
            while self.running:
                with self.lock:
                    if self.frame is None:
                        time.sleep(0.01)
                        continue
                    frame = self.frame.copy()

                import cv2
                ret, jpg = cv2.imencode(".jpg", frame)
                if not ret:
                    log("JPEG encode failed for frame")
                    continue

                jpg_bytes = jpg.tobytes()
                header = BOUNDARY + ("Content-Length: %d\r\n\r\n" % len(jpg_bytes)).encode() + jpg_bytes + b"\r\n"
                try:
                    client.sendall(header)
                    sent_count += 1
                    if sent_count % 100 == 0:
                        log("Client %s: sent %d frames, %d bytes/frame" % (str(client.getpeername()), sent_count, len(jpg_bytes)))
                except Exception as e:
                    log("Client %s send error: %s" % (str(client.getpeername()), e))
                    break
        except Exception as e:
            log("Client %s exception: %s" % (str(client.getpeername()), e))
        finally:
            try:
                client.close()
            except:
                pass

    def stop(self):
        log("Stopping...")
        self.running = False
        if self.capture:
            self.capture.release()
        if self.server:
            self.server.close()
        log("Stopped")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Video MJPEG Streamer")
    parser.add_argument("--device", type=int, default=0, help="Camera index")
    parser.add_argument("--width", type=int, default=640, help="Width")
    parser.add_argument("--height", type=int, default=480, help="Height")
    parser.add_argument("--fps", type=int, default=15, help="FPS")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port")
    args = parser.parse_args()

    print("=" * 40)
    print("  Video MJPEG Streamer")
    print("=" * 40)
    print("Device: %d" % args.device)
    print("Resolution: %dx%d" % (args.width, args.height))
    print("FPS: %d" % args.fps)
    print("Port: %d" % args.port)
    print("")

    streamer = Streamer(
        device=args.device,
        width=args.width,
        height=args.height,
        fps=args.fps,
        port=args.port
    )

    if not streamer.start():
        log("Failed to start")
        return

    ip = get_local_ip()
    print("=" * 40)
    print("Stream URL: http://%s:%d/stream" % (ip, args.port))
    print("Press Ctrl+C to stop")
    print("=" * 40)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        streamer.stop()


if __name__ == "__main__":
    main()
