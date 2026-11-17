import av
import cv2
import time
from contextlib import contextmanager


class DataRecorder:
    def __init__(self, device_idx: int, width: int, height: int, fps: float | int, video_path: str, device_fourcc: str = None):
        self.height = height
        self.width = width
        self.fps = fps
        self.video_path = video_path
        self.container = None
        self.stream = None

        self.cam = cv2.VideoCapture(device_idx, cv2.CAP_V4L2)
        if device_fourcc:
            self.cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*device_fourcc))
        # only a request, the driver picks the closest mode it supports (OBS virtual cam is fixed by OBS's output size tho)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cam.set(cv2.CAP_PROP_FPS, fps)

        # cam.set() can return True even if it got ignored, so the first frame is the real answer
        success, sample_frame = self.cam.read()
        if not success or sample_frame is None:
            raise RuntimeError(f"Camera {device_idx} failed to return a frame at startup.")

        # PyAV would silently resize a different size when saving
        real_height, real_width = sample_frame.shape[:2]
        if real_width != width or real_height != height:
            raise RuntimeError(f"Camera sends {real_width}x{real_height}, not {width}x{height}. "
                               "Set the same size in OBS and in the settings.")

    def read(self):
        success, frame = self.cam.read()
        if not success or frame is None:
            return None
        return frame  # (H, W, 3) BGR

    def measure_fps(self):
        # reads one second worth of frames, returns how long it really took (1.0 = camera matches fps)
        for _ in range(5):
            self.cam.read()  # warmup
        start = time.monotonic()
        for _ in range(round(self.fps)):
            self.cam.read()
        return time.monotonic() - start

    @contextmanager
    def open_file(self):
        self.container = av.open(self.video_path, "w")
        options = {
            "coder": "range_def", # ~15% smaller than the default coder, still lossless
            "threads": "2",
        }
        self.stream = self.container.add_stream("ffv1", rate=round(self.fps), options=options) # FFV1 is always lossless
        self.stream.width = self.width
        self.stream.height = self.height
        self.stream.pix_fmt = "bgr0" # FFV1 has no bgr24, bgr0 is BGR plus an unused byte
        try:
            yield
        finally:
            packets = self.stream.encode()  # no (frame) = flush the frames the encoder still holds
            for packet in packets:
                self.container.mux(packet)
            self.container.close()

    def write(self):
        # returns whether a frame was written, so the inputs are only logged for real frames
        bgr = self.read()
        if bgr is None:
            return False

        frame = av.VideoFrame.from_ndarray(bgr, format="bgr24")
        packets = self.stream.encode(frame) # the encoder can hold frames back, so 0 or more packets
        for packet in packets:
            self.container.mux(packet)
        return True

    def release(self):
        # release hardware handles clean when done
        self.cam.release()
