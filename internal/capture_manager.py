import cv2
from torchcodec.encoders import Encoder
import torch
class DataRecorder:
    def __init__(self, device_idx: int, width: int, height: int, fps: float | int, video_path: str, device_fourcc: str = None):
        self.height = height
        self.width = width
        self.fps = fps
        self.video_path = video_path
        
        self.cam = cv2.VideoCapture(device_idx)
        if device_fourcc:
            self.cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*device_fourcc))
        self.cam.set(3, width)
        self.cam.set(4, height)
        self.cam.set(5, fps)

        success, sample_frame = self.cam.read()
        if not success or sample_frame is None:
            raise RuntimeError(f"Camera {device_idx} failed to return a frame at startup.")
        real_h, real_w = sample_frame.shape[:2]
        if real_w != width or real_h != height:
            print("Camera rejected requested resolution! Using actual camera output for writer.")

        self.encoder = Encoder()
        self.video_stream = self.encoder.add_video(
            height=height,
            width=width,
            frame_rate=fps,
            codec="libx264",
            preset="veryfast",
            extra_options={"threads": "2"}
        )
    def read(self):
        success, frame = self.cam.read()
        if not success or frame is None:
            return None
        return frame

    def open_file(self):
        return self.encoder.open_file(self.video_path)
    
    def write(self):
        frame = self.read()
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tsr_frame = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
            self.video_stream.add_frames(tsr_frame)

    def release(self):
        # release hardware handles clean when done
        self.cam.release()