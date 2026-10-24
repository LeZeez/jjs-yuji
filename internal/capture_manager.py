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
        # only a request, the driver picks the closest mode it supports (OBS virtual cam is fixed by OBS's output size tho)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cam.set(cv2.CAP_PROP_FPS, fps)

        # set() can return True even if it got ignored, so the first frame is the real answer
        success, sample_frame = self.cam.read()
        if not success or sample_frame is None:
            raise RuntimeError(f"Camera {device_idx} failed to return a frame at startup.")
        real_h, real_w = sample_frame.shape[:2]
        if real_w != width or real_h != height:
            print(f"Camera rejected requested resolution! Using actual camera output for writer ({real_w}x{real_h}).")
            

        self.encoder = Encoder()
        self.video_stream = self.encoder.add_video(
            height=real_h,
            width=real_w,
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