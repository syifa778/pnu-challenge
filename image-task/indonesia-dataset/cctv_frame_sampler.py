import cv2
from pathlib import Path
from tqdm import tqdm
import argparse

class FrameSampler:
    """
    Sample frames at a fixed time interval (e.g. every 1s, 0.5s)
    """

    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps

        print(f"[INFO] {self.fps:.2f} FPS | {self.duration:.1f}s")

    def sample_by_interval(self, interval_sec=1.0):
        step = max(1, int(self.fps * interval_sec))
        indices = list(range(0, self.total_frames, step))

        print(f"[INFO] Sampling every {interval_sec}s → {len(indices)} frames")
        return indices

    def extract_frames(self, indices, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        indices = set(indices)
        frame_idx = 0
        saved = 0

        pbar = tqdm(total=len(indices), desc="Extracting")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            if frame_idx in indices:
                out = output_dir / f"frame_{frame_idx:08d}.jpg"
                cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved += 1
                pbar.update(1)

            frame_idx += 1

        pbar.close()
        self.cap.release()
        print(f"[SUCCESS] Extracted {saved} frames")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--interval', type=float, default=1.0,
                        help='Sampling interval in seconds (e.g. 1, 0.5, 2)')
    args = parser.parse_args()

    sampler = FrameSampler(args.video)
    indices = sampler.sample_by_interval(interval_sec=args.interval)
    sampler.extract_frames(indices, args.output)

if __name__ == "__main__":
    main()
