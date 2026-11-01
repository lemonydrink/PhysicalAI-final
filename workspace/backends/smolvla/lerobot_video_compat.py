"""Direct PyAV reader for pinned LeRobot with torchvision versions without VideoReader.

Dataset layout is unchanged. This opt-in reader adapter avoids TorchCodec ABI
requirements; returned image tensors retain LeRobot's CHW float32 [0,1] contract.
"""
import av
import numpy as np
import torch


def decode(video_path, timestamps, tolerance_s, backend=None):
    with av.open(str(video_path)) as container:
        frames = [(float(frame.time), frame.to_ndarray(format='rgb24'))
                  for frame in container.decode(video=0)]
    times = np.asarray([time for time, _ in frames])
    selected = []
    for timestamp in timestamps:
        index = int(np.argmin(abs(times - float(timestamp))))
        if abs(times[index] - float(timestamp)) > tolerance_s:
            raise ValueError(f'No video frame within {tolerance_s}s of {timestamp}')
        selected.append(frames[index][1])
    return torch.from_numpy(np.stack(selected)).permute(0, 3, 1, 2).float() / 255


def enable():
    import lerobot.datasets.lerobot_dataset as dataset
    dataset.decode_video_frames = decode
