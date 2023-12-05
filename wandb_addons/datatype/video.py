import base64
import os
import tempfile
from typing import List, Union

import imageio
import numpy as np

import wandb

HTML_VIDEO_FORMAT = """
<center>
    <video controls>
        <source src="data:video/{format};base64,{encoded_string}" type="video/{format}">
    </video>
</center>
"""


def create_video_from_np_arrays(frames: List[np.array], fps: int) -> str:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, "output.mp4")
        writer = imageio.get_writer(temp_file_path, fps=fps)
        for frame in frames:
            writer.append_data((frame).astype(np.uint8))
        writer.close()
        with open(temp_file_path, "rb") as video_file:
            encoded_string = base64.b64encode(video_file.read()).decode("utf-8")
    return encoded_string


def loggable_video(
    video: Union[str, List[np.array]], video_format: str = "mp4", fps: int = 30
) -> Union[wandb.Html, None]:
    if isinstance(video, str):
        if os.path.isfile(video):
            with open(video, "rb") as video_file:
                encoded_string = base64.b64encode(video_file.read()).decode("utf-8")
    elif isinstance(video, list):
        encoded_string = create_video_from_np_arrays(video, fps)
    else:
        wandb.termwarn("Unable to log video", repeat=False)
        return
    html_string = HTML_VIDEO_FORMAT.format(
        encoded_string=encoded_string, format=video_format
    )
    return wandb.Html(html_string)
