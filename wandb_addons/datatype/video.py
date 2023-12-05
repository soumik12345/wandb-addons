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
    """Function that logs a playable video with controls to contrast the default
    uncontrollable gif offered by [`wandb.Video`](https://docs.wandb.ai/ref/python/data-types/video).

    !!! example "Example WandB Run"
        [https://wandb.ai/geekyrakshit/test/runs/q0ni305v](https://wandb.ai/geekyrakshit/test/runs/q0ni305v)

    !!! example "Logging a video file"
        ```python
        import wandb
        from wandb_addons.datatype import loggable_video

        with wandb.init(project="test", entity="geekyrakshit"):
            wandb.log({"Test-Video": "video.mp4"})
        ```

    !!! example "Logging a list of numpy arrays corresponding to frames"
        ```python
        import numpy as np

        import wandb
        from wandb_addons.datatype import loggable_video

        with wandb.init(project="test", entity="geekyrakshit"):
            frames = [np.ones((256, 256, 3)) * 255] * 10 + [np.zeros((256, 256, 3))] * 10
            wandb.log({"Test-Video": loggable_video(frames)})
        ```

    Arguments:
        video (Union[str, List[np.array]]): The path to a video file or a list of
            numpy arrays of shape `(H, W, C)` corresponding to the frames of the video.
        video_format (str): Format of the video.
        fps (int): Frame-rate of the video, applicable only when logging list of
            numpy arrays.

    Returns:
        (Union[wandb.Html, None]): A `wandb.Html` object that can be passed to a WandB
            loggable dictionary.
    """
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
