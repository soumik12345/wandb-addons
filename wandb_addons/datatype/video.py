import base64
import os

import wandb

HTML_VIDEO_FORMAT = """
<center>
    <video controls>
        <source src="data:video/mp4;base64,{encoded_string}" type="video/mp4">
    </video>
</center>
"""


def loggable_video(video: str):
    if isinstance(video, str):
        if os.path.isfile(video):
            with open(video, "rb") as video_file:
                encoded_string = base64.b64encode(video_file.read()).decode("utf-8")
    else:
        wandb.termwarn("Unable to log video", repeat=False)
    html_string = HTML_VIDEO_FORMAT.format(encoded_string=encoded_string)
    return wandb.Html(html_string)
