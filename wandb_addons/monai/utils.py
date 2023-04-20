from typing import List, Union

import wandb
import torch

from monai.config import NdarrayTensor
from monai.transforms import rescale_array


def plot_2d_or_3d_image(
    image_key: str,
    data: Union[NdarrayTensor, List[NdarrayTensor]],
    index: int = 0,
    max_channels: int = 1,
    frame_dim: int = -3,
    max_frames: int = 24,
):
    data_index = data[index]
    # as the `d` data has no batch dim, reduce the spatial dim index if positive
    frame_dim = frame_dim - 1 if frame_dim > 0 else frame_dim

    viz_data = (
        data_index.detach().cpu().numpy()
        if isinstance(data_index, torch.Tensor)
        else data_index
    )

    print(viz_data.shape)

    # if viz_data.ndim == 2:
    #     viz_data = rescale_array(viz_data, 0, 1)
    #     wandb.log({image_key: wandb.Image(viz_data)}, commit=False)

    # elif viz_data.ndim == 3:
    #     if viz_data.shape[0] == 3 and max_channels == 3:  # RGB
    #         wandb.log({image_key: wandb.Image(viz_data)}, commit=False)
    #     else:
    #         wandb.log(
    #             {
    #                 image_key: [
    #                     wandb.Image(rescale_array(d2, 0, 1))
    #                     for d2 in viz_data[:max_channels]
    #                 ]
    #             },
    #             commit=False,
    #         )
