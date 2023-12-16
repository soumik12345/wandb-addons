import os
import random
from typing import Dict, Optional, Union

import numpy as np
import open3d as o3d
from PIL import Image

import wandb


class RGBDPointCloud(wandb.Object3D):
    """Format an RGB image and a depthmap such that it is logged as an interactive 3d point cloud.

    !!! example "Example WandB Run"
        [https://wandb.ai/geekyrakshit/test/runs/8ftwuuwf](https://wandb.ai/geekyrakshit/test/runs/8ftwuuwf)

    !!! example "Logging an RGB Image and a Depthmap as a Point Cloud"
        ```python
        from PIL import Image

        import wandb
        from wandb_addons.datatype import RGBDPointCloud

        with wandb.init(project="test"):
            rgb_image = Image.open("./docs/assets/sample_image.jpg")
            depth_image = Image.open("./docs/assets/sample_depth.png")
            wandb.log({"Test-RGBD": RGBDPointCloud(rgb_image, depth_image)})
        ```

    Arguments:
        rgb_image (Union[str, Image.Image, np.array]): The RGB image. Either a path to an image file, or a
            PIL Image, or a numpy array can be passed.
        depth_image (Union[str, Image.Image, np.array]): The Depthmap. Either a path to an image file, or a
            PIL Image, or a numpy array can be passed.
        camera_intrinsic_parameters (Optional[Dict[str, float]]): The camera intrinsic parameters as a
            dictionary. Defaults to `o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault` if not
            specified.
        sample_percentage (Optional[float]): The percentage of points to be sampled from the point cloud for
            visualization.
    """

    def __init__(
        self,
        rgb_image: Union[str, Image.Image, np.array],
        depth_image: Union[str, Image.Image, np.array],
        camera_intrinsic_parameters: Optional[Dict[str, float]] = None,
        sample_percentage: Optional[float] = None,
        **kwargs,
    ) -> None:
        self.camera_intrinsic_parameters = (
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
            if camera_intrinsic_parameters is None
            else camera_intrinsic_parameters
        )
        self.sample_percentage = sample_percentage
        rgb_image_numpy, point_cloud = self.create_point_cloud(rgb_image, depth_image)
        normalized_point_cloud = self.normalize_point_cloud(point_cloud)
        colored_point_cloud = self.get_colored_point_cloud(
            rgb_image_numpy, normalized_point_cloud
        )
        colored_point_cloud = (
            self.sample_points(colored_point_cloud)
            if self.sample_percentage is not None
            else colored_point_cloud
        )
        super().__init__(colored_point_cloud, **kwargs)

    def _get_images_as_numpy_arrays(
        self,
        rgb_image: Union[str, Image.Image, np.array],
        depth_image: Union[str, Image.Image, np.array],
    ):
        if isinstance(rgb_image, str) and os.path.isfile(rgb_image):
            rgb_image = Image.open(rgb_image)
        if isinstance(depth_image, str) and os.path.isfile(depth_image):
            depth_image = Image.open(depth_image)
        if isinstance(rgb_image, Image.Image):
            rgb_image = np.array(rgb_image)
        if isinstance(depth_image, Image.Image):
            depth_image = np.array(depth_image)
        assert (
            len(rgb_image.shape) == 3
        ), "Batched pair of RGB images and Depthmaps are not yet supported"
        assert rgb_image.shape[-1] == 3, "RGB image must have 3 channels"
        return rgb_image, depth_image

    def create_point_cloud(
        self,
        rgb_image: Union[str, Image.Image, np.array],
        depth_image: Union[str, Image.Image, np.array],
    ):
        rgb_image, depth_image = self._get_images_as_numpy_arrays(
            rgb_image, depth_image
        )
        rgb_image_numpy = rgb_image
        rgb_image = o3d.geometry.Image(rgb_image)
        depth_image = o3d.geometry.Image(depth_image)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_image, depth_image, convert_rgb_to_intensity=False
        )
        camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            self.camera_intrinsic_parameters
        )
        point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, camera_intrinsic
        )
        point_cloud.transform(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )
        return rgb_image_numpy, np.asarray(point_cloud.points)

    def normalize_point_cloud(self, point_cloud):
        min_values = np.min(point_cloud, axis=0)
        max_values = np.max(point_cloud, axis=0)
        range_values = max_values - min_values
        normalized_point_cloud = (point_cloud - min_values) / range_values
        return normalized_point_cloud

    def get_colored_point_cloud(self, rgb_image_numpy, normalized_point_cloud):
        rgb_image_numpy = rgb_image_numpy.reshape(-1, 3)
        colored_point_cloud = np.concatenate(
            (normalized_point_cloud, rgb_image_numpy), axis=-1
        )
        return colored_point_cloud

    def sample_points(self, colored_point_cloud):
        num_points = len(colored_point_cloud)
        num_sampled_points = int((num_points * self.sample_percentage) // 100)
        sampled_indices = random.sample(list(range(num_points)), num_sampled_points)
        colored_point_cloud = colored_point_cloud[sampled_indices]
        return colored_point_cloud
