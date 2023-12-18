import os
from typing import Optional, Union

import laspy as lp
import numpy as np
import open3d as o3d

import wandb


class VoxelizedPointCloud(wandb.Object3D):
    """Voxelizes a high-resolution point-cloud and format as a wandb-loggable 3D mesh.

    !!! example "Example WandB Run"
        Logging the voxelized mesh of
        [this point cloud](https://drive.google.com/file/d/1Zr1y8BSYRHBKxvs_nUXo2LSQr2i5ulgj/view?usp=sharing)
        takes up a memort of ~29 MB of space on the wandb run, whereas logging the raw
        point cloud takes up ~700 MB of space.

        [https://wandb.ai/geekyrakshit/test/runs/w2nrw85q](https://wandb.ai/geekyrakshit/test/runs/w2nrw85q)

    === "From Laspy File"

        ```python
        import wandb
        from wandb_addons.datatype import VoxelizedPointCloud

        with wandb.init(project="test"):
            wandb.log(
                {"Test-Point-Cloud": VoxelizedPointCloud("2021_heerlen_table.las")}
            )
        ```

    === "From Numpy Array"

        ```python
        import laspy as lp
        import numpy as np

        import wandb
        from wandb_addons.datatype import VoxelizedPointCloud

        point_cloud = lp.read("2021_heerlen_table.las")
        numpy_point_cloud = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z)).transpose()
        numpy_color_cloud = (
            np.vstack((point_cloud.red, point_cloud.green, point_cloud.blue)).transpose()
            / 65535
        )

        with wandb.init(project="test"):
            wandb.log(
                {"Test-Point-Cloud": VoxelizedPointCloud(numpy_point_cloud, numpy_color_cloud)}
            )

        ```

    !!! Info "Reference"
        [How to Automate Voxel Modelling of 3D Point Cloud with Python](https://towardsdatascience.com/how-to-automate-voxel-modelling-of-3d-point-cloud-with-python-459f4d43a227)

    Arguments:
        point_cloud ( Optional[Union[str, np.array]]): The point cloud. Either a path to
            a laspy file, or a numpy array of shape `(N, 3)` where `N` is the number of
            points.
        colors (Optional[np.array]): The colors of the point cloud. A numpy array of
            shape `(N, 3)` consisting of color values corresponsing to the point cloud
            in the range `[0, 1]`. This is only necessary to be specified for point
            clouds as numpy arrays.
        voxel_size_percentage (Optional[float]): The size of each voxel as a percentage
            of the maximum edge of the point cloud.
        voxel_precision (Optional[int]): The precision of the voxel size.
    """

    def __init__(
        self,
        point_cloud: Optional[Union[str, np.array]] = None,
        colors: Optional[np.array] = None,
        voxel_size_percentage: Optional[float] = 0.5,
        voxel_precision: Optional[int] = 4,
        **kwargs
    ) -> None:
        o3d_point_cloud = self.build_open3d_point_cloud(point_cloud, colors)
        voxel_mesh = self.voxelize_point_cloud(
            o3d_point_cloud, voxel_size_percentage, voxel_precision
        )
        voxel_file = os.path.join(wandb.run.dir, "voxelized_point_cloud.glb")
        o3d.io.write_triangle_mesh(voxel_file, voxel_mesh)
        super().__init__(open(voxel_file), **kwargs)

    def build_open3d_point_cloud(self, point_cloud, colors):
        if isinstance(point_cloud, str):
            point_cloud = lp.read(point_cloud)
            numpy_point_cloud = np.vstack(
                (point_cloud.x, point_cloud.y, point_cloud.z)
            ).transpose()
            numpy_color_cloud = (
                np.vstack(
                    (point_cloud.red, point_cloud.green, point_cloud.blue)
                ).transpose()
                / 65535
            )
            o3d_point_cloud = o3d.geometry.PointCloud()
            o3d_point_cloud.points = o3d.utility.Vector3dVector(numpy_point_cloud)
            o3d_point_cloud.colors = o3d.utility.Vector3dVector(numpy_color_cloud)
            return o3d_point_cloud
        elif isinstance(point_cloud, np.ndarray):
            o3d_point_cloud = o3d.geometry.PointCloud()
            o3d_point_cloud.points = o3d.utility.Vector3dVector(point_cloud)
            if colors is not None:
                o3d_point_cloud.colors = o3d.utility.Vector3dVector(colors)
            return o3d_point_cloud

    def voxelize_point_cloud(
        self, o3d_point_cloud, voxel_size_percentage, voxel_precision
    ):
        voxel_size = max(
            o3d_point_cloud.get_max_bound() - o3d_point_cloud.get_min_bound()
        ) * (voxel_size_percentage / 100.0)
        voxel_size = round(voxel_size, voxel_precision)
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
            o3d_point_cloud, voxel_size=voxel_size
        )
        voxels = voxel_grid.get_voxels()
        voxel_mesh = o3d.geometry.TriangleMesh()
        for v in voxels:
            cube = o3d.geometry.TriangleMesh.create_box(width=1, height=1, depth=1)
            cube.paint_uniform_color(v.color)
            cube.translate(v.grid_index, relative=False)
            voxel_mesh += cube
        voxel_mesh.translate([0.5, 0.5, 0.5], relative=True)
        voxel_mesh.scale(voxel_size, [0, 0, 0])
        voxel_mesh.merge_close_vertices(1e-7)
        voxel_mesh.translate(voxel_grid.origin, relative=True)
        return voxel_mesh
