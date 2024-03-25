import math
import random
import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

from threestudio import register
from threestudio.utils.base import Updateable
from threestudio.data.uncond import (
    RandomCameraDataModuleConfig,
    RandomCameraDataset,
    RandomCameraIterableDataset,
)
from threestudio.utils.config import parse_structured
from threestudio.utils.misc import get_rank
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *


@dataclass
class RandomMultiviewCameraDataModuleConfig(RandomCameraDataModuleConfig):
    relative_radius: bool = True
    n_view: int = 1
    zoom_range: Tuple[float, float] = (1.0, 1.0)
    sketch_poses: List = field(default_factory=list)
    sketch_fov: float = 49.1
    sketch_zoom: float = 1.0
    sketch_distance: float = 1.0
    sketch_perturb: float = 0.0

class RandomMultiviewCameraIterableDataset(RandomCameraIterableDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zoom_range = self.cfg.zoom_range

        self.sketch_elevations = []
        self.sketch_azimuths = []
        for pose in self.cfg.sketch_poses:
            self.sketch_elevations.append(pose["elevation"])
            self.sketch_azimuths.append(pose["azimuth"])

        self.num_sketches = len(self.sketch_elevations)
        self.sketch_elevations = torch.as_tensor(self.sketch_elevations, dtype=torch.float32)
        self.sketch_azimuths = torch.as_tensor(self.sketch_azimuths, dtype=torch.float32)
        self.sketch_fovs = torch.as_tensor(self.cfg.sketch_fov, dtype=torch.float32).repeat(self.num_sketches)
        self.sketch_zooms = torch.as_tensor(self.cfg.sketch_zoom, dtype=torch.float32).repeat(self.num_sketches)
        self.sketch_distances = torch.as_tensor(self.cfg.sketch_distance, dtype=torch.float32).repeat(self.num_sketches)
        self.sketch_perturb = torch.as_tensor(self.cfg.sketch_perturb, dtype=torch.float32).repeat((self.num_sketches, 3))

    def collate(self, batch) -> Dict[str, Any]:
        assert self.batch_size % self.cfg.n_view == 0, f"batch_size ({self.batch_size}) must be dividable by n_view ({self.cfg.n_view})!"
        real_batch_size = self.batch_size // self.cfg.n_view

        # sample elevation angles
        elevation_deg: Float[Tensor, "B"]
        elevation: Float[Tensor, "B"]
        if random.random() < 0.5:
            # sample elevation angles uniformly with a probability 0.5 (biased towards poles)
            elevation_deg = (
                torch.rand(real_batch_size)
                * (self.elevation_range[1] - self.elevation_range[0])
                + self.elevation_range[0]
            ).repeat_interleave(self.cfg.n_view, dim=0)
            elevation = elevation_deg * math.pi / 180
        else:
            # otherwise sample uniformly on sphere
            elevation_range_percent = [
                (self.elevation_range[0] + 90.0) / 180.0,
                (self.elevation_range[1] + 90.0) / 180.0,
            ]
            # inverse transform sampling
            elevation = torch.asin(
                2
                * (
                    torch.rand(real_batch_size)
                    * (elevation_range_percent[1] - elevation_range_percent[0])
                    + elevation_range_percent[0]
                )
                - 1.0
            ).repeat_interleave(self.cfg.n_view, dim=0)
            elevation_deg = elevation / math.pi * 180.0

        # Add elevations from sketches
        sketch_elevations_deg = self.sketch_elevations
        sketch_elevations_rad = torch.deg2rad(sketch_elevations_deg)
        elevation_deg = torch.cat((elevation_deg, sketch_elevations_deg))
        elevation = torch.cat((elevation, sketch_elevations_rad))


        # sample azimuth angles from a uniform distribution bounded by azimuth_range
        azimuth_deg: Float[Tensor, "B"]
        # ensures sampled azimuth angles in a batch cover the whole range
        azimuth_deg = (
            torch.rand(real_batch_size).reshape(-1,1) + torch.arange(self.cfg.n_view).reshape(1,-1)
        ).reshape(-1) / self.cfg.n_view * (
            self.azimuth_range[1] - self.azimuth_range[0]
        ) + self.azimuth_range[
            0
        ]
        azimuth_deg = torch.cat((azimuth_deg, self.sketch_azimuths))
        azimuth = azimuth_deg * math.pi / 180

        ######## Different from original ########
        # sample fovs from a uniform distribution bounded by fov_range
        fovy_deg: Float[Tensor, "B"] = (
            torch.rand(real_batch_size)
            * (self.fovy_range[1] - self.fovy_range[0])
            + self.fovy_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)
        fovy_deg = torch.cat((fovy_deg, self.sketch_fovs))
        fovy = fovy_deg * math.pi / 180

        # sample distances from a uniform distribution bounded by distance_range
        camera_distances: Float[Tensor, "B"] = (
            torch.rand(real_batch_size)
            * (self.camera_distance_range[1] - self.camera_distance_range[0])
            + self.camera_distance_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)
        camera_distances = torch.cat((camera_distances, self.sketch_distances))
        if self.cfg.relative_radius:
            scale = 1 / torch.tan(0.5 * fovy)
            camera_distances = scale * camera_distances

        # zoom in by decreasing fov after camera distance is fixed
        zoom: Float[Tensor, "B"] = (
            torch.rand(real_batch_size)
            * (self.zoom_range[1] - self.zoom_range[0])
            + self.zoom_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)
        zoom = torch.cat((zoom, self.sketch_zooms))
        fovy = fovy * zoom
        fovy_deg = fovy_deg * zoom
        ###########################################

        # convert spherical coordinates to cartesian coordinates
        # right hand coordinate system, x back, y right, z up
        # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
        camera_positions: Float[Tensor, "B 3"] = torch.stack(
            [
                camera_distances * torch.cos(elevation) * torch.cos(azimuth),
                camera_distances * torch.cos(elevation) * torch.sin(azimuth),
                camera_distances * torch.sin(elevation),
            ],
            dim=-1,
        )

        # default scene center at origin
        center: Float[Tensor, "B 3"] = torch.zeros_like(camera_positions)
        # default camera up direction as +z
        up: Float[Tensor, "B 3"] = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
            None, :
        ].repeat(self.batch_size + self.num_sketches, 1)

        # sample camera perturbations from a uniform distribution [-camera_perturb, camera_perturb]
        camera_perturb: Float[Tensor, "B 3"] = (
            torch.rand(real_batch_size, 3) * 2 * self.cfg.camera_perturb
            - self.cfg.camera_perturb
        ).repeat_interleave(self.cfg.n_view, dim=0)
        camera_perturb = torch.cat((camera_perturb, self.sketch_perturb))
        camera_positions = camera_positions + camera_perturb

        # sample center perturbations from a normal distribution with mean 0 and std center_perturb
        center_perturb: Float[Tensor, "B 3"] = (
            torch.randn(real_batch_size, 3) * self.cfg.center_perturb
        ).repeat_interleave(self.cfg.n_view, dim=0)
        center_perturb = torch.cat((center_perturb, self.sketch_perturb))
        center = center + center_perturb
        # sample up perturbations from a normal distribution with mean 0 and std up_perturb
        up_perturb: Float[Tensor, "B 3"] = (
            torch.randn(real_batch_size, 3) * self.cfg.up_perturb
        ).repeat_interleave(self.cfg.n_view, dim=0)
        up_perturb = torch.cat((up_perturb, self.sketch_perturb))
        up = up + up_perturb

        # sample light distance from a uniform distribution bounded by light_distance_range
        light_distances: Float[Tensor, "B"] = (
            torch.rand(real_batch_size)
            * (self.cfg.light_distance_range[1] - self.cfg.light_distance_range[0])
            + self.cfg.light_distance_range[0]
        ).repeat_interleave(self.cfg.n_view, dim=0)

        sketch_light_distances = (
            torch.rand(1)
            * (self.cfg.light_distance_range[1] - self.cfg.light_distance_range[0])
            + self.cfg.light_distance_range[0]
        ).repeat_interleave(self.num_sketches, dim=0)
        light_distances = torch.cat((light_distances, sketch_light_distances))

        if self.cfg.light_sample_strategy == "dreamfusion":
            # sample light direction from a normal distribution with mean camera_position and std light_position_perturb
            light_perturb = torch.randn(real_batch_size, 3).repeat_interleave(self.cfg.n_view, dim=0)
            sketch_light_perturb = torch.randn(1, 3).repeat_interleave(self.num_sketches, dim=0)
            light_perturb = torch.cat((light_perturb, sketch_light_perturb))

            light_direction: Float[Tensor, "B 3"] = F.normalize(
                camera_positions
                + light_perturb * self.cfg.light_position_perturb,
                dim=-1,
            )
            # get light position by scaling light direction by light distance
            light_positions: Float[Tensor, "B 3"] = (
                light_direction * light_distances[:, None]
            )
        elif self.cfg.light_sample_strategy == "magic3d":
            # sample light direction within restricted angle range (pi/3)
            local_z = F.normalize(camera_positions, dim=-1)
            local_x = F.normalize(
                torch.stack(
                    [local_z[:, 1], -local_z[:, 0], torch.zeros_like(local_z[:, 0])],
                    dim=-1,
                ),
                dim=-1,
            )
            local_y = F.normalize(torch.cross(local_z, local_x, dim=-1), dim=-1)
            rot = torch.stack([local_x, local_y, local_z], dim=-1)
            light_azimuth = (
                torch.rand(real_batch_size) * math.pi - 2 * math.pi
            ).repeat_interleave(self.cfg.n_view, dim=0)  # [-pi, pi]
            light_elevation = (
                torch.rand(real_batch_size) * math.pi / 3 + math.pi / 6
            ).repeat_interleave(self.cfg.n_view, dim=0)  # [pi/6, pi/2]
            light_positions_local = torch.stack(
                [
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.cos(light_azimuth),
                    light_distances
                    * torch.cos(light_elevation)
                    * torch.sin(light_azimuth),
                    light_distances * torch.sin(light_elevation),
                ],
                dim=-1,
            )
            light_positions = (rot @ light_positions_local[:, :, None])[:, :, 0]
        else:
            raise ValueError(
                f"Unknown light sample strategy: {self.cfg.light_sample_strategy}"
            )

        lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
        right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
        up = F.normalize(torch.cross(right, lookat), dim=-1)
        c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
            [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
            dim=-1,
        )
        c2w: Float[Tensor, "B 4 4"] = torch.cat(
            [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
        )
        c2w[:, 3, 3] = 1.0

        # get directions by dividing directions_unit_focal by focal length
        focal_length: Float[Tensor, "B"] = 0.5 * self.height / torch.tan(0.5 * fovy)
        directions: Float[Tensor, "B H W 3"] = self.directions_unit_focal[
            None, :, :, :
        ].repeat(self.batch_size+self.num_sketches, 1, 1, 1)
        directions[:, :, :, :2] = (
            directions[:, :, :, :2] / focal_length[:, None, None, None]
        )

        # Importance note: the returned rays_d MUST be normalized!
        rays_o, rays_d = get_rays(directions, c2w, keepdim=True)

        proj_mtx: Float[Tensor, "B 4 4"] = get_projection_matrix(
            fovy, self.width / self.height, 0.1, 1000.0
        )  # FIXME: hard-coded near and far
        mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

        return {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "mvp_mtx": mvp_mtx,
            "camera_positions": camera_positions,
            "c2w": c2w,
            "light_positions": light_positions,
            "elevation": elevation_deg,
            "azimuth": azimuth_deg,
            "camera_distances": camera_distances,
            "height": self.height,
            "width": self.width,
            "fovy": fovy_deg,
            "num_sketches": self.num_sketches
        }


@register("random-multiview-camera-datamodule")
class RandomMultiviewCameraDataModule(pl.LightningDataModule):
    cfg: RandomMultiviewCameraDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(RandomMultiviewCameraDataModuleConfig, cfg)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = RandomMultiviewCameraIterableDataset(self.cfg)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = RandomCameraDataset(self.cfg, "val")
        if stage in [None, "test", "predict"]:
            self.test_dataset = RandomCameraDataset(self.cfg, "test")

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            # very important to disable multi-processing if you want to change self attributes at runtime!
            # (for example setting self.width and self.height in update_step)
            num_workers=0,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset, batch_size=1, collate_fn=self.val_dataset.collate
        )
        # return self.general_loader(self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )
