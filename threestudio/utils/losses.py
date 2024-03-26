import torch
import torch.nn as nn

from threestudio.utils import transform

def sketch_distance(points, density, mvp_mtx, distances):
    proj = mvp_mtx[:, :3]
    points_img = transform.project_to_images(points=points, proj=proj)
    points_img = points_img.unsqueeze(2) # (B, N, 1, 2)
    distances = torch.nn.functional.grid_sample(input=distances, grid=points_img, padding_mode="border")
    distances = distances.mean(0).squeeze(0)
    breakpoint()
