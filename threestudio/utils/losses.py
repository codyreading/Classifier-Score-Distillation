import torch
import torch.nn as nn

from threestudio.utils import transform, ops

def sketch_distance(points, density, mvp_mtx, distances, delta = 0.2, sigma = 2):
    # Compute distances
    proj = mvp_mtx[:, :3]
    points_img = transform.project_to_images(points=points, proj=proj)
    points_img = points_img.unsqueeze(2) # (B, N, 1, 2)
    point_distances = torch.nn.functional.grid_sample(input=distances, grid=points_img, padding_mode="border")
    point_distances = point_distances.mean(0).squeeze(0)

    # Convert to weight
    weight = ops.gaussian_weighted_distance(point_distances)

    # Convert to occupancy
    occ = 1 - torch.exp(-delta * density)
    occ = occ.clamp(min=0, max=1.1)
    occ_target = torch.zeros_like(occ)

    # Compute loss
    loss = ops.ce_pq_loss(occ, occ_target, weight=weight)
    loss = loss / occ.numel()
    return loss
