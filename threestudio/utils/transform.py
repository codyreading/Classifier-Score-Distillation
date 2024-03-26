import torch
import kornia
import einops

def project_to_images(points, proj):
    # Reshapes for batch projection
    points_img = kornia.geometry.conversions.convert_points_to_homogeneous(points)

    # Project points
    points_img = proj @ points_img.T # (B, 3, N)
    points_img = einops.rearrange(points_img, "B D N -> B N D") # (B, N, 3)
    points_img = kornia.geometry.conversions.convert_points_from_homogeneous(points_img)
    return points_img