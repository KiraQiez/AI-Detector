# math_detector.py
import numpy as np
from PIL import Image, ImageDraw


# --------------------------
# 1. Compute 16x16 texture grid (edge-based)
# --------------------------
def compute_texture_grid(original_img, grid_size=16):
    gray = original_img.convert("L")
    arr = np.array(gray, dtype=np.float32) / 255.0

    H, W = arr.shape
    cell_h = H // grid_size
    cell_w = W // grid_size

    # Edge strength using gradients
    gx, gy = np.gradient(arr)
    edge_mag = np.sqrt(gx**2 + gy**2)

    # Crop to fit grid
    H_crop = cell_h * grid_size
    W_crop = cell_w * grid_size
    cropped = edge_mag[:H_crop, :W_crop]

    # Reshape into blocks
    blocks = cropped.reshape(grid_size, cell_h, grid_size, cell_w)

    # Mean edge magnitude per block
    edge_grid = blocks.mean(axis=(1, 3))

    # Normalize 0–1
    e_min = edge_grid.min()
    e_max = edge_grid.max()
    grid_vals = (edge_grid - e_min) / (e_max - e_min + 1e-8)

    return grid_vals


# --------------------------
# 2. Label for math detector
# --------------------------
def get_math_label(texture_score):
    if texture_score < 0.12:
        return "Likely AI (smooth / low edges)"
    elif texture_score > 0.22:
        return "Likely Real (high detail / edges)"
    else:
        return "Uncertain (medium texture)"


# --------------------------
# 3. Create colored 16×16 grid overlay
# --------------------------
def create_math_grid_overlay(original_img, grid_vals, overlay_alpha=0.5):
    grid_size = grid_vals.shape[0]

    # 3 color quantization
    bins = np.digitize(grid_vals, [0.4, 0.7])
    color_map = np.array(
        [
            [0, 255, 0],      # green
            [255, 255, 0],    # yellow
            [255, 0, 0],      # red
        ], dtype=np.uint8
    )

    colored_small = color_map[bins]

    # Upscale
    heatmap_img = Image.fromarray(colored_small).resize(
        original_img.size, resample=Image.NEAREST
    )

    # Blend
    base = original_img.convert("RGBA")
    overlay = Image.blend(base, heatmap_img.convert("RGBA"), alpha=overlay_alpha)

    # Draw grid
    draw = ImageDraw.Draw(overlay)
    w, h = overlay.size
    cell_w = w / grid_size
    cell_h = h / grid_size
    grid_color = (0, 0, 0, 160)

    for i in range(grid_size + 1):
        draw.line([(i * cell_w, 0), (i * cell_w, h)], fill=grid_color)
        draw.line([(0, i * cell_h), (w, i * cell_h)], fill=grid_color)

    return overlay.convert("RGB")
