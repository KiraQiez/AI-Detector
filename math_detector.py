import numpy as np
from PIL import Image, ImageDraw


# --------------------------
# 1. Compute 16x16 texture grid 
# --------------------------
def compute_texture_grid(original_img, grid_size=16):

    # 1) Convert to grayscale
    gray = original_img.convert("L")
    arr = np.array(gray, dtype=np.float32)  # (H, W)

    H, W = arr.shape
    cell_h = H // grid_size
    cell_w = W // grid_size

    if cell_h == 0 or cell_w == 0:
        raise ValueError("Image too small for the chosen grid size.")

    # 2) Crop to fit full grid
    H_crop = cell_h * grid_size
    W_crop = cell_w * grid_size
    arr_cropped = arr[:H_crop, :W_crop]

    # 3) Init result grid
    std_grid = np.zeros((grid_size, grid_size), dtype=np.float32)

    # 4) Compute std per block
    for y in range(grid_size):
        for x in range(grid_size):
            block = arr_cropped[
                y * cell_h : (y + 1) * cell_h,
                x * cell_w : (x + 1) * cell_w
            ]
            std_grid[y, x] = block.std()

    # 5) Map std values to [0,1] 
    grid_vals = std_grid / 64.0
    grid_vals = np.clip(grid_vals, 0.0, 1.0)

    return grid_vals


# --------------------------
# 2. Get label from texture score
# --------------------------
def get_math_label(texture_score):
    if texture_score < 0.25:
        return "AI"
    elif texture_score > 0.45:
        return "Real"
    else:
        return "Uncertain"


# --------------------------
# 3. Create colored overlay grid
# --------------------------
def create_math_grid_overlay(original_img, grid_vals, overlay_alpha=0.5):

    grid_size = grid_vals.shape[0]

    bins = np.digitize(grid_vals, [0.3, 0.6])  # 0,1,2
    color_map = np.array([
        [0, 255, 0],     # green = smooth = low std = likely AI
        [255, 255, 0],   # yellow = medium std
        [255, 0, 0],     # red = high std = likely Real
    ], dtype=np.uint8)

    colored_small = color_map[bins]

    # Upscale to original size
    heatmap_img = Image.fromarray(colored_small).resize(
        original_img.size,
        resample=Image.NEAREST
    )

    # Blend overlay
    base = original_img.convert("RGBA")
    overlay = Image.blend(base, heatmap_img.convert("RGBA"), alpha=overlay_alpha)

    # Draw grid lines
    draw = ImageDraw.Draw(overlay)
    w, h = overlay.size
    cell_w = w / grid_size
    cell_h = h / grid_size
    grid_color = (0, 0, 0, 150)

    for i in range(grid_size + 1):
        draw.line([(i * cell_w, 0), (i * cell_w, h)], fill=grid_color)
        draw.line([(0, i * cell_h), (w, i * cell_h)], fill=grid_color)

    return overlay.convert("RGB")
