import cv2
import numpy as np
import os

# Configurationx
IMAGE_PATH = "/Users/jnoael/Mechatronics_Vision_2026/ladle/0_raw.png"
# Grids (Changes very little in practice but determines where points are placed)
MIN_GRID = 40
GRID_ROWS = MIN_GRID
GRID_COLS = MIN_GRID
# Folder for where images are placed, not needed for real-time model
OUTPUT_DIR = "ladle"
# Deciding hexes for what counts as a point, closer to #0000FF is closer to 0 and #FF0000 is closer to 1
HEX_COLOR_LOW = "#0000FF"   # 0 for heatmap (non-point reference)
HEX_COLOR_HIGH = "#FF0000"  # 1 for heatmap (is-point reference)
# Related to the hexcodes above, 0.5 means anything at or above 0.5 from a distance of HEX_COLOR_LOW to HEX_COLOR_HIGH
THRESHOLD = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Converts hexcode to (B, G, R), *note: cv2 uses (B, G, R) not (R, G, B)
def hex_to_bgr(hexcode):
    hexcode = hexcode.lstrip('#')
    r = int(hexcode[0:2], 16)
    g = int(hexcode[2:4], 16)
    b = int(hexcode[4:6], 16)
    return np.array([b, g, r], dtype=np.float32)

# Loads image
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load {path}")
    return img

# Overlays grid on the image, uses height of img to decide steps for rows and width for steps of cols
def overlay_grid(img, rows, cols, color=(0,0,0)):
    h, w = img.shape[:2]
    out = img.copy()

    for c in range(1, cols):
        x = int(c * w / cols)
        cv2.line(out, (x,0), (x,h), color, 1)

    for r in range(1, rows):
        y = int(r * h / rows)
        cv2.line(out, (0,y), (w,y), color, 1)

    return out

def compute_cell_averages(img, rows, cols):
    """
    Returns:
        block_image: image where each cell is filled with the avg color
        avg_colors:  rows x cols x 3 array of BGR means
        centers:     list of (x,y) centers per cell
    """
    h, w = img.shape[:2]
    avg_colors = np.zeros((rows, cols, 3), dtype=np.float32)
    block = np.zeros_like(img)
    centers = []

    for r in range(rows):
        y0 = int(r     * h / rows)
        y1 = int((r+1) * h / rows)
        for c in range(cols):
            x0 = int(c     * w / cols)
            x1 = int((c+1) * w / cols)

            cell = img[y0:y1, x0:x1]
            mean = cell.reshape(-1,3).mean(axis=0)
            avg_colors[r, c] = mean
            block[y0:y1, x0:x1] = mean.astype(np.uint8)

            xc = (x0 + x1)//2
            yc = (y0 + y1)//2
            centers.append((xc, yc))

    return block, avg_colors, np.array(centers)

def score_color_similarity(avg_colors, hex_low, hex_high):
    """
    Computes a 0–1 score for each cell based on similarity to two colors.
    Uses Euclidean distance in BGR space.

    Score = distance_to_low / (distance_to_low + distance_to_high)
    """
    c_low = hex_to_bgr(hex_low)
    c_high = hex_to_bgr(hex_high)

    # Convert avg_colors (rows x cols x 3) to (N x 3)
    rows, cols, _ = avg_colors.shape
    flat = avg_colors.reshape(-1, 3)

    d_low = np.linalg.norm(flat - c_low, axis=1)
    d_high = np.linalg.norm(flat - c_high, axis=1)

    score = d_low / (d_low + d_high + 1e-9)
    score = 1 - score  # ensure low=0, high=1
    return score.reshape(rows, cols)

def fit_line(points, method="reg_lin_reg"):
    """
    Fit y = m*x + b using numpy polyfit.
    Returns (m, b)
    """
    match method:
        case "lin_reg"|0: #Linear Regression
            xs = points[:,0]
            ys = points[:,1]
            m, b = np.polyfit(xs, ys, 1)
            return m, b
        case "svd"|1:
            pts = points.astype(np.float64)
            mean = pts.mean(axis=0)
            U, S, Vt = np.linalg.svd(pts - mean)
            direction = Vt[0] 
            px, py = mean
            dx, dy = direction

            if abs(dx) < 1e-9:
                return None
            m = dy / dx
            b = py - m * px
            return m, b

def draw_line_on_img(img, line_params, color=(0,255,0), thickness=2):
    m, b = line_params
    h, w = img.shape[:2]
    x0, x1 = 0, w-1
    y0 = int(m*x0 + b)
    y1 = int(m*x1 + b)
    out = img.copy()
    cv2.line(out, (x0,y0), (x1,y1), color, thickness)
    return out

def main():
    # 1. load img
    img = load_image(IMAGE_PATH)
    h, w = img.shape[:2]

    # 2. overlay grid
    grid_img = overlay_grid(img, GRID_ROWS, GRID_COLS)
    cv2.imwrite(f"{OUTPUT_DIR}/1_grid.jpg", grid_img)

    # 3. average color per cell
    block_img, avg_colors, centers = compute_cell_averages(img, GRID_ROWS, GRID_COLS)
    cv2.imwrite(f"{OUTPUT_DIR}/2_averaged.jpg", block_img)

    # 4. score cells from 0→1 using chosen hexcodes
    scores = score_color_similarity(avg_colors, HEX_COLOR_HIGH, HEX_COLOR_LOW)

    # 5. threshold
    flat_scores = scores.reshape(-1)
    mask = flat_scores >= THRESHOLD

    selected_points = centers[mask]

    # (visualization: draw points)
    pts_img = block_img.copy()
    for (x,y) in selected_points:
        cv2.circle(pts_img, (int(x),int(y)), 4, (0,0,0), -1)

    cv2.imwrite(f"{OUTPUT_DIR}/3_points.jpg", pts_img)

    for i in range(2):
        # 6. fit a line to selected points
        print(i)
        if len(selected_points) >= 2:
            m, b = fit_line(points=selected_points, method=i)
        else:
            raise RuntimeError("Not enough points above threshold to fit a line.")

        line_on_block = draw_line_on_img(pts_img, (m,b))
        cv2.imwrite(f"{OUTPUT_DIR}/4_fitted_line_{i}.jpg", line_on_block)

    # 7. show line over raw image
    line_on_raw = draw_line_on_img(img, (m,b))
    cv2.imwrite(f"{OUTPUT_DIR}/5_line_on_raw.jpg", line_on_raw)

main()