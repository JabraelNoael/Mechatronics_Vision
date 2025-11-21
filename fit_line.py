import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

IMAGE_PATH = r"data/images/redladle.png"
MIN_GRID = 50
GRID_COLS = MIN_GRID
GRID_ROWS = MIN_GRID
WINDOW_SCALE = 1.0
OUTPUT_DIR = "ladle_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    global h, w
    h, w = img.shape[:2]
    if img is None:
        raise FileNotFoundError(f"Could not load image at {path}")
    return img

def overlay_grid(img, cols, rows, color=(0,0,0), thickness=1):
    img_grid = img.copy()
    # vertical lines
    for i in range(1, cols):
        x = int(i * w / cols)
        cv2.line(img_grid, (x, 0), (x, h-1), color, thickness)
    # horizontal lines
    for j in range(1, rows):
        y = int(j * h / rows)
        cv2.line(img_grid, (0, y), (w-1, y), color, thickness)
    return img_grid

def compute_cell_averages(img, cols, rows):
    avg_colors = np.zeros((rows, cols, 3), dtype=np.float32)
    block = np.zeros_like(img)
    for r in range(rows):
        y0 = int(round(r * h / rows))
        y1 = int(round((r+1) * h / rows))
        for c in range(cols):
            x0 = int(round(c * w / cols))
            x1 = int(round((c+1) * w / cols))
            cell = img[y0:y1, x0:x1]
            if cell.size == 0:
                continue
            mean_bgr = cell.reshape(-1, 3).mean(axis=0)
            avg_colors[r, c] = mean_bgr
            block[y0:y1, x0:x1] = mean_bgr.astype(np.uint8)
    return block, avg_colors

def redness_score_from_bgr(avg_colors):
    # avg_colors is rows x cols x 3 (B,G,R)
    B = avg_colors[..., 0].astype(np.float32)
    R = avg_colors[..., 2].astype(np.float32)
    eps = 1e-6
    score = R / (R + B + eps)
    score = np.clip(score, 0.0, 1.0)
    return score

def heatmap_from_scores(scores, target_shape):
    # normalize to 0..255
    norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    small = (norm * 255).astype(np.uint8)
    # use OpenCV colormap on a small array then resize
    cmap_small = cv2.applyColorMap(small, cv2.COLORMAP_JET) # BGR
    # scale up to target_shape
    heatmap = cv2.resize(cmap_small, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)
    return heatmap

def fit_line_to_points(pts):
    if len(pts) < 2:
        return None
    # Use numpy polyfit: fit y = m*x + b
    xs = pts[:, 0].astype(np.float32)
    ys = pts[:, 1].astype(np.float32)
    m, b = np.polyfit(xs, ys, 1)
    return (m, b)

def draw_fitted_line(img, line_params, color=(0,0,0), thickness=4):
    h,w = img.shape[:2]
    m,b = line_params
    # compute two points at left and right edges
    x0, x1 = 0, w-1
    y0 = int(round(m*x0 + b))
    y1 = int(round(m*x1 + b))
    cv2.line(img, (x0, y0), (x1, y1), color, thickness)

def centers_of_cells(avg_colors, img_shape):
    rows, cols = avg_colors.shape[:2]
    h, w = img_shape[:2]
    centers = []
    coords = []
    for r in range(rows):
        y0 = int(round(r * h / rows))
        y1 = int(round((r+1) * h / rows))
        yc = int((y0 + y1) / 2)
        for c in range(cols):
            x0 = int(round(c * w / cols))
            x1 = int(round((c+1) * w / cols))
            xc = int((x0 + x1) / 2)
            centers.append((xc, yc))
            coords.append((r,c))
    return np.array(centers), coords  # centers shape (rows*cols, 2)

def main():
    img = load_image(IMAGE_PATH)
    h, w = img.shape[:2]
    display_h, display_w = int(h * WINDOW_SCALE), int(w * WINDOW_SCALE)

    orig_with_grid = overlay_grid(img, GRID_COLS, GRID_ROWS, thickness=1)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "1_original_with_grid.jpg"), orig_with_grid)
    cv2.namedWindow("original_with_grid", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("original_with_grid", display_w, display_h)
    cv2.imshow("original_with_grid", orig_with_grid)

    averaged_img, avg_colors = compute_cell_averages(img, GRID_COLS, GRID_ROWS)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "2_averaged_blocks.jpg"), averaged_img)
    cv2.namedWindow("averaged_blocks", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("averaged_blocks", display_w, display_h)
    cv2.imshow("averaged_blocks", averaged_img)

    # compute centers for all cells
    centers, coords = centers_of_cells(avg_colors, img.shape)  # centers in image coordinates

    # Create window for interactive threshold + fitted line
    win_name = "selected_points_fit"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, display_w, display_h)

    # trackbar callback uses closure to access state
    state = {"threshold": 80}  # trackbar 0..100
    def on_trackbar(pos):
        state["threshold"] = pos
        thresh = pos / 100.0
        # determine which cells pass threshold
        score_flat = scores.flatten()
        selected_mask = score_flat > thresh
        sel_idxs = np.where(selected_mask)[0]
        # Build visualization: start from averaged_img copy
        vis = averaged_img.copy()
        # Draw small circles at selected centers and collect pts
        pts = []
        for idx in sel_idxs:
            (x,y) = centers[idx]
            pts.append((x,y))
            cv2.circle(vis, (int(x), int(y)), radius=6, color=(0,0,0), thickness=-1)  # black dot
            cv2.circle(vis, (int(x), int(y)), radius=3, color=(0,255,255), thickness=-1)  # small highlight
        # Fit line if we have >= 2 points
        if len(pts) >= 2:
            pts_arr = np.array(pts)
            line_params = fit_line_to_points(pts_arr)
            if line_params is not None:
                draw_fitted_line(vis, line_params, color=(0,0,0), thickness=2)
        # show original with grid + fitted line overlay too
        orig_vis = orig_with_grid.copy()
        if len(pts) >= 2 and line_params is not None:
            draw_fitted_line(orig_vis, line_params, color=(0,0,255), thickness=2)

        # compose combined display: left=orig_vis, right=vis
        left = cv2.resize(orig_vis, (int(display_w/2), display_h), interpolation=cv2.INTER_AREA)
        right = cv2.resize(vis, (int(display_w/2), display_h), interpolation=cv2.INTER_AREA)
        combined = np.hstack([left, right])

        # update heatmap colorbar with cutoff marker
        hm_and_cb = build_heatmap_and_colorbar_display(scores, (display_h, int(display_w/2)), thresh)
        # stack combined over heatmap for a 2-row display (if space)
        # Resize hm_and_cb to same width as combined
        hm_and_cb_resized = cv2.resize(hm_and_cb, (combined.shape[1], int(combined.shape[0]/2)), interpolation=cv2.INTER_AREA)
        # Stack vertically to make final image
        final_canvas = np.vstack([combined, hm_and_cb_resized])
        cv2.imshow(win_name, final_canvas)
        # also save snapshots for demonstration
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"selected_thresh_{pos:03d}.jpg"), final_canvas)

    # create trackbar
    cv2.createTrackbar('Redness threshold', win_name, state["threshold"], 100, on_trackbar)
    # call once to initialize
    on_trackbar(state["threshold"])

    print("Interactive windows opened.")
    print("Adjust the 'Redness threshold' slider to choose which grid cells count as 'points' (>= threshold).")
    print("Press ESC or 'q' to quit. Images saved to:", OUTPUT_DIR)

    # loop until user exits
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == 27 or key == ord('q'):
            break
        # trackbar callback will update display

    # on exit, save some final files
    cv2.imwrite(os.path.join(OUTPUT_DIR, "final_original_with_grid.jpg"), orig_with_grid)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "final_averaged_blocks.jpg"), averaged_img)
    print("Exiting, windows will close.")
    cv2.destroyAllWindows()

main()
