import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import RegularGridInterpolator


def compute_gst_gpu(
    data: np.ndarray,
    rho: float = 3.0,
    progress_callback=None,
    log_callback=None,
) -> tuple[np.ndarray, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if log_callback:
        log_callback(f"[GST] Device selected: {device.type}")

    if progress_callback:
        progress_callback(5)

    with torch.inference_mode():
        img = torch.from_numpy(np.asarray(data, dtype=np.float32, order="C")).to(device).unsqueeze(0).unsqueeze(0)
        if progress_callback:
            progress_callback(20)

        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device).view(1, 1, 3, 3)

        grad_x = F.conv2d(img, sobel_x, padding=1)
        grad_y = F.conv2d(img, sobel_y, padding=1)
        if progress_callback:
            progress_callback(40)

        j_xx = grad_x * grad_x
        j_yy = grad_y * grad_y
        j_xy = grad_x * grad_y

        kernel_size = max(3, int(rho * 3))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = torch.ones((1, 1, kernel_size, kernel_size), device=device) / float(kernel_size * kernel_size)

        j_xx = F.conv2d(j_xx, kernel, padding=kernel_size // 2)
        j_yy = F.conv2d(j_yy, kernel, padding=kernel_size // 2)
        j_xy = F.conv2d(j_xy, kernel, padding=kernel_size // 2)
        if progress_callback:
            progress_callback(65)

        trace = j_xx + j_yy
        det = j_xx * j_yy - j_xy * j_xy
        disc = torch.sqrt(torch.clamp((trace * 0.5) ** 2 - det, min=0))

        lam1 = (trace * 0.5) + disc
        lam2 = (trace * 0.5) - disc
        if progress_callback:
            progress_callback(85)

        magnitude = torch.sqrt(torch.clamp(lam1 - lam2, min=0)).squeeze().cpu().numpy()
        orientation = (0.5 * torch.atan2(2 * j_xy, j_xx - j_yy)).squeeze().cpu().numpy()
        if progress_callback:
            progress_callback(100)
        return magnitude, orientation


def rk4_trace(orientation_field: np.ndarray, seeds: np.ndarray, step: float = 0.5, iters: int = 200) -> list[np.ndarray]:
    h, w = orientation_field.shape
    x_coords = np.arange(w, dtype=np.float32)
    y_coords = np.arange(h, dtype=np.float32)
    interp = RegularGridInterpolator((y_coords, x_coords), orientation_field, bounds_error=False, fill_value=0.0)

    results: list[np.ndarray] = []
    for sx, sy in seeds:
        cx, cy = float(sx), float(sy)
        line = [[cx, cy]]
        for _ in range(iters):
            theta1 = float(interp((cy, cx)))
            v1x, v1y = np.cos(theta1 + np.pi / 2.0), np.sin(theta1 + np.pi / 2.0)

            cx2, cy2 = cx + v1x * step * 0.5, cy + v1y * step * 0.5
            theta2 = float(interp((cy2, cx2)))
            v2x, v2y = np.cos(theta2 + np.pi / 2.0), np.sin(theta2 + np.pi / 2.0)

            cx3, cy3 = cx + v2x * step * 0.5, cy + v2y * step * 0.5
            theta3 = float(interp((cy3, cx3)))
            v3x, v3y = np.cos(theta3 + np.pi / 2.0), np.sin(theta3 + np.pi / 2.0)

            cx4, cy4 = cx + v3x * step, cy + v3y * step
            theta4 = float(interp((cy4, cx4)))
            v4x, v4y = np.cos(theta4 + np.pi / 2.0), np.sin(theta4 + np.pi / 2.0)

            cx += (v1x + 2 * v2x + 2 * v3x + v4x) * step / 6.0
            cy += (v1y + 2 * v2y + 2 * v3y + v4y) * step / 6.0

            if not (0 <= cx < w and 0 <= cy < h):
                break
            line.append([cx, cy])

        results.append(np.asarray(line, dtype=np.float32))
    return results


def rk4_trace_gpu(
    orientation_field: np.ndarray,
    seeds: np.ndarray,
    step: float = 0.5,
    iters: int = 200,
    progress_callback=None,
) -> list[np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    orientation = torch.as_tensor(orientation_field, dtype=torch.float32, device=device)
    h, w = orientation.shape

    seed_tensor = torch.as_tensor(seeds, dtype=torch.float32, device=device)
    x = seed_tensor[:, 0].clone()
    y = seed_tensor[:, 1].clone()
    active = torch.ones_like(x, dtype=torch.bool)

    hist_x = [x.detach().cpu().numpy()]
    hist_y = [y.detach().cpu().numpy()]

    def sample_theta(x_pos: torch.Tensor, y_pos: torch.Tensor) -> torch.Tensor:
        x0 = torch.floor(x_pos).long().clamp(0, w - 1)
        y0 = torch.floor(y_pos).long().clamp(0, h - 1)
        x1 = (x0 + 1).clamp(0, w - 1)
        y1 = (y0 + 1).clamp(0, h - 1)

        wx = (x_pos - x0.float()).clamp(0.0, 1.0)
        wy = (y_pos - y0.float()).clamp(0.0, 1.0)

        q00 = orientation[y0, x0]
        q10 = orientation[y0, x1]
        q01 = orientation[y1, x0]
        q11 = orientation[y1, x1]

        top = q00 * (1.0 - wx) + q10 * wx
        bottom = q01 * (1.0 - wx) + q11 * wx
        return top * (1.0 - wy) + bottom * wy

    update_every = max(1, iters // 20)
    for iteration in range(iters):
        if not torch.any(active):
            break

        t1 = sample_theta(x, y)
        v1x, v1y = torch.cos(t1 + np.pi / 2.0), torch.sin(t1 + np.pi / 2.0)

        x2 = x + v1x * step * 0.5
        y2 = y + v1y * step * 0.5
        t2 = sample_theta(x2, y2)
        v2x, v2y = torch.cos(t2 + np.pi / 2.0), torch.sin(t2 + np.pi / 2.0)

        x3 = x + v2x * step * 0.5
        y3 = y + v2y * step * 0.5
        t3 = sample_theta(x3, y3)
        v3x, v3y = torch.cos(t3 + np.pi / 2.0), torch.sin(t3 + np.pi / 2.0)

        x4 = x + v3x * step
        y4 = y + v3y * step
        t4 = sample_theta(x4, y4)
        v4x, v4y = torch.cos(t4 + np.pi / 2.0), torch.sin(t4 + np.pi / 2.0)

        x_next = x + (v1x + 2 * v2x + 2 * v3x + v4x) * step / 6.0
        y_next = y + (v1y + 2 * v2y + 2 * v3y + v4y) * step / 6.0

        still_inside = (x_next >= 0) & (x_next < w) & (y_next >= 0) & (y_next < h)
        active = active & still_inside

        x = torch.where(active, x_next, x)
        y = torch.where(active, y_next, y)

        hist_x.append(torch.where(active, x, torch.full_like(x, float("nan"))).detach().cpu().numpy())
        hist_y.append(torch.where(active, y, torch.full_like(y, float("nan"))).detach().cpu().numpy())

        if progress_callback and ((iteration + 1) % update_every == 0 or iteration == iters - 1):
            progress_callback(min(100, int((iteration + 1) * 100 / max(1, iters))))

    all_x = np.stack(hist_x, axis=0)
    all_y = np.stack(hist_y, axis=0)
    results: list[np.ndarray] = []
    for idx in range(all_x.shape[1]):
        valid = ~(np.isnan(all_x[:, idx]) | np.isnan(all_y[:, idx]))
        if not np.any(valid):
            results.append(np.empty((0, 2), dtype=np.float32))
            continue
        results.append(np.column_stack((all_x[valid, idx], all_y[valid, idx])).astype(np.float32))
    return results


def kmeans_torch(
    features: np.ndarray,
    n_clusters: int,
    max_iter: int = 25,
    progress_callback=None,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.as_tensor(features, dtype=torch.float32, device=device)
    n_samples = data.shape[0]

    rng_idx = torch.randperm(n_samples, device=device)[:n_clusters]
    centroids = data[rng_idx].clone()

    labels = torch.zeros((n_samples,), dtype=torch.long, device=device)
    for iteration in range(max_iter):
        distances = torch.cdist(data, centroids)
        labels = torch.argmin(distances, dim=1)

        new_centroids = []
        for cluster_id in range(n_clusters):
            members = data[labels == cluster_id]
            if members.numel() == 0:
                new_centroids.append(centroids[cluster_id])
            else:
                new_centroids.append(members.mean(dim=0))
        centroids = torch.stack(new_centroids, dim=0)

        if progress_callback:
            progress_callback(min(100, int((iteration + 1) * 100 / max_iter)))

    return labels.detach().cpu().numpy()


def vector_field_sample(vector_array: np.ndarray, x_pos: float, y_pos: float) -> tuple[float, float]:
    y_idx = int(y_pos)
    x_idx = int(x_pos)
    y_idx = min(max(0, y_idx), vector_array.shape[1] - 1)
    x_idx = min(max(0, x_idx), vector_array.shape[2] - 1)
    v_comp, u_comp = vector_array[:, y_idx, x_idx]
    return float(u_comp), float(v_comp)


def rk4_trace_vector_field(
    x0: float,
    y0: float,
    step: float,
    steps: int,
    vector_array: np.ndarray,
    num_decimals: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    path_x = [x0]
    path_y = [y0]

    for _ in range(max(1, int(steps))):
        u0, v0 = vector_field_sample(vector_array, x0, y0)
        k1_x = step * u0
        k1_y = step * v0

        u1, v1 = vector_field_sample(vector_array, x0 + 0.5 * k1_x, y0 + 0.5 * k1_y)
        k2_x = step * u1
        k2_y = step * v1

        u2, v2 = vector_field_sample(vector_array, x0 + 0.5 * k2_x, y0 + 0.5 * k2_y)
        k3_x = step * u2
        k3_y = step * v2

        u3, v3 = vector_field_sample(vector_array, x0 + k3_x, y0 + k3_y)
        k4_x = step * u3
        k4_y = step * v3

        x0 += (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6.0
        y0 += (k1_y + 2 * k2_y + 2 * k3_y + k4_y) / 6.0

        path_x.append(x0)
        path_y.append(y0)

    return np.round(np.asarray(path_x), num_decimals), np.round(np.asarray(path_y), num_decimals)
