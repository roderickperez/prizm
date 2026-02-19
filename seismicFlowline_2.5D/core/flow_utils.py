import os
import numpy as np
import scipy
import segyio
from shapely.geometry import LineString

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

# Color constants for terminal output
C_CYAN = '\033[96m'
C_BOLD = '\033[1m'
C_END = '\033[0m'

from scipy.interpolate import interp1d

if HAS_CUPY:
    import cupyx.scipy.ndimage as cp_ndimage

def print_banner():
    banner = f"""{C_BOLD}{C_CYAN}
 ███████╗ ███████╗ ██╗ ███████╗ ███╗   ███╗ ██╗  ██████╗ ███████╗ ██╗      ██████╗  ██╗    ██╗
 ██╔════╝ ██╔════╝ ██║ ██╔════╝ ████╗ ████║ ██║ ██╔════╝ ██╔════╝ ██║     ██╔═══██╗ ██║    ██║
 ███████╗ █████╗   ██║ ███████╗ ██╔████╔██║ ██║ ██║      █████╗   ██║     ██║   ██║ ██║ █╗ ██║
 ╚════██║ ██╔══╝   ██║ ╚════██║ ██║╚██╔╝██║ ██║ ██║      ██╔══╝   ██║     ██║   ██║ ██║███╗██║
 ███████║ ███████╗ ██║ ███████║ ██║ ╚═╝ ██║ ██║ ╚██████╗ ██║      ███████╗╚██████╔╝ ╚███╔███╔╝
 ╚══════╝ ╚══════╝ ╚═╝ ╚══════╝ ╚═╝     ╚═╝ ╚═╝  ╚═════╝ ╚═╝      ╚══════╝ ╚═════╝   ╚══╝╚══╝ 

                           >>>>>>>>> [ VERSION: ORIGINAL ] <<<<<<<<<<<< 
    {C_END}"""
    print(banner)

# surface class
class Surface():
    def __init__(self, path, x_seed=None):
        self.x_seed = x_seed
        self.path = np.asarray(path)
        self.label = None
        
        # Optimized path rounding and integer conversion
        if self.path.size > 0:
            path_rounded = np.round(self.path, 0)
            if np.isnan(path_rounded).any():
                path_rounded = np.nan_to_num(path_rounded, nan=0)
            # Use vectorized operations to create tuples efficiently
            self.tuple_path = [tuple(p) for p in path_rounded.astype(int)]
        else:
            self.tuple_path = []
        
        if self.path.shape[0] > 1:
            self.linestring = LineString([(y,x) for y,x in list(zip(self.path[:,0],self.path[:,1]))])
        else:
            self.linestring = None
        self.line_weight = np.ones(len(self.path))

    def create_weighted_path(self):
        x = self.path[:, 1]
        y = self.path[:, 0]

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        return segments


def vector_field(x, y, vector_array):
    """CPU Version of vector field lookup."""
    y_idx = int(y)
    x_idx = int(x)
    y_idx = min(max(0, y_idx), vector_array.shape[1] - 1)
    x_idx = min(max(0, x_idx), vector_array.shape[2] - 1)
    v, u = vector_array[:, y_idx, x_idx]
    return u, v


def runge_kutta_4(x0, y0, h, steps, vector_array, num_decimals=2):
    """CPU Version of RK4."""
    # Fourth-order Runge-Kutta method
    path_x = [x0]
    path_y = [y0]

    for _ in range(steps):
        u0, v0 = vector_field(x0, y0, vector_array)
        k1_x = h * u0
        k1_y = h * v0

        u1, v1 = vector_field(x0 + 0.5 * k1_x, y0 + 0.5 * k1_y, vector_array)
        k2_x = h * u1
        k2_y = h * v1

        u2, v2 = vector_field(x0 + 0.5 * k2_x, y0 + 0.5 * k2_y, vector_array)
        k3_x = h * u2
        k3_y = h * v2

        u3, v3 = vector_field(x0 + k3_x, y0 + k3_y, vector_array)
        k4_x = h * u3
        k4_y = h * v3

        x0 += (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6
        y0 += (k1_y + 2*k2_y + 2*k3_y + k4_y) / 6

        path_x.append(x0)
        path_y.append(y0)

    path_x = np.array(path_x)
    path_y = np.array(path_y)

    path_x = np.round(path_x, num_decimals)
    path_y = np.round(path_y, num_decimals)
    return path_x, path_y


def gpu_rk4_vectorized(x0, y0, h, steps, vector_array_gpu):
    """
    GPU Vectorized RK4 using Cupy.
    Processes a batch of points (x0, y0) simultaneously.
    Properly handles boundaries by setting out-of-bounds points to NaN.
    """
    xp = cp.get_array_module(x0)
    num_points = x0.shape[0]
    height = vector_array_gpu.shape[1]
    width = vector_array_gpu.shape[2]
    
    # Initialize paths: [steps, num_points]
    all_x = xp.full((steps + 1, num_points), xp.nan, dtype=xp.float32)
    all_y = xp.full((steps + 1, num_points), xp.nan, dtype=xp.float32)
    
    all_x[0] = x0
    all_y[0] = y0
    
    curr_x = x0.copy()
    curr_y = y0.copy()
    
    # Mask to track points that are still within bounds
    active_mask = xp.ones(num_points, dtype=bool)
    
    # Vectorized compute
    for i in range(steps):
        # 1. Update mask: point must be inside [0, width-1] and [0, height-1]
        active_mask &= (curr_x >= 0) & (curr_x <= width - 1) & \
                      (curr_y >= 0) & (curr_y <= height - 1)
        
        if not active_mask.any():
            break
            
        # Get indices for active points
        ix = xp.round(curr_x).astype(xp.int32)
        iy = xp.round(curr_y).astype(xp.int32)
        
        # We need to clip for safe indexing of inactive points
        ix_clip = xp.clip(ix, 0, width - 1)
        iy_clip = xp.clip(iy, 0, height - 1)
        
        # k1
        u0 = vector_array_gpu[1, iy_clip, ix_clip]
        v0 = vector_array_gpu[0, iy_clip, ix_clip]
        k1_x = h * u0
        k1_y = h * v0
        
        # k2 (estimated)
        ix2 = xp.clip(xp.round(curr_x + 0.5*k1_x).astype(xp.int32), 0, width - 1)
        iy2 = xp.clip(xp.round(curr_y + 0.5*k1_y).astype(xp.int32), 0, height - 1)
        u1 = vector_array_gpu[1, iy2, ix2]
        v1 = vector_array_gpu[0, iy2, ix2]
        k2_x = h * u1
        k2_y = h * v1
        
        # k3
        ix3 = xp.clip(xp.round(curr_x + 0.5*k2_x).astype(xp.int32), 0, width - 1)
        iy3 = xp.clip(xp.round(curr_y + 0.5*k2_y).astype(xp.int32), 0, height - 1)
        u2 = vector_array_gpu[1, iy3, ix3]
        v2 = vector_array_gpu[0, iy3, ix3]
        k3_x = h * u2
        k3_y = h * v2
        
        # k4
        ix4 = xp.clip(xp.round(curr_x + k3_x).astype(xp.int32), 0, width - 1)
        iy4 = xp.clip(xp.round(curr_y + k3_y).astype(xp.int32), 0, height - 1)
        u3 = vector_array_gpu[1, iy4, ix4]
        v3 = vector_array_gpu[0, iy4, ix4]
        k4_x = h * u3
        k4_y = h * v3
        
        curr_x += (k1_x + 2*k2_x + 2*k3_x + k4_x) / 6
        curr_y += (k1_y + 2*k2_y + 2*k3_y + k4_y) / 6
        
        # Only store active points
        all_x[i+1, active_mask] = curr_x[active_mask]
        all_y[i+1, active_mask] = curr_y[active_mask]
        
    return all_x, all_y


def gpu_structure_tensor_2d(image, sigma, rho):
    """
    Compute 2D Structure Tensor on GPU using high-fidelity Gaussian derivatives.
    Returns: [Iyy, Ixx, Ixy] of shape (3, H, W)
    """
    image_gpu = cp.asarray(image, dtype=cp.float32)
    dy = cp_ndimage.gaussian_filter(image_gpu, sigma, order=(1, 0))
    dx = cp_ndimage.gaussian_filter(image_gpu, sigma, order=(0, 1))
    Ixx = dx * dx
    Ixy = dx * dy
    Iyy = dy * dy
    if rho > 0:
        Ixx = cp_ndimage.gaussian_filter(Ixx, rho)
        Ixy = cp_ndimage.gaussian_filter(Ixy, rho)
        Iyy = cp_ndimage.gaussian_filter(Iyy, rho)
    return cp.stack([Iyy, Ixx, Ixy])


def eig_special_2d_gpu(S):
    """Compute reflection-parallel vector field from Structure Tensor on GPU."""
    Iyy, Ixx, Ixy = S
    angle = 0.5 * cp.arctan2(2 * Ixy, Ixx - Iyy)
    vx = -cp.sin(angle)
    vy = cp.cos(angle)
    flip_mask = vx < 0
    vx = cp.where(flip_mask, -vx, vx)
    vy = cp.where(flip_mask, -vy, vy)
    return None, cp.stack([vy, vx])


def gpu_structure_tensor_3d(image, sigma, rho):
    """
    Compute 3D Structure Tensor on GPU.
    Returns: [Izz, Iyy, Ixx, Izy, Izx, Iyx] of shape (6, D, H, W)
    """
    image_gpu = cp.asarray(image, dtype=cp.float32)
    # Sanitize input image
    image_gpu = cp.nan_to_num(image_gpu, nan=0.0, posinf=0.0, neginf=0.0)
    
    dz = cp_ndimage.gaussian_filter(image_gpu, sigma, order=(1, 0, 0))
    dy = cp_ndimage.gaussian_filter(image_gpu, sigma, order=(0, 1, 0))
    dx = cp_ndimage.gaussian_filter(image_gpu, sigma, order=(0, 0, 1))
    
    Izz = dz * dz; Iyy = dy * dy; Ixx = dx * dx
    Izy = dz * dy; Izx = dz * dx; Iyx = dy * dx
    
    if rho > 0:
        Izz = cp_ndimage.gaussian_filter(Izz, rho)
        Iyy = cp_ndimage.gaussian_filter(Iyy, rho)
        Ixx = cp_ndimage.gaussian_filter(Ixx, rho)
        Izy = cp_ndimage.gaussian_filter(Izy, rho)
        Izx = cp_ndimage.gaussian_filter(Izx, rho)
        Iyx = cp_ndimage.gaussian_filter(Iyx, rho)
        
    return cp.stack([Izz, Iyy, Ixx, Izy, Izx, Iyx])


def eig_special_3d_gpu(S):
    """
    Compute Normal Vector from 3D Structure Tensor on GPU.
    The eigenvector corresponding to the LARGEST eigenvalue of the Gradient Structure Tensor 
    is the Normal to the structure (plane).
    
    S: [Izz, Iyy, Ixx, Izy, Izx, Iyx]
    Returns: normal_vector (3, D, H, W) -> [vz, vy, vx]
    """
    # Sanitize Tensor to avoid CUSOLVER errors
    S = cp.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)

    D, H, W = S.shape[1:]
    
    # Pack into (..., 3, 3)
    # S = [Izz, Iyy, Ixx, Izy, Izx, Iyx]
    #      0    1    2    3    4    5
    # Matrix:
    # [[Izz, Izy, Izx],
    #  [Izy, Iyy, Iyx],
    #  [Izx, Iyx, Ixx]]
    
    # Flatten spatial dims for batch processing
    S_flat = S.reshape(6, -1).T # (N, 6)
    N = S_flat.shape[0]
    
    # Process in chunks to avoid CUSOLVER limits or memory issues
    # A safe batch size is usually around 1-5 million.
    # N could be ~6M (200*300*100).
    BATCH_SIZE = 1000000 
    
    normal_flat = cp.empty((N, 3), dtype=cp.float32)
    
    for i in range(0, N, BATCH_SIZE):
        end = min(i + BATCH_SIZE, N)
        batch_slice = S_flat[i:end]
        n_batch = batch_slice.shape[0]
        
        matrices = cp.empty((n_batch, 3, 3), dtype=cp.float32)
        matrices[:, 0, 0] = batch_slice[:, 0] # Izz
        matrices[:, 0, 1] = batch_slice[:, 3] # Izy
        matrices[:, 0, 2] = batch_slice[:, 4] # Izx
        matrices[:, 1, 0] = batch_slice[:, 3] # Izy
        matrices[:, 1, 1] = batch_slice[:, 1] # Iyy
        matrices[:, 1, 2] = batch_slice[:, 5] # Iyx
        matrices[:, 2, 0] = batch_slice[:, 4] # Izx
        matrices[:, 2, 1] = batch_slice[:, 5] # Iyx
        matrices[:, 2, 2] = batch_slice[:, 2] # Ixx
        
        # Compute eigenvalues/vectors
        # eigh returns sorted eigenvalues w and eigenvectors v
        # w (..., M), v (..., M, M)
        try:
            w, v = cp.linalg.eigh(matrices)
            # Largest eigenvalue is last index (-1)
            normal_flat[i:end] = v[..., :, -1]
        except Exception as e:
            # Fallback for failed batch - could be singular matrices or NaNs survived?
            # Initialize with default normal (e.g., [1, 0, 0] -> Inline direction)
            print(f"Warning: Batch {i}-{end} failed: {e}. Using default normal.")
            normal_flat[i:end] = cp.array([1.0, 0.0, 0.0], dtype=cp.float32)

    # Reshape back to (3, D, H, W)
    normal = normal_flat.T.reshape(3, D, H, W)
    
    # Orient normals consistently (e.g. towards positive Z)
    vz = normal[0]
    flip_mask = vz < 0
    normal = cp.where(flip_mask, -normal, normal)
    
    return normal


def extract_surfaces(seismic_slice, vector_array, sample_intervals, mode='peak', device='cpu', kwargs={}):
    """Extract surfaces using either CPU or GPU."""
    if device == 'gpu' and HAS_CUPY:
        return extract_surfaces_gpu(seismic_slice, vector_array, sample_intervals, mode, kwargs)
    
    surfaces = []
    num_peaks_all = 0
    for sample_interval_x in sample_intervals:
        seeds = np.arange(0, seismic_slice.shape[1] - 1, sample_interval_x)
        for x_seed in seeds:
            trace = seismic_slice[:, x_seed]
            if mode == 'peak': peaks, _ = scipy.signal.find_peaks(trace, **kwargs)
            elif mode == 'trough': peaks, _ = scipy.signal.find_peaks(-trace, **kwargs)
            else: peaks, _ = scipy.signal.find_peaks(np.abs(trace), **kwargs)
            num_peaks_all += len(peaks)
            for peak in peaks:
                y0, x0 = peak, x_seed
                px, py = runge_kutta_4(x0, y0, 1.0, seismic_slice.shape[1] - x_seed, vector_array)
                px_b, py_b = runge_kutta_4(x0, y0, 1.0, x_seed, vector_array * -1)
                mask = (px >= 0) & (px < seismic_slice.shape[1]) & (py >= 0) & (py < seismic_slice.shape[0])
                px, py = px[mask], py[mask]
                mask_b = (px_b >= 0) & (px_b < seismic_slice.shape[1]) & (py_b >= 0) & (py_b < seismic_slice.shape[0])
                px_b, py_b = px_b[mask_b], py_b[mask_b]
                if len(px) > 0 and len(px_b) > 0:
                    merged = np.concatenate([np.column_stack([py_b[::-1], px_b[::-1]]), np.column_stack([py, px])])
                    surfaces.append(Surface(merged, x_seed))
    return surfaces, num_peaks_all


def extract_surfaces_gpu(seismic_slice, vector_array, sample_intervals, mode='peak', kwargs={}):
    """High-performance GPU vectorized surface extraction."""
    seismic_gpu = cp.asarray(seismic_slice)
    vector_gpu = cp.asarray(vector_array)
    surfaces = []
    num_peaks_all = 0
    h, width, height = 1.0, seismic_slice.shape[1], seismic_slice.shape[0]
    for sample_interval_x in sample_intervals:
        seeds_x = np.arange(0, width - 1, sample_interval_x)
        all_seed_x, all_seed_y = [], []
        for x in seeds_x:
            trace = seismic_slice[:, x]
            # Ensure trace is numpy for scipy.signal.find_peaks
            if hasattr(trace, 'get'):
                trace_cpu = trace.get()
            else:
                trace_cpu = trace
                
            if mode == 'peak': peaks, _ = scipy.signal.find_peaks(trace_cpu, **kwargs)
            elif mode == 'trough': peaks, _ = scipy.signal.find_peaks(-trace_cpu, **kwargs)
            else: peaks, _ = scipy.signal.find_peaks(np.abs(trace_cpu), **kwargs)
            for p in peaks:
                all_seed_x.append(x)
                all_seed_y.append(p)
        if not all_seed_x: continue
        num_seeds = len(all_seed_x)
        num_peaks_all += num_seeds
        seeds_x_gpu = cp.array(all_seed_x, dtype=cp.float32)
        seeds_y_gpu = cp.array(all_seed_y, dtype=cp.float32)
        all_x_f, all_y_f = gpu_rk4_vectorized(seeds_x_gpu, seeds_y_gpu, h, width, vector_gpu)
        all_x_b, all_y_b = gpu_rk4_vectorized(seeds_x_gpu, seeds_y_gpu, h, width, vector_gpu * -1)
        cpu_x_f, cpu_y_f, cpu_x_b, cpu_y_b = all_x_f.get(), all_y_f.get(), all_x_b.get(), all_y_b.get()
        for i in range(num_seeds):
            xf, yf = cpu_x_f[:, i], cpu_y_f[:, i]
            xb, yb = cpu_x_b[:, i], cpu_y_b[:, i]
            xf, yf = xf[~np.isnan(xf)], yf[~np.isnan(yf)]
            xb, yb = xb[~np.isnan(xb)], yb[~np.isnan(yb)]
            if len(xf) > 0 and len(xb) > 0:
                merged = np.concatenate([np.column_stack([yb[::-1], xb[::-1]]), np.column_stack([yf, xf])])
                surfaces.append(Surface(merged, all_seed_x[i]))
    return surfaces, num_peaks_all


def surface_to_feature_vector(surfaces, max_size=None, only_y=False):
    if max_size is None:
        max_size = 0
        for surface in surfaces: max_size = max(max_size, surface.path.shape[0])
    feature_vectors = []
    for surface in surfaces:
        template = np.zeros((max_size, 2)) - 1
        template[:surface.path.shape[0], :] = surface.path
        feature_vectors.append(template)
    feature_vectors = np.array(feature_vectors)
    if only_y: feature_vectors = feature_vectors[..., 0]
    feature_vectors = feature_vectors.reshape(len(feature_vectors), -1)
    return feature_vectors, max_size

def sort_tops(seismic_slice, surfaces, clusterer):
    x = np.arange(seismic_slice.shape[1])
    boundary_base = np.stack((np.ones(seismic_slice.shape[1]) * (seismic_slice.shape[0] - 1), x)).T
    boundary_top = np.stack((np.zeros(seismic_slice.shape[1]), x)).T
    
    # cuML compatibility: ensure labels are numpy for dictionary keys
    labels = clusterer.labels_
    if hasattr(labels, 'get'):
        labels = labels.get()
    
    unique_labels = np.unique(labels)
    label_boundaries = {int(label): [np.inf, -np.inf, 0, 0] for label in unique_labels}
    for surface in surfaces:
        mean_depth = surface.path[:,0].mean()
        if mean_depth < label_boundaries[surface.kmeans_label][0]:
            label_boundaries[surface.kmeans_label][2], label_boundaries[surface.kmeans_label][0] = surface.path, mean_depth
        if mean_depth > label_boundaries[surface.kmeans_label][1]:
            label_boundaries[surface.kmeans_label][3], label_boundaries[surface.kmeans_label][1] = surface.path, mean_depth
    mean_depth_tops = [values[0] for values in label_boundaries.values()]
    tops = [values[2] for values in label_boundaries.values()]
    tops.extend([boundary_top, boundary_base])
    mean_depth_tops.extend([0, seismic_slice.shape[0]])
    return [tops[ind] for ind in np.argsort(mean_depth_tops)]

def interpolate_boundary(boundary_path, image_shape):
    height, width = image_shape
    boundary_path = np.asarray(boundary_path)
    if boundary_path.size == 0:
        return np.vstack((np.full(width, height // 2), np.arange(width))).T
    y_coords, x_coords = boundary_path[:, 0], boundary_path[:, 1]
    _, unique_indices = np.unique(x_coords, return_index=True)
    ux, uy = x_coords[np.sort(unique_indices)], y_coords[np.sort(unique_indices)]
    if len(ux) < 2: new_y_coords = np.full(width, np.mean(uy))
    else: new_y_coords = interp1d(ux, uy, kind='linear', fill_value="extrapolate")(np.arange(width))
    new_y_coords = np.clip(np.nan_to_num(new_y_coords, nan=height // 2), 0, height - 1)
    return np.vstack((new_y_coords, np.arange(width))).T

def save_segy_volume(data, output_path, template_path, trace_indices):
    """
    Save a robust SEGY volume by mapping sub-volume data to original headers.
    data: (Samples, Traces) numpy array.
    trace_indices: List/array of original trace indices in the template SEGY.
    """
    with segyio.open(template_path, "r", ignore_geometry=True) as src:
        # Create a clean spec for the new file
        spec = segyio.spec()
        spec.sorting = 1 
        spec.format = src.format
        spec.samples = src.samples[:data.shape[0]]
        spec.tracecount = len(trace_indices)
        
        # Data must be (Traces, Samples) for dst.trace assignment
        flat_data = data.T.astype(np.float32)
        
        with segyio.create(output_path, spec) as dst:
            dst.text[0] = src.text[0]
            dst.bin = src.bin
            dst.bin[segyio.BinField.Samples] = len(spec.samples)
            
            for i, old_idx in enumerate(trace_indices):
                dst.header[i] = src.header[old_idx]
                dst.header[i][segyio.TraceField.TRACE_SAMPLE_COUNT] = len(spec.samples)
            
            dst.trace = flat_data
    return output_path

def inspect_segy_headers(segy_path, num_traces=5):
    """
    Interactively inspect SEGY headers using a rich table.
    """
    from rich.table import Table
    from rich.console import Console
    console = Console()
    
    if not os.path.exists(segy_path):
        console.print(f"[bold red]Error: File {segy_path} not found.[/bold red]")
        return

    with segyio.open(segy_path, ignore_geometry=True) as s:
        n_traces = s.tracecount
        n_samples = s.samples.size
        
        table = Table(title=f"SEGY Header Inspection: {os.path.basename(segy_path)}")
        table.add_column("Trace #", style="cyan")
        table.add_column("Inline (189)", style="magenta")
        table.add_column("Xline (193)", style="yellow")
        table.add_column("X (181)", style="green")
        table.add_column("Y (185)", style="blue")
        table.add_column("Samples", style="white")

        # Inspect first few, middle, and last trace if possible
        indices = list(range(min(num_traces, n_traces)))
        if n_traces > num_traces:
             indices.append(n_traces // 2)
             indices.append(n_traces - 1)
        
        for i in sorted(list(set(indices))):
            h = s.header[i]
            table.add_row(
                str(i),
                str(h[segyio.TraceField.INLINE_3D]),
                str(h[segyio.TraceField.CROSSLINE_3D]),
                str(h[segyio.TraceField.SourceX]),
                str(h[segyio.TraceField.SourceY]),
                str(n_samples)
            )
        
    console.print(table)
    console.print(f"Total Traces: [bold cyan]{n_traces}[/bold cyan] | Samples: [bold cyan]{n_samples}[/bold cyan]")