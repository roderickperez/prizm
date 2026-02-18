import numpy as np
import bruges as bg
import torch
import torch.nn.functional as F
from scipy.ndimage import uniform_filter1d


def compute_gst_pytorch(data: np.ndarray, sigma: float = 1.0, rho: float = 1.0) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.inference_mode():
        img_tensor = torch.from_numpy(np.asarray(data, dtype=np.float32, order="C")).to(device)

        if img_tensor.ndim == 2:
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
            conv_op = F.conv2d
            is_3d = False
        else:
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
            conv_op = F.conv3d
            is_3d = True

        def get_gaussian_kernel(kernel_size: int, sig: float) -> torch.Tensor:
            x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32, device=device)
            kernel = torch.exp(-x**2 / (2 * sig**2))
            kernel = kernel / kernel.sum()
            return kernel

        kernel_size = int(4 * sigma + 1) | 1
        pad = kernel_size // 2
        kernel_1d = get_gaussian_kernel(kernel_size, sigma)

        if is_3d:
            k_z = kernel_1d.view(1, 1, -1, 1, 1)
            k_y = kernel_1d.view(1, 1, 1, -1, 1)
            k_x = kernel_1d.view(1, 1, 1, 1, -1)
            img_s = conv_op(img_tensor, k_z, padding=(pad, 0, 0))
            img_s = conv_op(img_s, k_y, padding=(0, pad, 0))
            img_s = conv_op(img_s, k_x, padding=(0, 0, pad))
            g_z = img_s[:, :, 2:, :, :] - img_s[:, :, :-2, :, :]
            g_y = img_s[:, :, :, 2:, :] - img_s[:, :, :, :-2, :]
            g_x = img_s[:, :, :, :, 2:] - img_s[:, :, :, :, :-2]
            g_z = F.pad(g_z, (0, 0, 0, 0, 1, 1))
            g_y = F.pad(g_y, (0, 0, 1, 1, 0, 0))
            g_x = F.pad(g_x, (1, 1, 0, 0, 0, 0))
            components = [g_z, g_y, g_x]
        else:
            k_y = kernel_1d.view(1, 1, -1, 1)
            k_x = kernel_1d.view(1, 1, 1, -1)
            img_s = conv_op(img_tensor, k_y, padding=(pad, 0))
            img_s = conv_op(img_s, k_x, padding=(0, pad))
            g_y = img_s[:, :, 2:, :] - img_s[:, :, :-2, :]
            g_x = img_s[:, :, :, 2:] - img_s[:, :, :, :-2]
            g_y = F.pad(g_y, (0, 0, 1, 1))
            g_x = F.pad(g_x, (1, 1, 0, 0))
            components = [g_y, g_x]

        kernel_size_rho = int(4 * rho + 1) | 1
        pad_rho = kernel_size_rho // 2
        k_rho = get_gaussian_kernel(kernel_size_rho, rho)

        n_dim = 3 if is_3d else 2
        tensor_smooth = []

        if is_3d:
            kr_z = k_rho.view(1, 1, -1, 1, 1)
            kr_y = k_rho.view(1, 1, 1, -1, 1)
            kr_x = k_rho.view(1, 1, 1, 1, -1)
            kernels = [kr_z, kr_y, kr_x]
            paddings = [(pad_rho, 0, 0), (0, pad_rho, 0), (0, 0, pad_rho)]
        else:
            kr_y = k_rho.view(1, 1, -1, 1)
            kr_x = k_rho.view(1, 1, 1, -1)
            kernels = [kr_y, kr_x]
            paddings = [(pad_rho, 0), (0, pad_rho)]

        for i in range(n_dim):
            for j in range(i, n_dim):
                comp = components[i] * components[j]
                for kernel, padding in zip(kernels, paddings):
                    comp = conv_op(comp, kernel, padding=padding)
                tensor_smooth.append(comp)

        if not is_3d:
            syy, sxy, sxx = tensor_smooth[0], tensor_smooth[1], tensor_smooth[2]
            trace = sxx + syy
            det = sxx * syy - sxy**2
            delta = torch.sqrt(torch.clamp((trace / 2) ** 2 - det, min=0))
            l1 = trace / 2 + delta
            l2 = trace / 2 - delta
            coherence = (l1 - l2) / (l1 + l2 + 1e-6)
            return coherence.squeeze().cpu().numpy()

        szz, szy, szx, syy, syx, sxx = tensor_smooth
        _, _, d, h, w = sxx.shape
        n = d * h * w
        matrix = torch.zeros((n, 3, 3), device=device)
        matrix[:, 0, 0] = sxx.view(-1)
        matrix[:, 1, 1] = syy.view(-1)
        matrix[:, 2, 2] = szz.view(-1)
        matrix[:, 0, 1] = matrix[:, 1, 0] = syx.view(-1)
        matrix[:, 0, 2] = matrix[:, 2, 0] = szx.view(-1)
        matrix[:, 1, 2] = matrix[:, 2, 1] = szy.view(-1)
        vals = torch.linalg.eigvalsh(matrix)
        e1 = vals[:, 0]
        e3 = vals[:, 2]
        metric = (e3 - e1) / (e3 + e1 + 1e-6)
        return metric.view(d, h, w).cpu().numpy()


def compute_attribute(
    data: np.ndarray,
    attr_name: str,
    dt: float,
    duration: float,
    window_size: int,
    sigma: float,
    rho: float,
    export_mode: bool = False,
) -> np.ndarray:
    is_3d = data.ndim == 3

    if attr_name == "Envelope":
        attr = bg.attribute.envelope(data)
    elif attr_name == "Instantaneous Phase":
        attr = bg.attribute.instantaneous_phase(data)
    elif attr_name == "Instantaneous Frequency":
        attr = bg.attribute.instantaneous_frequency(data, dt=dt)
    elif attr_name == "Energy":
        attr = bg.attribute.energy(data, duration=duration, dt=dt)
    elif attr_name == "Similarity (GST)":
        try:
            attr = bg.attribute.similarity(data, duration=duration, dt=dt, kind="gst")
        except Exception:
            attr = np.zeros_like(data)
    elif attr_name == "RMS Amplitude":
        win = max(1, int(window_size))
        squared = np.square(data, dtype=np.float32)
        mean_sq = uniform_filter1d(squared, size=win, axis=-1, mode="nearest")
        np.sqrt(mean_sq, out=mean_sq)
        attr = mean_sq
    elif attr_name == "Cosine of Phase":
        attr = np.cos(bg.attribute.instantaneous_phase(data))
    elif attr_name == "Gradient Structure Tensor (GPU)":
        attr = compute_gst_pytorch(data, sigma=sigma, rho=rho)
    else:
        attr = data

    if is_3d and not export_mode and attr.shape == data.shape:
        attr = attr[:, :, attr.shape[-1] // 2]

    return attr
