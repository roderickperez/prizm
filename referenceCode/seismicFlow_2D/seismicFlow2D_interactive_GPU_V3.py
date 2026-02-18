import os
import sys
import numpy as np
import segyio
import matplotlib.pyplot as plt
from structure_tensor import eig_special_2d, structure_tensor_2d
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from scipy.interpolate import interp1d

# Add parent directory to path to import flow_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flow_utils import *

console = Console()

def print_rich_banner():
    banner_text = """[bold cyan]
 ███████╗ ███████╗ ██╗ ███████╗ ███╗   ███╗ ██╗  ██████╗ ███████╗ ██╗      ██████╗  ██╗    ██╗
 ██╔════╝ ██╔════╝ ██║ ██╔════╝ ████╗ ████║ ██║ ██╔════╝ ██╔════╝ ██║     ██╔═══██╗ ██║    ██║
 ███████╗ █████╗   ██║ ███████╗ ██╔████╔██║ ██║ ██║      █████╗   ██║     ██║   ██║ ██║ █╗ ██║
 ╚════██║ ██╔══╝   ██║ ╚════██║ ██║╚██╔╝██║ ██║ ██║      ██╔══╝   ██║     ██║   ██║ ██║███╗██║
 ███████║ ███████╗ ██║ ███████║ ██║ ╚═╝ ██║ ██║ ╚██████╗ ██║      ███████╗╚██████╔╝ ╚███╔███╔╝
 ╚══════╝ ╚══════╝ ╚═╝ ╚══════╝ ╚═╝     ╚═╝ ╚═╝  ╚═════╝ ╚══════╝ ╚══════╝ ╚═════╝   ╚══╝╚══╝ 

                            >>>>>>>>> [ V3 - WHEELER DOMAIN ] <<<<<<<<<<<< 
    [/bold cyan]"""
    console.print(Panel(banner_text, border_style="cyan"))

def main():
    print_rich_banner()
    
    # 0. Device selection
    device_choice = Prompt.ask("Select processing device", choices=["cpu", "gpu"], default="gpu")
    console.print(f"Using [bold magenta]{device_choice.upper()}[/bold magenta] for performance heavy operations")
    # Configuration
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    segy_path = os.path.join(base_dir, "seismicData/segy/1_Original_Seismics.sgy")
    output_dir = os.path.join(base_dir, "seismicFlow2D_GPU/output")
    figures_dir = os.path.join(base_dir, "seismicFlow2D_GPU/figures")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # 1. Loading SEGY
    if not os.path.exists(segy_path):
        console.print(f"[bold red]Error: File not found at {segy_path}[/bold red]")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Scanning SEGY headers...", total=None)
        with segyio.open(segy_path, ignore_geometry=True) as s:
            # Get all inlines and xlines from headers
            ilines = s.attributes(segyio.TraceField.INLINE_3D)[:]
            xlines = s.attributes(segyio.TraceField.CROSSLINE_3D)[:]
            
            unique_ilines = np.sort(np.unique(ilines))
            unique_xlines = np.sort(np.unique(xlines))
            num_samples = s.samples.size
            
            progress.update(task, description="Headers Scanned!")

    # 2. Volume Info
    table = Table(title="Volume Information (Sparse)")
    table.add_column("Dimension", style="cyan")
    table.add_column("Size", style="magenta")
    table.add_column("Range", style="yellow")
    table.add_row("UNIQUE INLINES", str(len(unique_ilines)), f"{unique_ilines.min()} - {unique_ilines.max()}")
    table.add_row("UNIQUE XLINES", str(len(unique_xlines)), f"{unique_xlines.min()} - {unique_xlines.max()}")
    table.add_row("TIME SAMPLES", str(num_samples), "N/A")
    console.print(table)

    # 3. Slice Extraction
    extraction_mode = Prompt.ask("Extract by [bold]inline[/bold] or [bold]xline[/bold]?", choices=["inline", "xline"], default="inline")
    
    if extraction_mode == "inline":
        choices = [str(x) for x in unique_ilines]
        default_idx = str(unique_ilines[len(unique_ilines)//2])
        slice_idx = int(Prompt.ask(f"Select Inline", choices=choices, default=default_idx))
        
        # Filter traces
        trace_indices = np.where(ilines == slice_idx)[0]
        slice_xlines = xlines[trace_indices]
        
        # Sort traces by xline to ensure spatial continuity
        sort_idx = np.argsort(slice_xlines)
        trace_indices = trace_indices[sort_idx]
        
    else:
        choices = [str(x) for x in unique_xlines]
        default_idx = str(unique_xlines[len(unique_xlines)//2])
        slice_idx = int(Prompt.ask(f"Select Xline", choices=choices, default=default_idx))
        
        # Filter traces
        trace_indices = np.where(xlines == slice_idx)[0]
        slice_ilines = ilines[trace_indices]
        
        # Sort traces by inline
        sort_idx = np.argsort(slice_ilines)
        trace_indices = trace_indices[sort_idx]

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as p:
        p.add_task(f"Extracting {len(trace_indices)} traces for slice...", total=None)
        with segyio.open(segy_path, ignore_geometry=True) as s:
            # Read traces and stack them into 2D array [traces, time]
            seismic_2d = np.stack([s.trace[i] for i in trace_indices])
            seismic_2d = seismic_2d.T # Transform to [time, traces]

    amplitude_max = np.percentile(np.abs(seismic_2d), 99)

    # 4. Visualize Selected Slice
    if Confirm.ask("Show selected 2D slice?"):
        plt.figure(figsize=(10, 8))
        plt.imshow(seismic_2d, vmin=-amplitude_max, vmax=amplitude_max, cmap='Greys', aspect='auto')
        plt.colorbar(label='Amplitude')
        plt.title(f'{extraction_mode.upper()} {slice_idx}')
        plt.xlabel('Trace Index')
        plt.ylabel('Time/Depth Index')
        
        fig_path = os.path.join(figures_dir, f"slice_{extraction_mode}_{slice_idx}.png")
        plt.savefig(fig_path)
        console.print(f"[green]Saved slice visualization to {fig_path}[/green]")

    # 5. Gradient Structure Tensor
    vector_path = os.path.join(output_dir, f"vector_field_{extraction_mode}_{slice_idx}.npy")
    
    if os.path.exists(vector_path) and Confirm.ask("Vector field exists. Use cached version?"):
        vector_array = np.load(vector_path)
    else:
        sigma = 1.0
        rho = 2.0
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as p:
            task = p.add_task(f"Calculating Structure Tensor on {device_choice.upper()}...", total=None)
            import time
            start_st = time.time()
            
            if device_choice == 'gpu' and HAS_CUPY:
                # Use high-performance GPU kernels
                S = gpu_structure_tensor_2d(seismic_2d, sigma=sigma, rho=rho)
                _, vector_array_gpu = eig_special_2d_gpu(S)
                vector_array = vector_array_gpu.get() # Transfer back for visualization and caching
            else:
                # Fallback to CPU
                S = structure_tensor_2d(seismic_2d, sigma=sigma, rho=rho)
                _, vector_array = eig_special_2d(S)
                
            st_time = time.time() - start_st
            np.save(vector_path, vector_array)
            p.update(task, description=f"Structure Tensor Computed in {st_time:.2f}s!")
    
    if Confirm.ask("Visualize Gradient Structure Tensor & Vector Field?"):
        # Magnitude plot
        mag = np.sqrt(vector_array[0]**2 + vector_array[1]**2)
        plt.figure(figsize=(10, 8))
        plt.imshow(mag, cmap='viridis', aspect='auto')
        plt.colorbar(label='Magnitude')
        plt.title("Vector Field Magnitude")
        plt.savefig(os.path.join(figures_dir, f"vector_mag_{extraction_mode}_{slice_idx}.png"))

        # Orientation plot
        orientation = np.arctan2(vector_array[1], vector_array[0])
        plt.figure(figsize=(10, 8))
        plt.imshow(orientation, cmap='twilight', aspect='auto')
        plt.colorbar(label='Radians')
        plt.title("Vector Field Orientation (Local Dip)")
        plt.savefig(os.path.join(figures_dir, f"vector_orientation_{extraction_mode}_{slice_idx}.png"))

    # 6. Surface Extraction
    if Confirm.ask("Extract surfaces using RK4?"):
        sample_int = IntPrompt.ask("Trace sample interval", default=100)
        mode = Prompt.ask("Extraction mode", choices=['peak', 'trough', 'both'], default='both')
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeRemainingColumn(),
            console=console
        ) as p:
            task = p.add_task(f"Tracing flowlines on {device_choice.upper()}...", total=None)
            import time
            start_rk = time.time()
            surfaces, _ = extract_surfaces(
                seismic_2d,
                vector_array,
                [sample_int],
                mode=mode,
                device=device_choice,
                kwargs={"height": None, "distance": None, "prominence": None}
            )
            rk_time = time.time() - start_rk
            p.update(task, description=f"Extracted {len(surfaces)} flowlines in {rk_time:.2f}s!")
            
            if device_choice == 'gpu':
                console.print(f"[bold green]GPU Speedup achieved![/bold green] (Tracing took {rk_time:.4f}s)")

        # 6b. Unconformity Extraction (Heatmap & Ranking)
        if Confirm.ask("Extract major unconformities based on overlap?"):
            num_unconf = IntPrompt.ask("Number of top surfaces to extract as unconformities", default=20)
            
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as p:
                p.add_task("Calculating Overlap Heatmap...", total=None)
                
                # Create heatmap
                overlap_heatmap = np.zeros(seismic_2d.shape)
                for surf in surfaces:
                    # Clip coordinates to seismic bounds
                    y_idx = np.clip(surf.path[:, 0].astype(int), 0, seismic_2d.shape[0] - 1)
                    x_idx = np.clip(surf.path[:, 1].astype(int), 0, seismic_2d.shape[1] - 1)
                    # Batch increment heatmap
                    np.add.at(overlap_heatmap, (y_idx, x_idx), 1)
                
                # Score surfaces
                for surf in surfaces:
                    y_idx = np.clip(surf.path[:, 0].astype(int), 0, seismic_2d.shape[0] - 1)
                    x_idx = np.clip(surf.path[:, 1].astype(int), 0, seismic_2d.shape[1] - 1)
                    surf.overlap_score = np.mean(overlap_heatmap[y_idx, x_idx])
                
                # Sort and select top ranked
                unconformities = sorted(surfaces, key=lambda s: getattr(s, 'overlap_score', 0), reverse=True)[:num_unconf]
                console.print(f"[bold yellow]Identified {len(unconformities)} Major Unconformities.[/bold yellow]")
        else:
            unconformities = []

        # 7. Visualize Results
        if Confirm.ask("Visualize extraction results?"):
            # Option A: Flowlines only
            fig, ax = plt.subplots(figsize=(12, 10))
            for surf in surfaces:
                ax.plot(surf.path[:, 1], surf.path[:, 0], linewidth=0.1, color='k', alpha=0.5)
            ax.set_title("Extracted Flowlines (Geometries Only)")
            ax.invert_yaxis()
            plt.savefig(os.path.join(figures_dir, f"flowlines_{extraction_mode}_{slice_idx}.png"))

            # Option B: Red lines overlay on Seismic
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.imshow(seismic_2d, cmap='Greys', vmin=-amplitude_max, vmax=amplitude_max, aspect='auto')
            for surf in surfaces:
                ax.plot(surf.path[:, 1], surf.path[:, 0], linewidth=0.1, color='r', alpha=0.8)
            ax.set_title("Flowlines Overlay on Seismic")
            plt.savefig(os.path.join(figures_dir, f"overlay_{extraction_mode}_{slice_idx}.png"))

            # Option C: Colormap lines
            cmap_choice = Prompt.ask("Choose colormap for flowlines", default="jet")
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.imshow(seismic_2d, cmap='Greys', vmin=-amplitude_max, vmax=amplitude_max, aspect='auto', alpha=0.3)
            
            # Simple color mapping based on index or depth
            colors = plt.get_cmap(cmap_choice)(np.linspace(0, 1, len(surfaces)))
            for i, surf in enumerate(surfaces):
                ax.plot(surf.path[:, 1], surf.path[:, 0], linewidth=0.1, color=colors[i], alpha=0.9)
            
            sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap_choice), norm=plt.Normalize(vmin=0, vmax=len(surfaces)))
            plt.colorbar(sm, ax=ax, label='Surface Index')
            ax.set_title(f"Flowlines with {cmap_choice} Colormap")
            plt.savefig(os.path.join(figures_dir, f"colormap_{extraction_mode}_{slice_idx}.png"))

            # Option D: Unconformities Only (Blank Background)
            if unconformities:
                fig, ax = plt.subplots(figsize=(12, 10))
                for surf in unconformities:
                    ax.plot(surf.path[:, 1], surf.path[:, 0], linewidth=0.1, color='r')
                ax.set_title(f"Top {len(unconformities)} Major Unconformities")
                ax.set_xlim(0, seismic_2d.shape[1]); ax.set_ylim(seismic_2d.shape[0], 0)
                plt.savefig(os.path.join(figures_dir, f"unconformities_only_{extraction_mode}_{slice_idx}.png"))

                # Option E: Unconformities Red Overlay on Seismic
                fig, ax = plt.subplots(figsize=(12, 10))
                ax.imshow(seismic_2d, cmap='Greys', vmin=-amplitude_max, vmax=amplitude_max, aspect='auto')
                for surf in unconformities:
                    ax.plot(surf.path[:, 1], surf.path[:, 0], linewidth=0.2, color='r', alpha=0.9)
                ax.set_title("Unconformities Overlay on Seismic")
                plt.savefig(os.path.join(figures_dir, f"unconformities_red_overlay_{extraction_mode}_{slice_idx}.png"))

    # 8. K-Means Segmentation
    if Confirm.ask("Generate segmentation map (K-Means)?"):
        num_clusters = IntPrompt.ask("Number of segments/clusters", default=10)
        only_y = Confirm.ask("Use only y-coordinates for clustering?", default=False)
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as p:
            task = p.add_task("Running K-Means Segmentation...", total=None)
            
            # Prepare feature vectors
            X, max_size = surface_to_feature_vector(surfaces, only_y=only_y)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            clusterer = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(X_scaled)
            
            # Reassign labels to surfaces
            for ind, surface in enumerate(surfaces):
                surface.kmeans_label = clusterer.labels_[ind]
            
            # Reconstruct segmentation map using sort_tops and interpolate_boundary logic
            centroids = clusterer.cluster_centers_
            sorted_indices = np.argsort(centroids[:, 1]) # Sort by y-coordinate
            
            # Crucial: Add placeholders for boundaries
            sorted_indices = np.append(sorted_indices, num_clusters + 1)
            sorted_indices = np.append(sorted_indices, num_clusters + 2)

            label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_indices)}
            
            # Map labels
            for surface in surfaces:
                surface.kmeans_label = label_mapping[surface.kmeans_label]
            
            # Create the mask
            sorted_tops = sort_tops(seismic_2d, surfaces, clusterer)
            sorted_tops_surfaces = [Surface(tops) for tops in sorted_tops]
            interp_tops = [Surface(interpolate_boundary(top.path, seismic_2d.shape)) for top in sorted_tops_surfaces]

            array_mask = np.zeros(seismic_2d.shape)
            _, Y = np.meshgrid(np.arange(seismic_2d.shape[1]), np.arange(seismic_2d.shape[0]))
            
            label_map_back = {v: k for k, v in label_mapping.items()}
            
            for i in range(len(interp_tops) - 1):
                y1 = interp_tops[i].path[:, 0]
                y2 = interp_tops[i+1].path[:, 0]
                if i in label_map_back:
                    array_mask[np.where((Y > y1) & (Y < y2))] = label_map_back[i]
            
            p.update(task, description="Segmentation Map Created!")

        # Final map plot
        plt.figure(figsize=(10, 8))
        plt.imshow(array_mask, cmap='jet', aspect='auto')
        plt.colorbar(label='Cluster ID')
        plt.title(f"Segmentation Map (k={num_clusters})")
        plt.savefig(os.path.join(figures_dir, f"segmentation_{extraction_mode}_{slice_idx}.png"))
        console.print(f"[green]Saved segmentation map to {figures_dir}[/green]")

    # 9. Wheeler Time Plot (Stratigraphic Flattening)
    if Confirm.ask("Generate a 2D Wheeler Time plot (Stratigraphic Flattening)?"):
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), console=console) as p:
            task = p.add_task("Processing Wheeler Transformation...", total=seismic_2d.shape[1])
            
            # Step 1: Define Target Depths (Mean Y for each surface)
            # Filter surfaces that have a decent length to avoid noisy mapping
            min_len = seismic_2d.shape[1] // 2
            valid_surfs = [s for s in surfaces if len(s.path) > min_len]
            
            if not valid_surfs:
                console.print("[yellow]Warning: Not enough long flowlines to perform robust flattening. Using all flowlines.[/yellow]")
                valid_surfs = surfaces

            target_depths = []
            for surf in valid_surfs:
                target_depths.append(np.mean(surf.path[:, 0]))
            
            # Sort by depth to maintain order
            sort_idx = np.argsort(target_depths)
            valid_surfs = [valid_surfs[i] for i in sort_idx]
            target_depths = sorted(target_depths)

            # Step 2: Create Warped Seismic
            seismic_wheeler = np.zeros_like(seismic_2d)
            num_rows, num_cols = seismic_2d.shape
            
            for x in range(num_cols):
                # For this trace, find the original Y coordinates of each flowline
                orig_ys = []
                for surf in valid_surfs:
                    # Find y at this x. Surfaces might not exist at all x.
                    mask = (surf.path[:, 1].astype(int) == x)
                    if np.any(mask):
                        orig_ys.append(surf.path[mask, 0][0])
                    else:
                        # Interpolate boundary if missing
                        orig_ys.append(np.interp(x, surf.path[:, 1], surf.path[:, 0]))
                
                # Boundary constraints: start and end of trace
                mapping_orig = [0.0] + orig_ys + [float(num_rows - 1)]
                mapping_target = [0.0] + target_depths + [float(num_rows - 1)]
                
                # Ensure mapping_target is strictly increasing for interp1d
                # mapping_target is already sorted, but might have duplicates if means are identical
                mapping_target = np.array(mapping_target)
                mapping_orig = np.array(mapping_orig)
                
                # Add tiny epsilon to duplicates to ensure strictly increasing
                mask = np.diff(mapping_target) <= 0
                while np.any(mask):
                    mapping_target[1:][mask] += 1e-5
                    mask = np.diff(mapping_target) <= 0

                # Define Mapping: F(y_wheeler) = y_original
                f_get_orig_y = interp1d(mapping_target, mapping_orig, kind='linear', bounds_error=False, fill_value="extrapolate")
                
                # Dense Wheeler coords
                y_wheelers = np.arange(num_rows)
                y_originals = f_get_orig_y(y_wheelers)
                
                # Sample original seismic at these y_originals
                f_sample_seismic = interp1d(np.arange(num_rows), seismic_2d[:, x], kind='linear', bounds_error=False, fill_value=0)
                seismic_wheeler[:, x] = f_sample_seismic(y_originals)
                
                p.update(task, advance=1)

            # Step 3: Visualizations
            # Option A: Wheeler Flowlines (Horizontal)
            plt.figure(figsize=(10, 8))
            for depth in target_depths:
                plt.axhline(depth, color='k', linewidth=0.5, alpha=0.5)
            plt.title("Wheeler Domain: Horizontal Flowlines")
            plt.xlabel("Trace Index")
            plt.ylabel("Wheeler Time / Depth")
            plt.gca().invert_yaxis()
            plt.savefig(os.path.join(figures_dir, f"wheeler_flowlines_{extraction_mode}_{slice_idx}.png"))

            # Option B: Wheeler Seismic
            plt.figure(figsize=(10, 8))
            plt.imshow(seismic_wheeler, cmap='Greys', vmin=-amplitude_max, vmax=amplitude_max, aspect='auto')
            plt.title("Wheeler Domain: Flattened Seismic Amplitude")
            plt.colorbar(label='Amplitude')
            plt.savefig(os.path.join(figures_dir, f"wheeler_seismic_{extraction_mode}_{slice_idx}.png"))

            # Option C: Wheeler Overlay
            plt.figure(figsize=(10, 8))
            plt.imshow(seismic_wheeler, cmap='Greys', vmin=-amplitude_max, vmax=amplitude_max, aspect='auto')
            for depth in target_depths:
                plt.axhline(depth, color='r', linewidth=0.1, alpha=0.7)
            plt.title("Wheeler Domain: Flattened Seismic + Flowlines Overlay")
            plt.savefig(os.path.join(figures_dir, f"wheeler_red_overlay_{extraction_mode}_{slice_idx}.png"))
            
            console.print(f"[bold green]Wheeler Time plot generated in {figures_dir}[/bold green]")

    console.print("[bold green]Workflow completed successfully![/bold green]")

if __name__ == "__main__":
    main()