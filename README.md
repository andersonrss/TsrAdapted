This is a customized version of [TripoSR](https://github.com/VAST-AI-Research/TripoSR), including possible fixes and minor extensions to support reproducibility and evaluation. 

## Key Modifications

- **Fixed cross-device (CPU/GPU) tensor inconsistencies.**
- **Resolved issues in the `--bake-texture` pipeline** to enable textured mesh generation.
- **Added 2D quality evaluation metrics**:
  - PSNR (Peak Signal-to-Noise Ratio)  
  - SSIM (Structural Similarity Index)  
  - LPIPS (Learned Perceptual Image Patch Similarity)
- **Integrated PyTorch Profiler** for layer-wise performance and memory profiling.

## Running Quality Metrics

To evaluate the visual fidelity of reconstructed objects, this repository includes scripts to compute PSNR, SSIM, and LPIPS metrics. 

- **Automatic Metric Evaluation**

To compute 2D reconstruction quality metrics (PSNR, SSIM, LPIPS) automatically from reconstructed objects:

```bash
python metrics_auto_rendered.py --obj_dir benchmark/objs --ref_dir benchmark/reference --out_dir benchmark/rendered
```

`--obj_dir` is the directory containing the reconstructed .obj files.

`--ref_dir` is the directory containing the reference images (ground truth).

`--out_dir` is the directory to save rendered images.

The script will automatically render a frame of each 3D object in `--ref-dir` and compare it to its corresponding ground truth image from the reference directory. The outputs include PSNR, SSIM, and LPIPS scores for each pair and summary statistics.

- **Manual Metric Evaluation**




```sh
pip install git+https://github.com/tatsy/torchmcubes.git@3aef8afa5f21b113afc4f4ea148baee850cbd472
```

## Installation Note

If you encounter the following error during installation:

> Failed to build torchmcubes
> ERROR: Failed to build installable wheels for some pyproject.toml based projects (torchmcubes)

Run the following command to install `torchmcubes` manually:

```sh
pip install git+https://github.com/tatsy/torchmcubes.git@3aef8afa5f21b113afc4f4ea148baee850cbd472
```


