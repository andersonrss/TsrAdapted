This is a customized version of [TripoSR](https://github.com/VAST-AI-Research/TripoSR), including possible fixes and minor extensions to support reproducibility and evaluation. The model can be executed in the same way as described in the original repository, with no changes required to the inference pipeline. Some additional packages are used to perform specific tasks. If a package is required by the command line, simply install it using:

```bash
pip install <package_name>
```

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

`--out_dir` is the output directory of rendered images.

The script will automatically render a frame of each 3D object in `--obj-dir` and compare it to its corresponding ground truth image from the reference directory `--ref-dir`. The outputs include PSNR, SSIM, and LPIPS scores for each pair, exported as a .CSV file. Reference images and reconstructed objects must follow the same order and naming convention to ensure correct metric calculation. 

- **Manual Metric Evaluation**

For manual metric evaluation, you must manually capture frames of the reconstructed objects using a 3D viewer or external software, and place them in the specified directory (`--obj_dir`) before running the script. After that, run:

```bash
python metrics_manual_rendered.py --ref_dir benchmark/reference --obj_dir benchmark/reconstructed
```

Where:

`--ref_dir` is the directory containing the reference images (ground truth).

`--obj_dir` is the directory with manually object rendered images.

The outputs include PSNR, SSIM, and LPIPS scores for each pair, exported as a .CSV file. Reference images and manually framed objetcs must follow the same order and naming convention to ensure correct metric calculation. 

## Installation Note

If you encounter the following error during installation:

> Failed to build torchmcubes
> ERROR: Failed to build installable wheels for some pyproject.toml based projects (torchmcubes)

Run the following command to install `torchmcubes` manually:

```sh
pip install git+https://github.com/tatsy/torchmcubes.git@3aef8afa5f21b113afc4f4ea148baee850cbd472
```


