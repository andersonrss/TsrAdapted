This is a customized version of [TripoSR](https://github.com/VAST-AI-Research/TripoSR), including possible fixes and minor extensions to support reproducibility and evaluation. 

## 🔧 Key Modifications

- ✅ **Fixed cross-device (CPU/GPU) tensor inconsistencies.**
- 🖼️ **Resolved issues in the `--bake-texture` pipeline** to enable textured mesh generation.
- 📏 **Added 2D quality evaluation metrics**:
  - PSNR (Peak Signal-to-Noise Ratio)  
  - SSIM (Structural Similarity Index)  
  - LPIPS (Learned Perceptual Image Patch Similarity)
- 📊 **Integrated PyTorch Profiler** for layer-wise performance and memory profiling.

## ⚙️ Installation Note

If you encounter the following error during installation:

> Failed to build torchmcubes
> ERROR: Failed to build installable wheels for some pyproject.toml based projects (torchmcubes)

Run the following command to install `torchmcubes` manually:

```sh
pip install git+https://github.com/tatsy/torchmcubes.git@3aef8afa5f21b113afc4f4ea148baee850cbd472
```
