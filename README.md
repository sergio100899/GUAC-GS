# GUAC-GS: Guided Uncertainty-Adaptive Cooperation for 3D Gaussian Splatting

*Based on the official 3D Gaussian Splatting implementation by Inria & MPII (Kerbl et al., 2023)*  

📄 [Original 3DGS Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_high.pdf) •  
🌐 [GUAC-GS Project Page](https://github.com/sergio100899/guac-3dgs)

---

## Overview

**GUAC-GS (Guided Uncertainty-Adaptive Cooperation for 3D Gaussian Splatting)** introduces an *adaptive uncertainty weighting framework* for **photometric-geometric optimization** in 3D Gaussian Splatting.  
It enhances training stability and scene reconstruction quality through a **four-stage progression of loss adaptation**.
This yields **smoother convergence**, **reduced overfitting**, and **consistent view synthesis quality**,  
while preserving the speed and real-time capabilities of the original 3DGS renderer.

---

## Key Contributions

- 🔹 **Uncertainty-driven dynamic loss weighting**  
  Progressive control of per-loss variance and temporal stabilization.

- 🔹 **Progress-aware regulation**  
  Distinguishes beneficial variance (learning progress) from noisy oscillations.

- 🔹 **Collaborative weighting mechanism**  
  Ensures that the total optimization energy remains balanced — losses cooperate, not compete.

- 🔹 **Seamless integration**  
  Fully compatible with *3DGS* training pipeline, TensorBoard logging, and real-time rendering.

---

## Setup

### Requirements
- CUDA-ready GPU (Compute ≥ 7.0, ≥ 12 GB VRAM)
- Python ≥ 3.9, PyTorch ≥ 2.0, CUDA 12.8
- uv

### Installation

```bash
# Clone including submodules
git clone --recursive https://github.com/sergio100899/guac-3dgs.git
cd guac-3dgs
uv venv && source venv/bin/activate
uv sync
```

---

## Training

Train with adaptive uncertainty control:

```bash
python train.py -s <path_to_COLMAP_dataset>
```

---

## Evaluation

```bash
python render.py -m <path_to_model>
python metrics.py -m <path_to_model>
```

---

## Citation

If you use this work, please cite both GUAC-GS and the original 3D Gaussian Splatting paper:

@misc{ortiz2025guacgs,
  title   = {GUAC-GS: Gaussian Uncertainty-Aware Cooperative 3D Gaussian Splatting},
  author  = {Sergio Ortiz},
  year    = {2025},
  howpublished = {\url{https://github.com/sergio100899/guac-3dgs}},
  note    = {Extension of 3D Gaussian Splatting (Kerbl et al., ACM TOG 2023)}
}

@Article{kerbl3Dgaussians,
  author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  journal      = {ACM Transactions on Graphics},
  number       = {4},
  volume       = {42},
  month        = {July},
  year         = {2023},
  url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}

---

## License

This repository builds upon the original
3D Gaussian Splatting by Inria & MPII (© 2023 GRAPHDECO).
That software is distributed for non-commercial, research and evaluation use
under its LICENSE.md
 terms.

All modifications and GUAC extensions
© 2025 Sergio Ortiz – provided under the same conditions
for non-commercial research use only.

For commercial inquiries, contact george.drettakis@inria.fr

---

## Acknowledgments

We thank the GRAPHDECO research group for releasing their implementation
and the broader community for continuing to extend Gaussian Splatting
toward new directions in efficiency, robustness, and uncertainty modeling.