# E-GRASP-Net

**Edge-Guided Region-Awareness Semi-Pyramid Network for Deformable Medical Image Registration**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)](https://pytorch.org/)

---

## 📋 Overview

E-GRASP-Net is an unsupervised deformable medical image registration framework that addresses three critical challenges in medical image analysis:

1. **Boundary Preservation** - Edge-Guided (EG) module for structural boundary enhancement
2. **Regional Focus** - Region Awareness (RA) module for anatomically salient region identification
3. **Multi-scale Efficiency** - Semi-Pyramid Network (SPN) for hierarchical deformation estimation

Our method achieves superior registration accuracy while maintaining computational efficiency and deformation regularity.

---

## 🎯 Key Features

| Module | Description | Mathematical Formulation |
|--------|-------------|-------------------------|
| **EG** | Sobel edge detection with tensor concatenation | $\tilde{I}(\mathbf{x}) = [I(\mathbf{x}), \|\nabla I(\mathbf{x})\|_2]^\top$ |
| **RA** | CAM-guided mutual cross-attention | $\mathbf{A}' = \text{Softmax}(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}) \odot (\mathbf{c}_M \mathbf{c}_F^\top)$ |
| **SPN** | Hierarchical deformation composition | $\phi(\mathbf{x}) = \mathbf{x} + \sum_{l=1}^{L} \phi_l(\mathbf{x} + \sum_{j=1}^{l-1} \phi_j(\mathbf{x}))$ |

---

## 📊 Performance Results

### OASIS Dataset
| Method | DSC ↑ | NJD % ↓ | Time (s) ↓ |
|--------|-------|---------|------------|
| VoxelMorph | 0.779 | 0.33 | 0.35 |
| TransMorph | 0.799 | 0.21 | 0.34 |
| RDP | 0.811 | 0.21 | 0.48 |
| **E-GRASP-Net (Ours)** | **0.821** | **0.19** | **0.21** |

### LPBA40 Dataset
| Method | DSC ↑ | NJD % ↓ | Time (s) ↓ |
|--------|-------|---------|------------|
| VoxelMorph | 0.676 | 0.29 | 0.35 |
| TransMorph | 0.661 | 0.21 | 0.35 |
| RDP | 0.728 | 0.29 | 0.47 |
| **E-GRASP-Net (Ours)** | **0.748** | **0.15** | **0.19** |

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA 11.7+ (for GPU acceleration)
- 16GB+ GPU memory recommended

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/E-GRASP-Net.git
cd E-GRASP-Net

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__} with CUDA {torch.cuda.is_available()}')"
```

### Requirements
```txt
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
scipy>=1.10.0
nibabel>=5.0.0
SimpleITK>=2.2.0
opencv-python>=4.7.0
matplotlib>=3.7.0
tqdm>=4.65.0
pyyaml>=6.0
tensorboard>=2.12.0
scikit-image>=0.20.0
pandas>=2.0.0
```

---

## 📁 Dataset Preparation

### Supported Datasets
- **OASIS**: 405 T1-weighted brain MRI volumes (28 anatomical ROIs)
- **LPBA40**: 40 high-resolution T1-weighted MRI scans (56 anatomical ROIs)

### Directory Structure
```
/home/phd/datasets/
├── oasis/
│   ├── train_images.txt
│   ├── val_images.txt
│   ├── test_images.txt
│   ├── images/          # NIfTI files
│   └── segmentations/   # Label maps
└── lpba40/
    ├── train_images.txt
    ├── val_images.txt
    ├── test_images.txt
    ├── images/
    └── segmentations/
```

### Preprocessing
```bash
# Run preprocessing script
python scripts/preprocess_data.py \
    --dataset oasis \
    --root /home/phd/datasets/oasis/ \
    --output-size 160 192 224 \
    --voxel-size 1.0

# Preprocessing includes:
# - Skull stripping (FreeSurfer)
# - Affine alignment
# - Intensity normalization
# - Resampling to isotropic 1mm³
```

---

## 🏋️ Training

### Single GPU Training
```bash
# Train on OASIS dataset
python scripts/train.py \
    --config configs/oasis_config.yaml \
    --dataset oasis \
    --gpu 0 \
    --batch-size 1 \
    --epochs 500

# Train on LPBA40 dataset
python scripts/train.py \
    --config configs/lpba40_config.yaml \
    --dataset lpba40 \
    --gpu 0 \
    --batch-size 1 \
    --epochs 500
```

### Multi-GPU Training
```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    scripts/train.py \
    --config configs/oasis_config.yaml \
    --dataset oasis \
    --distributed
```

### Configuration Options
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--learning-rate` | 1e-4 | Initial learning rate |
| `--lambda-reg` | 1.0 | Regularization weight |
| `--spn-start-layer` | 3 | SPN deformation start layer |
| `--window-size` | 9 | LNCC local window size |

---

## 🧪 Testing & Evaluation

### Inference
```bash
# Test on OASIS
python scripts/test.py \
    --checkpoint experiments/oasis/checkpoints/best_model.pth \
    --dataset oasis \
    --gpu 0

# Test on LPBA40
python scripts/test.py \
    --checkpoint experiments/lpba40/checkpoints/best_model.pth \
    --dataset lpba40 \
    --gpu 0
```

### Evaluation Metrics
```bash
# Compute DSC and NJD
python scripts/evaluate.py \
    --predictions experiments/oasis/results/ \
    --ground-truth /home/phd/datasets/oasis/segmentations/ \
    --metrics dsc njd
```

### Visualize Results
```bash
# Generate deformation field visualizations
python scripts/visualize_results.py \
    --input moving_image.nii.gz \
    --fixed fixed_image.nii.gz \
    --checkpoint best_model.pth \
    --output visualization/
```

---

## 🏗️ Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      E-GRASP-Net Architecture                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Moving Image ──→ [EG Module] ──→ ┌──────────────┐              │
│  (I_M)          [I_M, ∇I_M]      │              │              │
│                                  │   Dual-Stream │              │
│  Fixed Image ──→ [EG Module] ──→ │   Encoder    │              │
│  (I_F)          [I_F, ∇I_F]      │              │              │
│                                  └──────┬───────┘              │
│                                         │                       │
│                                  ┌──────▼───────┐              │
│                                  │   RA Module  │              │
│                                  │  [CAM + MCA] │              │
│                                  └──────┬───────┘              │
│                                         │                       │
│                                  ┌──────▼───────┐              │
│                                  │   SPN Decoder│              │
│                                  │  [Layer 3→1] │              │
│                                  └──────┬───────┘              │
│                                         │                       │
│                                  ┌──────▼───────┐              │
│                                  │ Deformation  │              │
│                                  │    Field φ   │              │
│                                  └──────────────┘              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Module Details

#### Edge-Guided (EG) Module
- 3D Sobel operator for gradient computation
- Edge magnitude: $I_e(\mathbf{x}) = \sqrt{G_x^2 + G_y^2 + G_z^2}$
- Input channels: 2 (intensity + edge)

#### Region Awareness (RA) Module
- CAM generation via 1×1×1 convolution
- Patch-level saliency aggregation
- Mutual cross-attention with CAM modulation

#### Semi-Pyramid Network (SPN)
- Deformation estimation from layer 3 (not layer 1)
- Hierarchical field composition
- Spatial Transformer Network for warping

---

## 📈 Loss Functions

### Total Loss
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{sim}}(I_F, I_M(\phi)) + \lambda \mathcal{L}_{\text{reg}}(\phi)$$

### Similarity Loss (LNCC)
$$\mathcal{L}_{\text{sim}} = \sum_{v \in \Omega} \frac{\left[\sum_{v_i}(I_F(v_i)-\bar{I}_F(v))(I_M(\phi)(v_i)-\overline{I_M(\phi)}(v))\right]^2}{\left[\sum_{v_i}(I_F(v_i)-\bar{I}_F(v))^2\right]\left[\sum_{v_i}(I_M(\phi)(v_i)-\overline{I_M(\phi)}(v))^2\right]}$$

### Regularization Loss
$$\mathcal{L}_{\text{reg}}(\phi) = \sum_{v \in \Omega} \|\nabla \phi(v)\|^2$$

---

## 📂 Repository Structure

```
E-GRASP-Net/
├── README.md
├── LICENSE
├── requirements.txt
├── setup.py
├── configs/
│   ├── default_config.py
│   ├── oasis_config.yaml
│   └── lpba40_config.yaml
├── data/
│   ├── dataset.py
│   ├── dataloader.py
│   ├── preprocessing.py
│   └── datasets/
│       ├── oasis.py
│       └── lpba40.py
├── models/
│   ├── e_grasp_net.py
│   ├── encoder.py
│   ├── decoder.py
│   └── modules/
│       ├── edge_guided.py
│       ├── region_awareness.py
│       └── semi_pyramid.py
├── losses/
│   ├── similarity_loss.py
│   ├── regularization_loss.py
│   └── total_loss.py
├── metrics/
│   ├── dice.py
│   ├── jacobian.py
│   └── evaluation.py
├── utils/
│   ├── logger.py
│   ├── visualizer.py
│   ├── checkpoint.py
│   └── sobel_operator.py
├── scripts/
│   ├── train.py
│   ├── test.py
│   ├── evaluate.py
│   ├── preprocess_data.py
│   └── visualize_results.py
├── experiments/
│   ├── oasis/
│   │   ├── checkpoints/
│   │   ├── logs/
│   │   └── results/
│   └── lpba40/
│       ├── checkpoints/
│       ├── logs/
│       └── results/
└── notebooks/
    ├── data_exploration.ipynb
    ├── model_analysis.ipynb
    └── visualization.ipynb
```

---

## 🔬 Ablation Studies

| Configuration | EG | RA | MA | SPN | DSC (OASIS) | DSC (LPBA40) |
|--------------|:--:|:--:|:--:|:--:|:-----------:|:------------:|
| Baseline | ✗ | ✗ | ✗ | ✗ | 0.793 | 0.723 |
| +EG | ✓ | ✗ | ✗ | ✗ | 0.796 | 0.726 |
| +RA | ✗ | ✓ | ✗ | ✗ | 0.801 | 0.730 |
| +MA | ✗ | ✗ | ✓ | ✗ | 0.798 | 0.729 |
| +SPN | ✗ | ✗ | ✗ | ✓ | 0.801 | 0.732 |
| **Full Model** | **✓** | **✓** | **✓** | **✓** | **0.821** | **0.748** |

**Key Findings:**
- RA module provides the largest single-component improvement (+0.008 OASIS)
- SPN enables efficient multi-scale refinement
- Full model achieves +0.028 DSC improvement over baseline

---

## 📊 Computational Efficiency

| Metric | E-GRASP-Net | VoxelMorph | TransMorph | RDP |
|--------|-------------|------------|------------|-----|
| Parameters (M) | 4.3 | 0.27 | 46.71 | 52.3 |
| GMACs | 427 | 304 | 663 | 4162 |
| GPU Memory (GB) | 14.21 | 9.97 | 12.92 | 8.82 |
| Inference Time (s) | 0.21 | 0.35 | 0.34 | 0.48 |

---

## 📄 Citation

If you use E-GRASP-Net in your research, please cite:

```bibtex
@article{anwar2025egraspnet,
  title={E-GRASP-Net: Edge-Guided Region-Awareness Semi-Pyramid Network for Deformable Medical Image Registration},
  author={Anwar, Muhammad},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2025},
  volume={XX},
  number={XX},
  pages={XX--XX},
  doi={10.1109/JBHI.2025.XXXXXXX}
}
```

### Related Publications
```bibtex
@article{anwar2025sthra,
  title={STHRA: Selective Transformer Hierarchical Reciprocal Attention-Based Deformable Medical Image Registration},
  author={Anwar, Muhammad and Yan, Zhenyu and Cao, Wei},
  journal={Multimedia Systems},
  year={2025}
}

@article{anwar2025enhancing,
  title={Enhancing 3D Medical Image Registration with Cross Attention, Residual Skips, and Cascade Attention},
  author={Anwar, Muhammad and He, Zhang and Cao, Wei},
  journal={Intelligent Data Analysis},
  year={2025}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **OASIS Dataset**: [Open Access Series of Imaging Studies](https://www.oasis-brains.org)
- **LPBA40 Dataset**: [Laboratory of Neuro Imaging](http://www.loni.usc.edu/Atlases/)
- **FreeSurfer**: [NeuroImage Processing Software](https://surfer.nmr.mgh.harvard.edu)

### Funding
- Guangdong Major Project of Basic and Applied Basic Research (No. 2023B0303000009)
- Princess Nourah bint Abdulrahman University Researchers Supporting Project (PNURSP2025R410)

---

## 📞 Contact

**Corresponding Author:** Muhammad Anwar  
**Affiliation:** College of Mechatronics and Control Engineering, Shenzhen University, China  
**Email:** muhammad.anwar@szu.edu.cn  
**ORCID:** 0000-000X-EFGH-ABCD

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## ⚠️ Disclaimer

This software is for research purposes only. It is not intended for clinical use or diagnostic purposes. Users are responsible for ensuring compliance with all applicable regulations and ethical guidelines when using medical imaging data.

---

## 📅 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-01 | Initial release with OASIS and LPBA40 support |
| 1.1.0 | 2025-03 | Added multi-GPU training support |
| 1.2.0 | 2025-06 | Improved visualization tools and documentation |

---

<div align="center">

**⭐ If you find this project helpful, please give it a star!**

[![GitHub stars](https://img.shields.io/github/stars/yourusername/E-GRASP-Net?style=social)](https://github.com/yourusername/E-GRASP-Net)
[![GitHub forks](https://img.shields.io/github/forks/yourusername/E-GRASP-Net?style=social)](https://github.com/yourusername/E-GRASP-Net)

</div>
