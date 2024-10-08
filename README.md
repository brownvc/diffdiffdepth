
# Differentiable Diffusion for Dense Depth Estimation from Multi-view Images
[Numair Khan](https://cs.brown.edu/~nkhan6)<sup>1</sup>,
[Min H. Kim](http://vclab.kaist.ac.kr/minhkim/)<sup>2</sup>,
[James Tompkin](http://www.jamestompkin.com)<sup>1</sup><br>
<sup>1</sup>Brown, <sup>2</sup>KAIST<br>
CVPR 2021

<img src="http://visual.cs.brown.edu/projects/diffdiffdepth-webpage/img/teaser_dino_rgb.png" width=200px><img src="http://visual.cs.brown.edu/projects/diffdiffdepth-webpage/img/teaser_dino_depth.gif" width=200px>

<img src="http://visual.cs.brown.edu/projects/diffdiffdepth-webpage/img/teaser_lego_rgb.png" width=200px><img src="http://visual.cs.brown.edu/projects/diffdiffdepth-webpage/img/teaser_lego_depth.gif" width=200px>

### [Paper](http://arxiv.org/abs/2106.08917) | [Supplemental](http://visual.cs.brown.edu/projects/diffdiffdepth-webpage/docs/khan2021_diffdiffdepth_supp.pdf) | [Presentation Video](http://visual.cs.brown.edu/projects/diffdiffdepth-webpage/video/diffdiffdepth_cvpr2021.mp4) 

## Citation
If you use this code in your work, please cite our paper:

```
@article{khan2021diffdiff,
  title={Differentiable Diffusion for Dense Depth Estimation from Multi-view Images
  author={Numair Khan, Min H. Kim, James Tompkin},
  journal={Computer Vision and Pattern Recognition},
  year={2021}
}
```
Code in pytorch_ssim is from `https://github.com/Po-Hsun-Su/pytorch-ssim`

## Included in this PyTorch code

* A differentiable Gaussian splatting algorithm based on radiative transport that models scene occlusion.
* A differentiable implementation of Szeliski's SIGGRAPH 2006 paper [Locally Adaptive Hierarchical Basis Preconditioning](https://dl.acm.org/doi/10.1145/1179352.1142005) for solving partial differential equations (PDEs) iteratively with very few steps.  
* A tile-based diffusion solver that combines the previous two approaches to scale to large images efficiently.

## Running the Code
* [Environment Setup](#environment)
* [Multiview Stereo](#mvs)
* [Lightfield Images](#lightfields)
* [Troubleshooting](#troubleshooting)

### Environment Setup
The code has been tested with Python3.6 using Pytorch=1.5.1.

The provided setup file can be used to install all dependencies and create a conda environment `diffdiffdepth`:

```$ conda  env create -f environment.yml```

```$ conda activate diffdiffdepth```

### Multiview Stereo
To run the code on multi-view stereo images you will first need to generate poses using [COLMAP](https://colmap.github.io). Once you have these, run the optimization by calling `run_mvs.py`:

```$ python run_mvs.py --input_dir=<COLMAP_project_directory> --src_img=<target_img> --output_dir=<output_directory>```

where `<target_img>` is the name of the image you want to compute depth for. Run `python run_mvs.py -h` to view additional optional arguments.

Example usage:

```$ python run_mvs.py --input_dir=colmap_dir --src_img=img0.png --output_dir=./results```

### Lightfield Images

Input sparse point clouds are derived from our light field depth estimation work code here: https://github.com/brownvc/lightfielddepth

### Troubleshooting

_We will add to this section as issues arise._
