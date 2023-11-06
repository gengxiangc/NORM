# Learning Neural Operators on Riemannian Manifolds

This repository contains code accompanying our paper ["Learning Neural Operators on Riemannian Manifolds"](https://arxiv.org/abs/2302.08166).

![images](img/fig_NORM_method.svg)


## Dependencies & packages
Dependencies:
* Python (tested on 3.8.11)
* PyTorch (tested on 1.8.0)

Additionally, we need an open-source Python package Lapy (https://github.com/Deep-MI/LaPy/tree/main) for differential geometry on triangle and tetrahedra meshes, which is used to calculate LBO basis. If you fail to install it, try to add the `lapy` folder included in our source code into your path.

## Data
The datasets of Case1-Case3 can be found in `datasets` folder. You can download the dataset(s) of Case4 and Case5 from [here](https://drive.google.com/drive/folders/1jS7YwY1Gs7rGOm1VXrkN_KvTzxGxTw6G?usp=sharing). 

```
Case1-DarcyFlow.mat
├── Input: `c_field`(1200*2290)
└── Output: `u_field`(1200*2290)
---------------------------------------------
Case2-Turbulence.mat
├── Input: `Input`(400*2673)
└── Output: `Output`(400*2673)
---------------------------------------------
Case3-HeatTransfer.mat
├── Input: `input`(300*186)
└── Output: `output`(300*7199)
---------------------------------------------
Case4-Composites.mat
├── Input: `T_field`(1200*8232)
└── Output: `D_field`(1200*8232)
---------------------------------------------
Case5-BloodFlow.mat
├── Input: `BC_time`(500*121*6)
└── Output: `velocity_x`(500*1656*121),`velocity_y`,`velocity_z`
```

## Usage

For all cases, you can directly run the codes by executing `main.py` to quickly obtain the results. Note that each experiment is repeated five times, the same setup as in our paper. Each case also retains the setting of hyperparameters in the paper.
```
python main.py 
```
Additionally, we provide the `Calculate_LBO_basis.py` in `datasets` folder to calculate the LBO basis for Case3, Case4 and Case5. The calculation of Case1 and Case2 are embedded in the corresponding `main.py`.


## Results
### Case1, Case2 and Case3
![images](img/Toycase.png)
---------------------------------------------------

----------------------------------------------------
### Video for Case5
https://github.com/gengxiangc/NORM/assets/45565440/3a4e6d53-237c-4170-aa96-dff96ab7ca6a

## Publication
If you found this repository useful, please consider citing our paper:
```
@article{chen2023laplace,
  title={Laplace neural operator for complex geometries},
  author={Chen, Gengxiang and Liu, Xu and Li, Yingguang and Meng, Qinglu and Chen, Lu},
  journal={arXiv preprint arXiv:2302.08166},
  year={2023}
}
```