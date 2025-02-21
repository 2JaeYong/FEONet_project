# Finite Element Operator Network for Solving Elliptic-type parametric PDEs

This is an official FeniCS and PyTorch implementation of the (1) [Finite Element Operator Network for Solving Elliptic-type parametric PDEs](https://arxiv.org/abs/2308.04690) paper and (2) [Error analysis for finite element operator learning methods for solving parametric second-order elliptic PDEs](https://arxiv.org/abs/2404.17868)  by Jae Yong Lee, Seungchan Ko, Youngjoon Hong.


## 1. 1D_variable_coefficient (p.13 & eq.(3.3) in the paper (1)).

In the folder "1D_variable_coefficient":

### Step1 - Assemble and save the matrices
To assemble the matrices, use the FeniCS-based code
```
FeniCS_assemble.ipynb
```

### Step2 - Save the train and test data by interpolation.
Using the 'create_data_interpol.py' code, you can interpolate using exact data.
```
python3 create_data_interpol.py --type varcoeff --num_data 1000 --basis_order 2 --kind train --ne 32 --ne_exact 300
```
```
python3 create_data_interpol.py --type varcoeff --num_data 1000 --basis_order 2 --kind validate --ne 32 --ne_exact 300
```

### Step3 - Train the FEONet
```
python3 -u FEONet_1D.py test --seed 0 --gpu 0 --type varcoeff --num_train_data 1000 --num_validate_data 1000 --basis_order 2 --ne 32 --model CNN1D --blocks 4 --ks 5 --filters 32 --epochs 100000 > train/log.out
```



## 2. 1D_nonlinear_Burgers (p.13 & eq.(3.4) in the paper(1)).

In the folder "1D_nonlinear_Burgers":

### Step1 - Assemble and save the matrices
To assemble the matrices, use the FeniCS-based code
```
FeniCS_assemble.ipynb
```

### Step2 - Save the train and test data by interpolation.
Using the 'create_data_interpol.py' code, you can interpolate using exact data.
```
python3 create_data_interpol.py --type burgers --num_data 3000 --basis_order 1 --kind train --ne 128 --ne_exact 512
```
```
python3 create_data_interpol.py --type burgers --num_data 3000 --basis_order 1 --kind validate --ne 128 --ne_exact 512
```

### Step3 - Train the FEONet
```
python3 -u FEONet_1D.py test --seed 0 --gpu 0 --type burgers --num_train_data 3000 --num_validate_data 3000 --basis_order 1 --ne 128 --model CNN1D --blocks 4 --ks 5 --filters 32 --epochs 100000 > train/log.out
```


## 3. 2D_circlehole (p.19 & eq.(5.2) in the paper(2)).

In the folder "2D_circlehole":

### Step1 - Assemble and save the matrices
To assemble the matrices, use the FeniCS-based code
```
FeniCS_assemble.ipynb
```

### Step2 - Save the train and test data by interpolation.
Using the 'create_data_interpol.py' code, you can interpolate using exact data.
```
python3 create_data_interpol.py --type circlehole --num_data 1000 --basis_order 2 --kind train --ne 555 --ne_exact 1530
```
```
python3 create_data_interpol.py --type circlehole --num_data 1000 --basis_order 2 --kind validate --ne 555 --ne_exact 1530
```

### Step3 - Train the FEONet
W/o preconditioning
```
python3 -u FEONet_2D.py test --seed 0 --gpu 1 --type circlehole --num_train_data 1000 --num_validate_data 1000 --basis_order 2 --ne 555 --model CNN2D --resol_in 20 --blocks 4 --ks 5 --filters 32 --epochs 100000 > train/log.out
```

With preconditioning
```
python3 -u FEONet_2D.py test_precond --seed 0 --gpu 2 --type circlehole --num_train_data 1000 --num_validate_data 1000 --basis_order 2 --ne 555 --model CNN2D --resol_in 20 --blocks 4 --ks 5 --filters 32 --epochs 100000 --do_precond > train/log_precond.out
```

## 4. 2D_stokes (p.15 & eq.(3.7) in the paper(1)).

In the folder "2D_stokes":

### Step1 - Assemble and save the matrices
To assemble the matrices, use the FeniCS-based code
```
FeniCS_assemble.ipynb
```

### Step2 - Save the train and test data by interpolation.
Using the 'create_data_interpol.py' code, you can interpolate using exact data.
```
python3 create_data_interpol.py --type stokes --num_data 1000 --basis_order 2x1 --kind train --ne 72 --ne_exact 450
```
```
python3 create_data_interpol.py --type stokes --num_data 1000 --basis_order 2x1 --kind validate --ne 72 --ne_exact 450
```

### Step3 - Train the FEONet
```
python3 -u FEONet_2D.py test_re --seed 1 --gpu 4 --type stokes --num_train_data 1000 --num_validate_data 1000 --basis_order 2x1 --ne 72 --model CNN2D --resol_in 50 --blocks 4 --ks 5 --filters 32 --epochs 100000 > train/log.out
```
## Citations

```
@article{lee2023finite,
  title={Finite Element Operator Network for Solving Elliptic-type parametric PDEs},
  author={Lee, Jae Yong and Ko, Seungchan and Hong, Youngjoon},
  journal={To apper in SIAM Journal on Scientific Computing},
  year={2025}
}
@article{hong2024error,
  title={Error analysis for finite element operator learning methods for solving parametric second-order elliptic PDEs},
  author={Hong, Youngjoon and Ko, Seungchan and Lee, Jaeyong},
  journal={arXiv preprint arXiv:2404.17868},
  year={2024}
}
```