import torch
import os
import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.sparse import diags
import scipy
from pprint import pprint

# ARGS
parser = argparse.ArgumentParser("2D_stokes")
## Data
parser.add_argument("--type", type=str, choices=['stokes'])
parser.add_argument("--num_data", type=int, default=3000)
parser.add_argument("--basis_order", type=str)
parser.add_argument("--kind", type=str, default='train', choices=['train', 'validate'])
parser.add_argument("--ne", type=int, default=32)
parser.add_argument("--ne_exact", type=int, default=300)


args = parser.parse_args()
gparams = args.__dict__

TYPE = gparams['type']
NUM_DATA = gparams['num_data']
BASIS_ORDER = gparams['basis_order']
KIND = gparams['kind']
NUM_ELEMENT = gparams['ne']
NUM_ELEMENT_exact = gparams['ne_exact']


# Seed
if KIND=='train':
    np.random.seed(5)
elif KIND=='validate':
    np.random.seed(10)
else:
    print('error!')

# Load exact data for interpolation
mesh=np.load(f'mesh/P{BASIS_ORDER}_ne{NUM_ELEMENT_exact}_{TYPE}.npz')
pos_u_exact = mesh['pos_u']
pos_p_exact = mesh['pos_p']
pickle_file_load = f'{KIND}_P{BASIS_ORDER}_{NUM_DATA}N{NUM_ELEMENT_exact}_{TYPE}'
with open(f'data/{pickle_file_load}.pkl', 'rb') as f:
    data_exact = pickle.load(f)
    
mesh=np.load(f'mesh/P{BASIS_ORDER}_ne{NUM_ELEMENT}_{TYPE}.npz', allow_pickle=True)
_, NUM_PTS, p, gfl = mesh['ne'], mesh['ng'], mesh['p'], mesh['gfl']
idx_sol=mesh['idx_sol']
NUM_BASIS = NUM_PTS

# def f(x,y,coeff):
#     m0, m1, n0, n1, n2, n3=coeff
#     return np.stack([m0*torch.sin(n0*x+n1*y), m1*torch.cos(n2*x+n3*y)],dim=1)


data = []
for n in tqdm(range(NUM_DATA)):
    # f_value, coeff_f=f(p[:,0],p[:,1],mesh[f'{KIND}_coeff_fs'][n])
    coeff_f=mesh[f'{KIND}_coeff_fs'][n]
    interp_u1=scipy.interpolate.CloughTocher2DInterpolator(pos_u_exact,data_exact[0][n],fill_value=0)
    interp_u2=scipy.interpolate.CloughTocher2DInterpolator(pos_u_exact,data_exact[1][n],fill_value=0)
    interp_p=scipy.interpolate.CloughTocher2DInterpolator(pos_p_exact,data_exact[2][n],fill_value=0)

    coeff_u1=interp_u1(p[idx_sol[0]])
    coeff_u2=interp_u2(p[idx_sol[1]])
    coeff_p=interp_p(p[idx_sol[2]])
    data.append([coeff_u1, coeff_u2, coeff_p, coeff_f])
data= np.array(data, dtype=object)

pickle_file_save = f'{KIND}_P{BASIS_ORDER}_{NUM_DATA}N{NUM_ELEMENT}_{TYPE}'
with open(f'data/{pickle_file_save}.pkl', 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
