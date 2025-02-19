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
parser = argparse.ArgumentParser("1D_variable_coefficient")
## Data
parser.add_argument("--type", type=str, choices=['varcoeff','burgers','circlehole'])
parser.add_argument("--num_data", type=int, default=3000)
parser.add_argument("--basis_order", type=int, help='P1->d=1, P2->d=2')
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
p_exact = mesh['p']
pickle_file_load = f'{KIND}_P{BASIS_ORDER}_{NUM_DATA}N{NUM_ELEMENT_exact}_{TYPE}'
with open(f'data/{pickle_file_load}.pkl', 'rb') as f:
    data_exact = pickle.load(f)

mesh=np.load(f'mesh/P{BASIS_ORDER}_ne{NUM_ELEMENT}_{TYPE}.npz')
_, NUM_PTS, p, c, gfl = mesh['ne'], mesh['ng'], mesh['p'], mesh['c'], mesh['gfl']
NUM_BASIS = NUM_PTS

    
def c_input(x,coeff):
    m0, m1, n0, n1=coeff
    return m0*np.sin(n0*x) + m1*np.cos(n1*x), coeff

data = []
for n in tqdm(range(NUM_DATA)):
    c_value, coeff_c=c_input(p,mesh[f'{KIND}_coeff_inputs'][n])
    interp=scipy.interpolate.CubicSpline(p_exact[p_exact.argsort()],data_exact[n][0][p_exact.argsort()])
    coeff_u=interp(p)
    data.append([coeff_u, c_value, coeff_c])
data= np.array(data, dtype=object)

pickle_file_save = f'{KIND}_P{BASIS_ORDER}_{NUM_DATA}N{NUM_ELEMENT}_{TYPE}'
with open(f'data/{pickle_file_save}.pkl', 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)