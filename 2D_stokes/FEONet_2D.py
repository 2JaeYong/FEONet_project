import torch
import time
import datetime
import os
import argparse
import pickle 
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import random
import sys
sys.path.append(os.path.abspath('..'))
from models import *


# ARGS
parser = argparse.ArgumentParser("2D_circlehole")
parser.add_argument('name', type=str, help='experiments name')
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--use_squeue", action='store_true')
parser.add_argument("--gpu", type=int, default=0)

## Data
parser.add_argument("--type", type=str, choices=['stokes'])
parser.add_argument("--num_train_data", type=int, default=3000)
parser.add_argument("--num_validate_data", type=int, default=3000)
parser.add_argument("--basis_order", type=str)
parser.add_argument("--ne", type=int, default=32)

## Train parameters
parser.add_argument("--model", type=str, default='NetA', choices=['FCNN', 'CNN1D', 'CNN2D']) 
parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--resol_in", type=int, default=None)
parser.add_argument("--blocks", type=int, default=0)
parser.add_argument("--ks", type=int, default=5)
parser.add_argument("--filters", type=int, default=32, choices=[8, 16, 32, 64])
parser.add_argument("--act", type=str, default='silu')
parser.add_argument("--epochs", type=int, default=80000)
parser.add_argument('--do_precond', action='store_true')

args = parser.parse_args()
gparams = args.__dict__

NAME=gparams['name']

## Random seed
random_seed=gparams['seed']
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

## Choose gpu
if not gparams['use_squeue']:
    gpu_id=str(gparams['gpu'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    use_cuda = torch.cuda.is_available()
    print("Is available to use cuda? : ",use_cuda)
    if use_cuda:
        print("-> GPU number ",gpu_id)

##Equation
TYPE = gparams['type']
NUM_TRAIN_DATA = gparams['num_train_data']
NUM_VALIDATE_DATA = gparams['num_validate_data']
BASIS_ORDER = gparams['basis_order']
NUM_ELEMENT = gparams['ne']

mesh=np.load(f'mesh/P{BASIS_ORDER}_ne{NUM_ELEMENT}_{TYPE}.npz', allow_pickle=True)
_, NUM_PTS, p = mesh['ne'], mesh['ng'], mesh['p']
NUM_BASIS = NUM_PTS
IDX_SOL = mesh['idx_sol']

MATRIX=mesh['matrix']
MATRIX = torch.tensor(MATRIX).cuda().float()

DO_PRECOND=gparams['do_precond']
if DO_PRECOND==True:
    PRECOND=mesh['precond']
    PRECOND = torch.tensor(PRECOND).cuda().float()

#Model
models = {
          'CNN1D': CNN1D,
          'CNN2D': CNN2D,
          }
MODEL = models[gparams['model']]
BLOCKS = int(gparams['blocks'])
KERNEL_SIZE = int(gparams['ks'])
FILTERS = int(gparams['filters'])
PADDING = (KERNEL_SIZE - 1)//2
ACT = gparams['act']
if gparams['model']=='CNN2D':
    RESOL_IN=gparams['resol_in']


#Train
EPOCHS = int(gparams['epochs'])
D_in = 2
D_out = NUM_BASIS

if gparams['batch_size'] is None: #Full-batch
    BATCH_SIZE_train = NUM_TRAIN_DATA
    BATCH_SIZE_validate = NUM_VALIDATE_DATA
else:
    BATCH_SIZE_train = gparams['batch_size']
    BATCH_SIZE_validate = gparams['batch_size']
    
#Save file
FOLDER = f'P{BASIS_ORDER}_{gparams["model"]}_{NAME}'
PATH = os.path.join('train', FOLDER)

# CREATE PATHING
if os.path.isdir(PATH) == False: os.makedirs(PATH);
elif os.path.isdir(PATH) == True:
    print("\n\nPATH ALREADY EXISTS!\n\nEXITING\n\n")
    # exit()


class Dataset(Dataset):
    def __init__(self, mesh, kind='train', num_train=False):
        self.kind=kind
        self.num_train=num_train
        pickle_file_load = f'{self.kind}_P{BASIS_ORDER}_{NUM_TRAIN_DATA}N{NUM_ELEMENT}_{TYPE}'
        with open(f'data/{pickle_file_load}.pkl', 'rb') as f:
            self.data = pickle.load(f)
        self.load_vector = mesh[f'{kind}_load_vectors']
    def __getitem__(self, idx):
        coeff_u1 = torch.FloatTensor(self.data[idx,0])
        coeff_u2 = torch.FloatTensor(self.data[idx,1])
        coeff_p = torch.FloatTensor(self.data[idx,2])
        coeff_f = torch.FloatTensor(self.data[idx,3])
        load_vec_f = torch.FloatTensor(self.load_vector[idx])
        return {'coeff_u1': coeff_u1, 'coeff_u2': coeff_u2, 'coeff_p': coeff_p, 'coeff_f': coeff_f, 'load_vec_f' : load_vec_f}

    def __len__(self):
        if self.kind=='train':
            return self.num_train
        else:
            return len(self.data)
        
lg_dataset = Dataset(mesh, kind='train', num_train=NUM_TRAIN_DATA)
trainloader = DataLoader(lg_dataset, batch_size=BATCH_SIZE_train, shuffle=True)
lg_dataset = Dataset(mesh, kind='validate')
validateloader = DataLoader(lg_dataset, batch_size=BATCH_SIZE_validate, shuffle=False)

print("Num train : {}, Num test: {}".format(len(trainloader.dataset), len(validateloader.dataset)))



if gparams['model']=='CNN2D':
    model = MODEL(ACT, RESOL_IN, D_in, FILTERS, D_out, kernel_size=KERNEL_SIZE, padding=PADDING, blocks=BLOCKS)
else:
    model = MODEL(ACT, D_in, FILTERS, D_out, kernel_size=KERNEL_SIZE, padding=PADDING, blocks=BLOCKS)

# SEND TO GPU (or CPU)
model.cuda()

# KAIMING INITIALIZATION
def weights_init(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight.data)
        torch.nn.init.zeros_(m.bias)

model.apply(weights_init)


def init_optim(model):
    params = {'history_size': 10,
              'tolerance_grad': 1E-15,
              'tolerance_change': 1E-15,
              'max_eval': 10,
                }
    return torch.optim.LBFGS(model.parameters(), **params)

optimizer = init_optim(model)




def weak_form(coeff_u, load_vec_f, matrix):
    # pts : N+1
    # num of basis : N+1
    # coeff_u : (num_f, 1, N+1)
    # 
    #return LHS, RHS : (num_f, N+1)

    ## LHS
    if DO_PRECOND==True:
       LHS = torch.stack([(matrix@precond).mm(coeff_u_one) for coeff_u_one in coeff_u.transpose(1,2)])
    else:
        LHS = torch.stack([matrix.mm(coeff_u_one) for coeff_u_one in coeff_u.transpose(1,2)])
    LHS=torch.sum(LHS,dim=-1)

    ## RHS
    RHS = load_vec_f.cuda()#.float()
    return LHS, RHS



def closure(model, coeff_f, load_vec_f, matrix, resol_in):
    def f(x,y,coeff):
        m0, m1, n0, n1, n2, n3=coeff[:,[0]], coeff[:,[1]], coeff[:,[2]], coeff[:,[3]], coeff[:,[4]], coeff[:,[5]]
        return torch.stack([m0*torch.sin(n0*x+n1*y), m1*torch.cos(n2*x+n3*y)],dim=1)
    grid_x=torch.linspace(0,1,resol_in)

    input_grid=torch.cartesian_prod(grid_x,grid_x)
    input_grid=input_grid.cuda()
    value_f=f(input_grid[:,0],input_grid[:,1], coeff_f).reshape(-1,D_in,resol_in,resol_in)
    pred_coeff_u = model(value_f)
    LHS, RHS = weak_form(pred_coeff_u, load_vec_f, matrix)
    
    ## Loss : summation on basis functions & mean on funcions f_i
    loss_wf=(LHS-RHS)**2
    loss=loss_wf.sum(dim=-1).mean(dim=-1)

    if DO_PRECOND==True:
        return  loss, (precond@pred_coeff_u.transpose(1,2)).transpose(1,2)
    else:
        return  loss, pred_coeff_u

def rel_L2_error(pred, true):
    return (torch.sum((true-pred)**2, dim=-1)/torch.sum((true)**2, dim=-1))**0.5

def log_gparams(gparams):
    cwd = os.getcwd()
    os.chdir(PATH)
    with open('parameters.txt', 'w') as f:
        for k, v in gparams.items():
            if k == 'losses':
                df = pd.DataFrame(gparams['losses'])
                df.to_csv('losses.csv')
            else:
                entry = f"{k}:{v}\n"
                f.write(entry)
    os.chdir(cwd)

log_gparams(gparams)
################################################
time0 = time.time()
losses=[]
train_rel_L2_errors_u1=[]
train_rel_L2_errors_u2=[]
train_rel_L2_errors_p=[]
test_rel_L2_errors_u1=[]
test_rel_L2_errors_u2=[]
test_rel_L2_errors_p=[]
for epoch in range(1, EPOCHS+1):
    model.train()
    loss_total = 0
    num_samples=0
    train_rel_L2_error_u1 = 0
    train_rel_L2_error_u2 = 0
    train_rel_L2_error_p = 0

    for batch_idx, sample_batch in enumerate(trainloader):
        optimizer.zero_grad()
        coeff_u1 = sample_batch['coeff_u1']
        coeff_u2 = sample_batch['coeff_u2']
        coeff_p = sample_batch['coeff_p']
        coeff_f = sample_batch['coeff_f'].cuda()
        load_vec_f = sample_batch['load_vec_f'].cuda()
        loss,u_pred = closure(model, coeff_f, load_vec_f, MATRIX, RESOL_IN)
        loss.backward()  

        optimizer.step(loss.item)
        loss_total += np.round(float(loss.item()), 4)
        num_samples += coeff_u1.shape[0]

        with torch.no_grad():
            model.eval()
            _,u_pred = closure(model, coeff_f, load_vec_f, MATRIX, RESOL_IN)
            u_pred=u_pred.squeeze().detach().cpu()
            u1_pred=u_pred[:,torch.tensor(IDX_SOL[0])]
            u2_pred=u_pred[:,torch.tensor(IDX_SOL[1])]
            p_pred=u_pred[:,torch.tensor(IDX_SOL[2])]

            train_rel_L2_error_u1 += torch.sum(rel_L2_error(u1_pred, coeff_u1))
            train_rel_L2_error_u2 += torch.sum(rel_L2_error(u2_pred, coeff_u2))
            train_rel_L2_error_p += torch.sum(rel_L2_error(p_pred, coeff_p))
    
    train_rel_L2_error_u1 /= num_samples
    train_rel_L2_error_u2 /= num_samples
    train_rel_L2_error_p /= num_samples
    

    if epoch%100==0:
        ## Test
        num_samples=0
        test_rel_L2_error_u1 = 0
        test_rel_L2_error_u2 = 0
        test_rel_L2_error_p = 0
        for batch_idx, sample_batch in enumerate(validateloader):
            with torch.no_grad():
                model.eval()
                coeff_u1 = sample_batch['coeff_u1']
                coeff_u2 = sample_batch['coeff_u2']
                coeff_p = sample_batch['coeff_p']
                coeff_f = sample_batch['coeff_f'].cuda()
                load_vec_f = sample_batch['load_vec_f'].cuda()
                _,u_pred = closure(model, coeff_f, load_vec_f, MATRIX, RESOL_IN)
                u_pred=u_pred.squeeze().detach().cpu()
                u1_pred=u_pred[:,torch.tensor(IDX_SOL[0])]
                u2_pred=u_pred[:,torch.tensor(IDX_SOL[1])]
                p_pred=u_pred[:,torch.tensor(IDX_SOL[2])]

                test_rel_L2_error_u1 += torch.sum(rel_L2_error(u1_pred, coeff_u1))
                test_rel_L2_error_u2 += torch.sum(rel_L2_error(u2_pred, coeff_u2))
                test_rel_L2_error_p += torch.sum(rel_L2_error(p_pred, coeff_p))

                num_samples += coeff_u1.shape[0]

        test_rel_L2_error_u1 /= num_samples
        test_rel_L2_error_u2 /= num_samples
        test_rel_L2_error_p /= num_samples
        
        ##Save and print
        losses.append(loss_total)
        train_rel_L2_errors_u1.append(train_rel_L2_error_u1)
        train_rel_L2_errors_u2.append(train_rel_L2_error_u2)
        train_rel_L2_errors_p.append(train_rel_L2_error_p)

        test_rel_L2_errors_u1.append(test_rel_L2_error_u1)
        test_rel_L2_errors_u2.append(test_rel_L2_error_u2)
        test_rel_L2_errors_p.append(test_rel_L2_error_p)
        torch.save({'model_state_dict': model.state_dict(),
                    'losses': losses,
                    'train_rel_L2_errors_u1': train_rel_L2_errors_u1,
                    'train_rel_L2_errors_u2': train_rel_L2_errors_u2,
                    'train_rel_L2_errors_p': train_rel_L2_errors_p,
                    'test_rel_L2_errors_u1': test_rel_L2_errors_u1,
                    'test_rel_L2_errors_u2': test_rel_L2_errors_u2,
                    'test_rel_L2_errors_p': test_rel_L2_errors_p,
        }, PATH + '/model.pt')
        # print("Epoch {0:4d}: weak_form_loss {1:4.1f}, train_rel_error {2:.5f}, test_rel_error {3:.5f}".format(epoch, loss_total, train_rel_L2_error, test_rel_L2_error))
        print(f"Epoch {epoch:4d}: wloss {loss_total:2.5f}, train_rel {train_rel_L2_error_u1:.5f}/{train_rel_L2_error_u2:.5f}/{train_rel_L2_error_p:.5f}, test_rel {test_rel_L2_error_u1:.5f}/{test_rel_L2_error_u2:.5f}/{test_rel_L2_error_p:.5f}")
        
torch.save({'model_state_dict': model.state_dict(),
            'losses': losses,
            'train_rel_L2_errors_u1': train_rel_L2_errors_u1,
            'train_rel_L2_errors_u2': train_rel_L2_errors_u2,
            'train_rel_L2_errors_p': train_rel_L2_errors_p,
            'test_rel_L2_errors_u1': test_rel_L2_errors_u1,
            'test_rel_L2_errors_u2': test_rel_L2_errors_u2,
            'test_rel_L2_errors_p': test_rel_L2_errors_p,
}, PATH + '/model.pt')
        
train_t=time.time()-time0
NPARAMS = sum(p.numel() for p in model.parameters() if p.requires_grad)

gparams['train_time'] = train_t
gparams['nParams'] = NPARAMS
gparams['batch_size_train'] = BATCH_SIZE_train
gparams['batch_size_validate'] = BATCH_SIZE_validate
gparams['path'] = PATH

log_gparams(gparams)
