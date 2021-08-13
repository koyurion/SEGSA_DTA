import os
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import time
import numpy as np
import gc
import sys
import random
sys.setrecursionlimit(50000)
import pickle

from step8_1_Abstract_train import Gflag,SEED,max_pocket_len   # ,select_fold,Task_flag,shell_num

random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)  # when using pandas/numpy

from layers import GraphConvolution

if Gflag:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.backends.cudnn.benchmark = True
else:
    torch.set_default_tensor_type('torch.FloatTensor')
torch.nn.Module.dump_patches = True



# Custom loss-1: pairwise
class Masked_BCELoss(nn.Module):
    def __init__(self):
        super(Masked_BCELoss, self).__init__()
        self.criterion = nn.BCELoss(reduce=False)

    def forward(self, pred, label, pairwise_mask, vertex_mask, seq_mask):
        batch_size = pred.size(0)
        loss_all = self.criterion(pred, label)
        loss_mask = torch.matmul(vertex_mask.view(batch_size, -1, 1),
                                 seq_mask.view(batch_size, 1, -1)) * pairwise_mask.view(-1, 1, 1)
        loss = torch.sum(loss_all * loss_mask) / torch.sum(pairwise_mask).clamp(min=1e-10)
        return loss


# Custom loss-1: inter
class Masked_MSELoss(nn.Module):
    def __init__(self):
        super(Masked_MSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduce=False)

    def forward(self, pred, label, inter_mask):
        batch_size = pred.size(0)
        loss_all = self.criterion(pred, label)
        loss = 1.0 * torch.sum(loss_all * inter_mask) / batch_size
        return loss




class Fingerprint(nn.Module):

    def __init__(self, nprofeat,nproBondfeat, nligfeat, nligBondfeat,npair, nhide,noutput,p_dropout_fea,p_dropout,topu_pro,topu_lig,FP_frame_flag = False):
        super(Fingerprint, self).__init__()

        # GCN
        self.topu_pro = topu_pro
        self.topu_lig = topu_lig
        self.pro_gc_aa = GraphConvolution(nprofeat, int(3 * npair/4))
        self.pro_gc_bond = GraphConvolution(nproBondfeat, int(npair/4))
        self.pro_gcs = nn.ModuleList(
            [GraphConvolution(npair, npair) for r in range(topu_pro - 1)])
        # self.pro_gc2 = GraphConvolution(npair, npair)

        self.lig_gc_atom = GraphConvolution(nligfeat,int(3 * npair / 4))
        self.lig_gc_bond = GraphConvolution(nligBondfeat,int(npair / 4))
        self.lig_gcs = nn.ModuleList(
            [GraphConvolution(npair, npair) for r in range(topu_lig - 1)])

        self.dropout_fea = nn.Dropout(p=p_dropout_fea)

        # Pairwise Interaction Prediction Module
        self.pairwise_protein = nn.Linear(npair, npair)
        self.pairwise_ligand = nn.Linear(npair, npair)


        # attention
        self.mol_res_attend_mol = nn.Linear(npair, npair)
        self.mol_res_attend_res = nn.Linear(npair, npair)


        self.FP_frame_flag = FP_frame_flag
        if self.FP_frame_flag:
            ndnn_input = npair * 2
        else:
            ndnn_input = npair * 3

        self.dropout = nn.Dropout(p=p_dropout)
        self.dnn1 = nn.Linear(ndnn_input, nhide)
        self.dnn2 = nn.Linear(nhide, int(nhide / 2))
        self.dnn3 = nn.Linear(int(nhide / 2), noutput)


    def mask_softmax(self, a, mask, dim=-1):
        a_max = torch.max(a, dim, keepdim=True)[
            0]  # the deal target of softmax is each column feature ,not row,so no need to warried about padding
        a_exp = torch.exp(a - a_max)
        a_exp = a_exp * mask
        a_softmax = a_exp / (torch.sum(a_exp, dim, keepdim=True))  # + 1e-6
        return a_softmax

    def ligand_fea_module(self, x_atom, x_bond, adj):

        x_atom = self.dropout_fea(F.relu(self.lig_gc_atom(x_atom, adj)))
        x_bond = self.dropout_fea(F.relu(self.lig_gc_bond(x_bond, adj.unsqueeze(2)))).squeeze(-2)
        # print("lig shape",x_atom.shape,x_bond.shape)
        x = torch.cat([x_atom,x_bond],dim=-1)
        for ngc in range(self.topu_lig - 1):
            x = self.dropout_fea(F.relu(self.lig_gcs[ngc](x,adj)))

        return x  # [batch,mol_length,512]

    def protein_fea_module(self,x_aa,x_bond,adj,bond_adj,chainRes_idx,batch_size):
        '''
            :return: FP_batch: [batch_size,pro_ff_num,200]
        '''
        x_aa = self.dropout_fea(F.relu(self.pro_gc_aa(x_aa, adj)))
        x_bond= self.dropout_fea(F.relu(self.pro_gc_bond(x_bond,bond_adj))).squeeze(-2)
        x = torch.cat([x_aa, x_bond], dim=-1)
        for ngc in range(self.topu_pro - 1):
            x = self.dropout_fea(F.relu(self.pro_gcs[ngc](x,adj)))
            
        # x = self.dropout_fea(F.relu(self.pro_gc2(x, adj)))

        pocket_res_x = [x[i][chainRes_idx[i]] for i in range(batch_size)]
        pocket_res_x = torch.stack(pocket_res_x, dim=0)

        return pocket_res_x

    def Pairwise_pred_module(self, atom_feature, residue_feature, atom_mask, pocket_mask, batch_size):
        # todo: vertex_mask, seq_mask,
        #  mol_feature(size)
        # print(FP_frame.shape)
        # print(atom_feature.shape)
        pairwise_p_feature = F.leaky_relu(self.pairwise_protein(residue_feature), 0.1)
        pairwise_l_feature = F.leaky_relu(self.pairwise_ligand(atom_feature), 0.1)
        
        pairwise_pred = torch.sigmoid(torch.matmul(pairwise_l_feature, pairwise_p_feature.transpose(1, 2)))
        pairwise_mask = torch.matmul(atom_mask.view(batch_size, -1, 1), pocket_mask.view(batch_size, 1, -1))
        # print(pairwise_pred.shape)
        # print(pairwise_mask.shape)
        pairwise_pred = pairwise_pred * pairwise_mask
        # if pad ,then all 0

        return pairwise_pred

    def attention_module(self, batch_size, atom_feature, residue_feature, pocket_mask, pairwise_pred, FP_frame_flag, atom_mask):
        # 1.atom_feature & FP_frame

        atom_att_weight = torch.sum(pairwise_pred,dim=1)
        softmax_atom_att_weight = self.mask_softmax(atom_att_weight, pocket_mask).view(batch_size, -1, 1)
        update_atom_FP_frame = torch.mul(residue_feature, softmax_atom_att_weight)
        atom_FP_batch_vector = torch.sum(update_atom_FP_frame, dim=1)

        # 1.2 atom_feature & pairwise_pred
        ligand_atom_att_weight = torch.sum(pairwise_pred, dim=-1)
        ligand_softmax_atom_att_weight = self.mask_softmax(ligand_atom_att_weight, atom_mask).view(batch_size, -1, 1)
        update_atom_feature = torch.mul(atom_feature, ligand_softmax_atom_att_weight)
        mol_feature = torch.sum(update_atom_feature, dim=1)

        # 2. mol_feature & FP_frame
        if FP_frame_flag:
            mol_FP_frame = update_atom_FP_frame
        else:
            mol_FP_frame = residue_feature

        h_mol = torch.relu(self.mol_res_attend_mol(mol_feature))
        h_res = torch.relu(self.mol_res_attend_res(mol_FP_frame))

        batch_size, max_aa_num, fea_size = h_res.shape

        # softmax
        ori_mol_att_weights = torch.matmul(h_res.unsqueeze(-2),
                                       h_mol.unsqueeze(1).unsqueeze(-1).expand(batch_size, max_aa_num, fea_size,
                                                                               1)).squeeze()
        mol_att_weights = self.mask_softmax(ori_mol_att_weights, pocket_mask).view(batch_size, -1, 1)
        update_mol_FP_frame = torch.mul(h_res, mol_att_weights)
        mol_FP_batch_vector = torch.sum(update_mol_FP_frame, dim=1)
        interout_pred = mol_att_weights.squeeze(1).view(batch_size,-1)

        # concat : [batch,512]
        if FP_frame_flag:
            three_feature = torch.cat((mol_feature, mol_FP_batch_vector), 1)
        else:
            three_feature = torch.cat((mol_feature, mol_FP_batch_vector, atom_FP_batch_vector), 1)

        attend_ohide = self.dropout(F.relu(self.dnn1(three_feature)))
        attend_ohide = self.dropout(F.relu(self.dnn2(attend_ohide)))
        attend_ypred = self.dnn3(attend_ohide)


        return attend_ypred, interout_pred, softmax_atom_att_weight

    def forward(self, atom_feature, bond_feature, atom_adj, atom_mask,residue_aa_feature, residue_bond_feature,residue_adj,bond_adj, pocket_mask,chainRes_idx):
        batch_size = atom_feature.shape[0]
        atom_feature = self.ligand_fea_module(atom_feature,bond_feature, atom_adj)
        residue_feature = self.protein_fea_module(residue_aa_feature, residue_bond_feature,residue_adj,bond_adj,chainRes_idx,batch_size)
        pairwise_pred = self.Pairwise_pred_module(atom_feature, residue_feature, atom_mask, pocket_mask, batch_size)

        attention_pred, interout_pred, softmax_atom_att_weight = \
            self.attention_module(batch_size,atom_feature, residue_feature, pocket_mask, pairwise_pred, self.FP_frame_flag, atom_mask)

        return pairwise_pred, attention_pred, interout_pred

