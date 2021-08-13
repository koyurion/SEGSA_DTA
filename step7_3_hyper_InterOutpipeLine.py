# 1. system packages
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from datetime import datetime
import numpy as np
import gc
import random
import sys

sys.path.append('../')

import warnings

warnings.filterwarnings("ignore")

import os
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math

import scipy.stats as stats
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, recall_score, precision_score

# 2. for reproduce
from step8_1_Abstract_train import GPU_ID, SEED

os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)  # when using pandas/numpy

# 3. my own packages

from step8_1_Abstract_train import Fingerprint, Masked_BCELoss, Masked_MSELoss
from step8_1_Abstract_train import Gflag, task_flag, cur_t, hyper_flag
from step8_1_Abstract_train import mnist_1107_fixedLoss_ThreeTask, MultitaskMnistLoss
from utils import load_data, get_GCN_protein_data, get_GCN_ligand_data, get_GCN_InterOut_data
from utils import return_GCN_ligand_featuredim, return_GCN_protein_featuredim

# 3 for GPU
if Gflag:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.backends.cudnn.benchmark = True
else:
    torch.set_default_tensor_type('torch.FloatTensor')
# to load/dump state_dict
torch.nn.Module.dump_patches = True


# ===================================================
# to pad pairwise babel with 0 when train
def pad_pairbabel(pairbabel, row_num, col_num):
    padded_babel = np.zeros((len(pairbabel), row_num, col_num))
    for i, arr in enumerate(pairbabel):
        padded_babel[i, :arr.shape[0], :arr.shape[1]] = arr
    return padded_babel


# train model
def train(model, dataset, train_index, test_index, optimizer, loss_func):
    batch_list = []
    total_all_loss = 0
    affinity_loss = 0
    pairwise_loss = 0
    inter_loss = 0
    pearson_loss = 0
    model.train()
    # mini batch
    for i in range(0, len(train_index), batch_size):
        batch = train_index[i:i + batch_size]  # index batch
        batch_list.append(batch)

    for counter, batch in enumerate(batch_list):
        batch_array = dataset[batch, :]
        pdbids_list = batch_array[:, 0]
        ligands_list = batch_array[:, 1]

        # ====== get ligand fea
        atom_feature, bond_feature, atom_adj, atom_mask = get_GCN_ligand_data(ligands_list, lig_feature_dict)
        atom_feature = torch.FloatTensor(atom_feature)
        bond_feature = torch.FloatTensor(bond_feature)
        atom_adj = torch.FloatTensor(atom_adj)
        atom_mask = torch.FloatTensor(atom_mask)

        # ====== get protein fea
        residue_aa_feature, residue_bond_feature, residue_adj,bond_adj = get_GCN_protein_data(pdbids_list, pro_feature_dict)
        residue_aa_feature = torch.FloatTensor(residue_aa_feature)
        residue_bond_feature = torch.FloatTensor(residue_bond_feature)
        residue_adj = torch.FloatTensor(residue_adj)
        bond_adj = torch.FloatTensor(bond_adj)

        # ======= get label and pairbabel
        aff_val = torch.FloatTensor(batch_array[:, 3].astype(np.float))

        pairwise_label = torch.FloatTensor(batch_array[:, -2].tolist())
        pairwise_mask = torch.FloatTensor(batch_array[:, -1].astype(float).reshape(-1, 1))

        # ==== get inter_val
        inter_vals, pocket_mask, chainRes_idx = get_GCN_InterOut_data(pdbids_list, interOut_dict)
        inter_vals = torch.FloatTensor(inter_vals)
        pocket_mask = torch.FloatTensor(pocket_mask)
        chainRes_idx = torch.LongTensor(chainRes_idx)

        if Gflag:
            atom_feature = atom_feature.cuda()
            bond_feature = bond_feature.cuda()
            atom_adj = atom_adj.cuda()
            atom_mask = atom_mask.cuda()

            residue_aa_feature = residue_aa_feature.cuda()
            residue_bond_feature = residue_bond_feature.cuda()
            residue_adj = residue_adj.cuda()
            bond_adj = bond_adj.cuda()

            aff_val = aff_val.cuda()
            pairwise_label = pairwise_label.cuda()
            pairwise_mask = pairwise_mask.cuda()
            inter_vals = inter_vals.cuda()
            pocket_mask = pocket_mask.cuda()
            chainRes_idx = chainRes_idx.cuda()

        optimizer.zero_grad()

        pairwise_pred, aff_pred, inter_pred = model(atom_feature, bond_feature, atom_adj, atom_mask, \
                                                    residue_aa_feature, residue_bond_feature, residue_adj,bond_adj, pocket_mask, chainRes_idx)




        total_loss, (loss_inter, loss_aff, loss_pairwise) = loss_func(
            [inter_pred, aff_pred, pairwise_pred],
            [inter_vals, aff_val.view(-1, 1), pairwise_label], pairwise_mask,
            torch.Tensor(atom_mask), torch.Tensor(pocket_mask), pocket_mask)

        
        if torch.isnan(total_loss):
            print("total_loss is nan !!!!")
        '''
        total_all_loss += float(total_loss.data * batch_size)
        affinity_loss += float(loss_aff.data * batch_size)
        pairwise_loss += float(loss_pairwise.data * batch_size)
        inter_loss += float(loss_inter.data * batch_size)
        '''
        total_loss.backward()
        # todo: clip 40
        nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

    # print("all counter time elapsed: {:.4f}s".format(time.time() - t_counter))
    # train_time_list.append(time.time() - t_counter)
    # all_total_loss.append(total_loss)

    # === to print loss ===
    '''
    pearson_loss = 0
    loss_list = [total_all_loss, affinity_loss, pairwise_loss,inter_loss]

    loss_name = ['total loss', 'affinity loss', "pairwise loss", 'inter loss']
    print_loss = [loss_name[i] + ' ' + str(round(loss_list[i] / float(len(train_index)), 6)) for i in
                  range(len(loss_name))]
    print(' '.join(print_loss))
    '''
    # === test ===
    # t_train_per = time.time()
    # train_performence = eval(model, dataset, train_index,loss_func)
    # print('train_perf', ' '.join([str(round(i, 6)) for i in train_performence]))
    # print("each train_performence time elapsed: {:.4f}s".format(time.time() - t_train_per))
    # test_tainTime_list.append(time.time() - t_train_per)

    # t_test_per = time.time()
    test_performence = eval(model, dataset, test_index, loss_func)
    # print('test_perf', ' '.join([str(round(i, 6)) for i in test_performence]))
    # print("each test_performence time elapsed: {:.4f}s".format(time.time() - t_test_per))
    # return train_performence, train_performence
    # test_testTime_list.append(time.time() - t_test_per)
    train_performence = [0, 0, 0, 0, 0, 0]
    return train_performence, test_performence



def eval(model, dataset, index_list, loss_func):
    model.eval()
    with torch.no_grad():
        if len(index_list) == 0:
            return [0, 0, 0, 0, 0]

        batch_list = []
        # ==== mini batch
        for i in range(0, len(index_list), batch_size):
            batch = index_list[i:i + batch_size]  # index batch
            batch_list.append(batch)

        pairwise_auc_list = []
        output_list = []
        label_list = []
        pairwise_pred_list = []
        inter_RMSE_list = []
        inter_pred_list = []
        for counter, batch in enumerate(batch_list):
            batch_array = dataset[batch, :]
            pdbids_list = batch_array[:, 0]
            ligands_list = batch_array[:, 1]
            # ====== get ligand fea
            atom_feature, bond_feature, atom_adj, atom_mask = get_GCN_ligand_data(ligands_list, lig_feature_dict)
            atom_feature = torch.FloatTensor(atom_feature)
            bond_feature = torch.FloatTensor(bond_feature)
            atom_adj = torch.FloatTensor(atom_adj)
            atom_mask = torch.FloatTensor(atom_mask)

            # ====== get protein fea
            residue_aa_feature, residue_bond_feature, residue_adj, bond_adj = get_GCN_protein_data(pdbids_list,
                                                                                                   pro_feature_dict)
            residue_aa_feature = torch.FloatTensor(residue_aa_feature)
            residue_bond_feature = torch.FloatTensor(residue_bond_feature)
            residue_adj = torch.FloatTensor(residue_adj)
            bond_adj = torch.FloatTensor(bond_adj)

            # ======= get label
            aff_val = torch.FloatTensor(batch_array[:, 3].astype(np.float))

            # to get aff and pairbabel
            pairwise_label = np.array(batch_array[:, -2].tolist())  # .astype(np.float)
            pairwise_mask = batch_array[:, -1]

            # ==== get inter_val
            inter_vals, pocket_mask, chainRes_idx = get_GCN_InterOut_data(pdbids_list, interOut_dict)
            inter_vals = torch.FloatTensor(inter_vals)
            pocket_mask = torch.FloatTensor(pocket_mask)
            chainRes_idx = torch.LongTensor(chainRes_idx)

            if Gflag:
                atom_feature = atom_feature.cuda()
                bond_feature = bond_feature.cuda()
                atom_adj = atom_adj.cuda()
                atom_mask = atom_mask.cuda()

                residue_aa_feature = residue_aa_feature.cuda()
                residue_bond_feature = residue_bond_feature.cuda()
                residue_adj = residue_adj.cuda()
                bond_adj = bond_adj.cuda()

                # aff_val = aff_val.cuda()
                # pairwise_label = pairwise_label.cuda()
                # pairwise_mask = pairwise_mask.cuda()
                # inter_vals = inter_vals.cuda()
                pocket_mask = pocket_mask.cuda()
                chainRes_idx = chainRes_idx.cuda()
            # t_test_per = time.time()
            pairwise_pred, aff_pred, inter_pred = model(atom_feature, bond_feature, atom_adj, atom_mask, \
                                                        residue_aa_feature, residue_bond_feature, residue_adj, bond_adj,
                                                        pocket_mask, chainRes_idx)

            # metric
            # === classification: AUC
            # t_AUC_per = time.time()
            for j in range(len(pairwise_mask)):
                if pairwise_mask[j]:
                    num_vertex = int(torch.sum(atom_mask[j, :]))
                    num_residue = int(torch.sum(pocket_mask[j, :]))
                    pairwise_pred_i = pairwise_pred[j, :num_vertex, :num_residue].cpu().detach().numpy().reshape(-1)
                    pairwise_label_i = pairwise_label[j, :num_vertex, :num_residue].reshape(-1)
                    pairwise_auc_list.append(roc_auc_score(pairwise_label_i, pairwise_pred_i))


            # === inter_val ===
            # t_InterOut_per = time.time()
            for j in range(len(pocket_mask)):
                num_aa = int(torch.sum(pocket_mask[j, :]))
                inter_pred_i = inter_pred[j, :num_aa].cpu().detach().numpy().reshape(-1)
                inter_label_i = inter_vals[j, :num_aa].reshape(-1)
                inter_RMSE_list.append(torch.sqrt(
                    F.mse_loss(torch.FloatTensor(inter_pred_i), torch.FloatTensor(inter_label_i), reduction='sum')))


            output_list += aff_pred.cpu().detach().numpy().reshape(-1).tolist()
            label_list += aff_val.reshape(-1).tolist()

        output_list = torch.FloatTensor(output_list)
        label_list = torch.FloatTensor(label_list)
        # == finall mettric
        rmse_value = math.sqrt(F.mse_loss(output_list, label_list))  # rmse
        pearson_s, pearson_p = stats.pearsonr(output_list, label_list)  # Pearsonr coefficient

        average_pairwise_auc = np.mean(pairwise_auc_list)  # mean AUC
        average_inter_RMSE = np.mean(inter_RMSE_list)

    return [rmse_value, pearson_s, average_pairwise_auc, average_inter_RMSE]


# ====================================================

enabled_tasks = (True, True, True)


def _get_fixed_loss_func(enabled_tasks: [bool], weights: [float], mnist_type: str):
    return mnist_1107_fixedLoss_ThreeTask.get_fixed_loss(enabled_tasks, weights, mnist_type)


def _get_learned_loss_func(enabled_tasks: [bool], model: Fingerprint, mnist_type: str):
    return mnist_1107_fixedLoss_ThreeTask.get_learned_loss(enabled_tasks, model.get_loss_weights(), mnist_type)


def _get_loss_func(model: Fingerprint, loss_type="learned", weights=0) -> MultitaskMnistLoss:
    if loss_type == 'fixed':
        return _get_fixed_loss_func(enabled_tasks, weights, mnist_type="???")
    elif loss_type == 'learned':
        return _get_learned_loss_func(enabled_tasks, model=model, mnist_type="???")
    else:
        raise ValueError(f'Unknown loss type: {loss_type}')


# =====================================================


def print_figure(train_log, test_log, fold_count, epoch):
    train_log = np.asarray(train_log)
    test_log = np.asarray(test_log)
    fig_title = task_flag + "_" + str(fold_count) + "_" + str(epoch) + "_RMSE"
    plt.plot(train_log, 'r')
    plt.plot(test_log, 'b')
    plt.legend(["train", "test"])
    plt.title(fig_title)
    plt.savefig(figure_path + fig_title + ".png")
    plt.close('all')


def f(weight_decay, learning_rate, npair, nhide, p_dropout_fea, p_dropout, topu_pro, topu_lig, \
      matrix_weight, inter_weight, direction=False):
    assert direction == False and type(direction) == bool
    hyper_params_list = [weight_decay, learning_rate, npair, nhide, p_dropout_fea, p_dropout, topu_pro, topu_lig, \
                         matrix_weight, inter_weight]
    # print("hyper_params_list", hyper_params_list)

    rep_all_RMSE_list = []
    rep_epoch_list = []
    rep_all_Pearons_list = []
    rep_all_auc_list = []
    rep_all_interRMSE_list = []

    best_param = {}
    for rep in range(n_rep):
        kf = KFold(n_splits=my_n_splits, shuffle=True, random_state=my_random_state)
        df_indexhome = kf.split(model_dataset)
        fold_count = 0
        for train_index, test_index in df_indexhome:
            fold_count += 1
            # print("repeat:", rep, "fold:", fold_count)

            # for reproduce
            random.seed(SEED)
            os.environ['PYTHONHASHSEED'] = str(SEED)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            np.random.seed(SEED)  # when using pandas/numpy

            # ====== each fold init a new model
            model = Fingerprint(int(round(nprofeat)),int(nproBondfeat), int(round(nligfeat)),int(nligBondfeat),
                                int(round(npair)), int(round(nhide)),
                                int(round(noutput)), p_dropout_fea, p_dropout, topu_pro, topu_lig, FP_frame_flag=False)
            if Gflag:
                model.cuda()

            # init_state_dict = torch.load(init_state_dict_path)
            # model.load_state_dict(init_state_dict)

            # ===
            weights_tuple = (aff_weight, matrix_weight, inter_weight)
            loss_func = _get_loss_func(model=model, loss_type="fixed", weights=weights_tuple)
            optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay, amsgrad=True)


            best_param["test_epoch"] = 0
            best_param["test_AllLoss"] = 9e8
            best_param["test_Pearson"] = -9e8
            best_param["test_RMSE"] = 9e8
            best_param["best_AUC"] = 0
            best_param["best_InterRMSE"] = 9e8

            np.random.shuffle(train_index)

            stop_flag = 0
            for epoch in range(epochs):

                # print("========== fold", fold_count, "Epoch:", epoch, "==========")
                # === tain & eval
                train_performence, test_performence = train(model, model_dataset, train_index, test_index,
                                                            optimizer, loss_func)

                test_RMSE = test_performence[0]
                test_Pearson = test_performence[1]

                if test_RMSE < best_param["test_RMSE"]:
                    best_param["test_RMSE"] = test_RMSE
                    best_param["test_Pearson"] = test_Pearson
                    best_param["test_epoch"] = epoch
                    best_param["best_AUC"] = test_performence[-2]
                    best_param["best_InterRMSE"] = test_performence[-1]

                if epoch - best_param["test_epoch"] >= stop_patience:
                    '''
                    print("STOP:", epoch, best_param["test_epoch"],
                          best_param["test_RMSE"], best_param["test_Pearson"], best_param["best_AUC"],
                          best_param["best_InterRMSE"])
                    '''
                    stop_flag = True
                    break

            if not stop_flag:
                pass
                '''
                print("UnSTOP:", epoch, best_param["test_epoch"], best_param["test_RMSE"],
                      best_param["test_Pearson"], best_param["best_AUC"], best_param["best_InterRMSE"])
                '''

            # print_figure(train_log, test_log, fold_count, epoch)

            rep_all_RMSE_list.append(best_param["test_RMSE"])  # append each rep each fold's best testRMSE
            rep_all_Pearons_list.append(best_param["test_Pearson"])
            rep_epoch_list.append(best_param["test_epoch"])
            rep_all_auc_list.append(best_param["best_AUC"])
            rep_all_interRMSE_list.append(best_param["best_InterRMSE"])

    mean_metric_list = [np.mean(rep_epoch_list), np.mean(rep_all_RMSE_list), np.mean(rep_all_Pearons_list), \
                        np.mean(rep_all_auc_list), np.mean(rep_all_interRMSE_list)]
    flag_metric = np.mean(rep_all_RMSE_list)

    if flag_metric < 1.38:
        hyper_params_path = task_path + "hyper_params_list/"
        if not os.path.exists(hyper_params_path):
            os.makedirs(hyper_params_path)
        torch.save(hyper_params_list, hyper_params_path + task_flag + "_" + "_hyperParmasList_" + '_' + str(
            flag_metric) + '.pt')

    log_f.write(','.join([str(i) for i in hyper_params_list]))
    log_f.write(',' + '#' + ',' + ','.join([str(round(i, 6)) for i in mean_metric_list]) + '\n')



if __name__ == "__main__":
    t_total = time.time()
    now = datetime.now().strftime("%Y%m%d_%H%M")

    # train setting
    batch_size = 64
    epochs = 300
    n_rep = 1
    my_n_splits = 5
    my_random_state = 42
    noutput = 1
    aff_weight = 1.0
    stop_patience = 30


    
    param_grid = {

        "weight_decay": [1e-4],
        "learning_rate": [0.003], # [1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3, 1.0]
        
        "npair": [256], # [128, 256, 512]
        "nhide": [512], # [64, 128, 256, 512, 1024]
        
        "topu_pro": [2],
        "topu_lig": [2],
        
        "p_dropout_fea": [0.1],
        "p_dropout": [0.3],

        "matrix_weight": [0.05],
        "inter_weight": [10.0],
    }
    
    interScreenFlag = False
    interNormalnizeFlag = "Abs"
    cutoff = 630
    neighbour_threshold = 6

    topu_threshold = param_grid["topu_pro"][0]
    weightEdge_flag = True  # False # True
    halfNormalizeFlag = False  # False # True

    mark_flag = "_neighbour" + str(neighbour_threshold) + "_topu" + str(topu_threshold) + \
                "_weightEdge" + str(weightEdge_flag) + "_halfNormalize" + str(
        halfNormalizeFlag) + "_interNormalnize" + str(interNormalnizeFlag)

    task_flag = task_flag + mark_flag + "_sub"+str(param_grid["weight_decay"][0])

    # Load data
    model_dataset, pro_feature_dict, lig_feature_dict, interOut_dict = load_data(interScreenFlag,
                                                                                 interNormalnizeFlag,
                                                                                 cutoff, neighbour_threshold,
                                                                                 topu_threshold, weightEdge_flag,
                                                                                 halfNormalizeFlag)

    # init model
    nligfeat, nligBondfeat = return_GCN_ligand_featuredim(model_dataset[0, 1], lig_feature_dict)
    nprofeat, nproBondfeat = return_GCN_protein_featuredim(model_dataset[0, 0], pro_feature_dict)

    # =================================================================
    # log setting
    task_path = "./"+ hyper_flag +"/" + task_flag + "/"
    log_path = task_path + "Log_homes/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    log_file = log_path + "hyper_" + task_flag + "_" + now + ".log"
    log_f = open(log_file, 'a')
    log_f.write(','.join(
        ['weight_decay', 'learning_rate', 'npair', 'nhide', 'p_dropout_fea', 'p_dropout', 'topu_pro', 'topu_lig', \
         'matrix_weight', 'inter_weight', 'best_epoch', 'best_RMSE', 'best_PearsonrS', 'best_AUC',
         'best_InterRMSE']) + '\n')

    # ============================================================================================

    for i in range(len(param_grid["learning_rate"])):
        random.seed(i)
        hyper_params_list = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
        hyper_params_list["learning_rate"] = param_grid["learning_rate"][i]
        # hyper_params_list = torch.load(hyper_params_list_path)
        f(**hyper_params_list)
    print("train Done")
    print("total time:", time.time() - t_total)