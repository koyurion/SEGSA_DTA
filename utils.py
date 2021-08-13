from step8_1_Abstract_train import SEED
import random
import os
import torch
import numpy as np
import pickle
import scipy.sparse as sp
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)  # when using pandas/numpy

from step8_1_Abstract_train import data_path_dict
from data_process.step6_pad.step6_4_InterOut_pad_mask_pipeline import pad_mask
from data_process.step6_pad.step6_3_InterOut_pad_ff_pipeline import pad_ff
from data_process.step5_ligandFeature.step5_1_return_ligand_feature_KDKI_pipeline import get_ligand_feature
from data_process.step4_proteinFeature.step4_3_build_contactsMap_pipeline import get_profea_pickle
from data_process.step2_pipeline.InterOut_step5_extractAndNormalize_pipeline import get_InterOut_pickle

def read_pickle(path):
    return pickle.load(open(path,"rb"))

def load_data(interScreenFlag,interNormalnizeFlag,cutoff,neighbour_threshold,topu_threshold,weightEdge_flag,halfNormalizeFlag):

    prex_path = "./data_process/"

    # 3.1 model_dataset
    ori_dataset = np.array(read_pickle(data_path_dict["model_dataset"]))

    # 5 ligand
    print("ligand...")
    prex_outputligand_path = prex_path + "step5_ligandFeature/"
    feature_dicts, lig_feature_dict = get_ligand_feature(prex_outputligand_path,ori_dataset,halfNormalizeFlag)
    ligand_ligs_list = lig_feature_dict.keys()

    
    # 6 pad_ff
    middle_path = "step6_pad/" 
    last_path =  "_cutoff"+ str(cutoff)+ "_neighbour" + str(neighbour_threshold) + "_topu" + str(topu_threshold)
    output_InterOut_Padded_path = prex_path + middle_path + "InterOutPadded" + last_path +  "_interNormalnize"+ str(interNormalnizeFlag) + ".pickle"
    output_dataset_Padded_path = prex_path + middle_path + "dataset_pairwisePadded" + last_path + ".pickle"
    output_protein_Padded_path = prex_path + middle_path + "proteinPadded" + last_path + "_weightEdge" + str(weightEdge_flag)+ "_halfNormalize"+ str(halfNormalizeFlag) +".pickle"
    
    if os.path.exists(output_protein_Padded_path) and os.path.exists(output_InterOut_Padded_path) and os.path.exists(output_protein_Padded_path):
        model_dataset = pickle.load(open(output_dataset_Padded_path, "rb"))
        pro_feature_dict = pickle.load(open(output_protein_Padded_path, "rb"))
        InterOut_dict = pickle.load(open(output_InterOut_Padded_path, "rb"))
        
    else:
        # 2 interOut
        print("interOut...")
        out_InterOut_filepath = prex_path + "step2_pipeline/dfs_dict_" + "screen" + str(interScreenFlag)+ "_normalnize" + str(interNormalnizeFlag) +".pickle"
        dfs_InterOut_dict = get_InterOut_pickle(out_InterOut_filepath,interScreenFlag,interNormalnizeFlag)
        
        # 3.2 nonCovalent_dict
        nonCovalent_dict = read_pickle(data_path_dict["nonCovalent_dict"])
    
        # 4 protein
        print("protein...")
        protein_fea_dict = get_profea_pickle(prex_path,ori_dataset,dfs_InterOut_dict, cutoff, neighbour_threshold, topu_threshold,weightEdge_flag,halfNormalizeFlag)
        
        print("pad_ff...")
        model_dataset, sorted_InterOut = pad_ff(ori_dataset,dfs_InterOut_dict,
                         ligand_ligs_list,nonCovalent_dict,protein_fea_dict)
    
        # 6 pad_mask
        print("pad_mask...")
        model_dataset, pro_feature_dict, InterOut_dict = pad_mask(
            model_dataset, sorted_InterOut, protein_fea_dict,
            output_InterOut_Padded_path,
            output_dataset_Padded_path,
            output_protein_Padded_path)
    print("padded protein,model_dataset,InterOut_dict done") 
    per = np.random.permutation(model_dataset.shape[0])
    model_dataset = model_dataset[per, :]
    
    del ori_dataset, feature_dicts
    
    return model_dataset, pro_feature_dict,lig_feature_dict,InterOut_dict

def get_proBondFea_array(bondFeaturesMat_Pad):
    bondFeaturesMat_Pad_array = np.array([i_bondMat.toarray() for i_bondMat in bondFeaturesMat_Pad])
    return bondFeaturesMat_Pad_array

def get_GCN_protein_data(pdbids_list, pro_feature_dict):
    residue_aa_feature = []
    residue_bond_feature = []
    res_adj = []
    bond_adj = []
    for pdb_name in pdbids_list:
        residue_aa_feature.append(pro_feature_dict[pdb_name]["nodeFeaturesMat_Pad"])
        residue_bond_feature.append(pro_feature_dict[pdb_name]["bondFeaturesMat_Pad"])
        res_adj.append(pro_feature_dict[pdb_name]["conactsMat_Pad"])
        bond_adj.append(pro_feature_dict[pdb_name]["bondConnectMat_Pad"])  #
    return np.array(residue_aa_feature), np.array(residue_bond_feature),np.array(res_adj), np.array(bond_adj)

def return_GCN_protein_featuredim(pdb_name,pro_feature_dict):
     residue_feature,residue_bond_feature,_,_ = get_GCN_protein_data([pdb_name], pro_feature_dict)
     return residue_feature.shape[-1],residue_bond_feature.shape[-1]

def get_GCN_ligand_data(ligands_list,lig_feature_dict):
    atom_feature = []
    bond_feature = []
    atom_adj = []
    atom_mask = []
    for lig_name in ligands_list:
        atom_feature.append(lig_feature_dict[lig_name]["atomFeatureMat_Pad"])
        bond_feature.append(lig_feature_dict[lig_name]["bondFeatureMat_Pad"])
        atom_adj.append(lig_feature_dict[lig_name]["conactsMat_Pad"])
        atom_mask.append(lig_feature_dict[lig_name]["mask"])
    return np.array(atom_feature), np.array(bond_feature),np.array(atom_adj), np.array(atom_mask)
    # 

def return_GCN_ligand_featuredim(lig_name, lig_feature_dict):
    atom_feature, bond_feature, _,_ = get_GCN_ligand_data([lig_name], lig_feature_dict)
    return atom_feature.shape[-1], bond_feature.shape[-1]

def get_GCN_InterOut_data(pdbids_list, interOut_dict):
    inter_vals = []
    pocket_mask = []
    chainRes_idx = []
    for pdb_name in pdbids_list:
        inter_vals.append(interOut_dict[pdb_name]["InterOutVal_Pad"])
        pocket_mask.append(interOut_dict[pdb_name]["pocketMask_Pad"])
        chainRes_idx.append(interOut_dict[pdb_name]["chainRes_idx_Pad"])

    return np.array(inter_vals), np.array(pocket_mask),np.array(chainRes_idx)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)