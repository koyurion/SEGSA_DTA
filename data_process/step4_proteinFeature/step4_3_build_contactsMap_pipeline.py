from data_process.step4_proteinFeature.aminoAcid_map import three_to_one, one_to_three,standard_aa_names
import Bio
import os
import shutil
import pickle
import numpy as np
import pandas as pd
import heapq
from sklearn.preprocessing import RobustScaler
from Bio.PDB import PDBParser,Select,PDBIO
from itertools import compress
import scipy.sparse as sp

def normalize(mx):
    """Pytorch_version fea/adj;TF_version fea"""
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(mx)

def edge_normalize(mx):
    """Pytorch_version fea/adj;TF_version fea"""
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(-1))
    r_inv = np.power(rowsum, -1)
    r_inv[np.isinf(r_inv)] = 0.
    r_inv = r_inv.reshape((mx.shape[0], mx.shape[1], -1))
    assert (r_inv.shape[-1] == 1) and (mx.shape[0] == mx.shape[1])
    return mx * r_inv

def normalize_half(adj):
    """TF_version adj: Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocsr()

def get_distance(idx,aa_CAcoord_array,neighbour_threshold,weightEdge_flag = True):
    dis = np.sqrt(
                np.sum(np.square(aa_CAcoord_array[idx,:] - aa_CAcoord_array[:,:]),1)
            )
    new_dis = []
    for d in dis:
        if d <= neighbour_threshold:
            if weightEdge_flag:
                new_dis.append(np.power(d, -1))
            else:
                new_dis.append(1.0)
        else:
            new_dis.append(0.)
    new_dis = np.array(new_dis)
    new_dis[np.isinf(new_dis)] = 0.
    # assert (new_dis == 0).all() == False
    return new_dis

def preprocess_df_aaIndex1():
    df_aaIndex_1 = pd.read_csv("/step4_proteinFeature/output/df_indices_aaindex1.csv")
    aaindex1_fea_dict = dict()
    df_aaIndex_1.index = df_aaIndex_1.iloc[:, 0]
    select_flags = np.array([df_aaIndex_1.iloc[2, :] == "None"]).astype(np.bool).reshape(-1)
    columns = df_aaIndex_1.iloc[0, select_flags]
    df_aaIndex_1_noMissVal = df_aaIndex_1.iloc[3:, select_flags]
    df_aaIndex_1_noMissVal.columns = columns
    # normalize
    transformer = RobustScaler().fit(df_aaIndex_1_noMissVal.values)
    df_aaIndex1_noMissVal_normalized = transformer.transform(df_aaIndex_1_noMissVal.values)
    for idx in range(len(df_aaIndex_1_noMissVal.index)):
        aa_threeToOne_name = three_to_one(df_aaIndex_1_noMissVal.index[idx])
        aaindex1_fea_dict[aa_threeToOne_name] = np.array(df_aaIndex1_noMissVal_normalized[idx, :])
    return aaindex1_fea_dict


def preprocess_df_aaIndex3():
    df_aaIndex_3 = pd.read_csv("step4_proteinFeature/output/df_indices_aaindex3.csv")
    aaindex3_fea_dict = dict()

    # columns
    df_aaIndex_3.columns = df_aaIndex_3.iloc[0, :]
    df_aaIndex_3 = df_aaIndex_3.iloc[1:, :]
    # index
    df_aaIndex_3.index = df_aaIndex_3.iloc[:, 0]
    df_aaIndex_3 = df_aaIndex_3.iloc[:, 1:]

    # normalize
    transformer = RobustScaler().fit(df_aaIndex_3.values)
    df_aaIndex_3_normalized = transformer.transform(df_aaIndex_3.values)

    # get dict
    for idx in range(len(df_aaIndex_3.index)):
        aaindex3_fea_dict[df_aaIndex_3.index[idx]] = np.array(df_aaIndex_3_normalized[idx, :])
    return aaindex3_fea_dict


def get_edge_features_mat(aa_name_list,conacts_mat,aaindex3_fea_dict):
    bond_features_mat_list = []
    bond_concats_mat_list = []
    max_neighbor_num = 0
    fea_len = len(aaindex3_fea_dict[list(aaindex3_fea_dict.keys())[0]])
    assert fea_len == 40
    for i in range(conacts_mat.shape[0]):
        sub_bond_fea_mat = []
        sub_bond_conact_mat = []
        res1 = aa_name_list[i]
        for j in range(conacts_mat.shape[1]):
            if i != j and conacts_mat[i,j] != 0:
                res2 = aa_name_list[j]
                feature_vector = aaindex3_fea_dict.get(res1 + res2) if type(aaindex3_fea_dict.get(res1 + res2,0)) != int else aaindex3_fea_dict.get(res2+res1)
                sub_bond_fea_mat.append(feature_vector)
                sub_bond_conact_mat.append(conacts_mat[i,j])

        if len(sub_bond_conact_mat) > max_neighbor_num:
            max_neighbor_num = len(sub_bond_conact_mat)
            
        if len(sub_bond_fea_mat) == 0 and len(sub_bond_conact_mat)==0:
            print("isolated node:",aa_name_list[i])
            sub_bond_fea_mat = [np.zeros(fea_len)]
            sub_bond_conact_mat = [0]
        sub_bond_fea_mat = normalize(sp.csr_matrix(sub_bond_fea_mat))
        bond_features_mat_list.append(sub_bond_fea_mat)
        bond_concats_mat_list.append(sub_bond_conact_mat)

    # bond_features_mat = np.array(bond_features_mat)
    # bond_concats_mat = np.array(bond_concats_mat)
    # print("bond_features_mat", bond_features_mat.shape)
    # print("bond_concats_mat", bond_concats_mat.shape)
    return bond_features_mat_list,bond_concats_mat_list,max_neighbor_num


def cutOff_resLen(aa_CAcoord_list,hetatm_atom_coord_list,cutoff = 630):
    # return boolean list
    aa_CAcoord_list = np.array(aa_CAcoord_list)
    hetatm_atom_coord_list = np.array(hetatm_atom_coord_list)
    min_dis_list = []
    for idx, res_coor in enumerate(aa_CAcoord_list):
        min_disTolig = min(np.sqrt(
            np.sum(np.square(res_coor - hetatm_atom_coord_list[:, :]), 1)
        ))
        min_dis_list.append([idx,min_disTolig])
    select_idx_list = np.array(heapq.nsmallest(cutoff, min_dis_list, key=lambda s: s[-1]))[:, 0]
    boolean_select_list = [True if idx in select_idx_list else False for idx in range(len(aa_CAcoord_list))]
    return boolean_select_list

def return_topu_idx_len(neighbour_idxs,conactsMat):
    topu_idx_list = []
    for idx in neighbour_idxs:
        topu_neighbour = conactsMat[idx, :]
        topu_idx = np.nonzero(topu_neighbour)  # nonzero: no matter weight or unweight
        # print(topu_idx)
        topu_idx_list.extend(topu_idx[0].tolist())
    topu_idx_list= list(set(topu_idx_list))
    return topu_idx_list


def topu_resLen(dfs_InterOut_dict,pdbid,conacts_mat, aa_idx_list, topu_threshold):
    interOut_aaidx_list = np.array(dfs_InterOut_dict[pdbid]["pdbid1"])
    neighbour_idxs = [aa_idx_list.index(i) for i in interOut_aaidx_list]
    for count in range(topu_threshold):
        topu_idx_list = return_topu_idx_len(neighbour_idxs, conacts_mat)
        neighbour_idxs = list(set(topu_idx_list + neighbour_idxs))
    boolean_select_list = [True if idx in neighbour_idxs else False for idx in range(len(aa_idx_list))]
    return boolean_select_list

###
def get_aa_idx(pdbid,pdb_file,ligand,aaindex1_fea_dict,dfs_InterOut_dict,\
               cutoff,neighbour_threshold,topu_threshold,weightEdge_flag,halfNormalizeFlag,aaindex3_fea_dict):

    # get all res
    p = PDBParser()
    structure = p.get_structure(pdbid, pdb_file)
    aa_idx_list = []
    aa_CAcoord_list = []
    hetatm_atom_coord_list = []
    aa_name_list = []
    meet_aa_list = []
    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            if chain_id == ' ':
                print("warning chainId empty:",pdbid)
                continue
            for res in chain:
                # remove HETATM and not Amino acid
                if res.get_id()[0] != ' ':  # or res.get_id()[2] != ' ':  # remove HETATM
                    if ligand == res.get_resname():
                        hetatm_atom_coord_list = [a.get_coord() for a in res]
                    continue
                if res.get_resname().upper() not in standard_aa_names:
                    continue
                if "CA" not in [atom.get_name() for atom in res]:
                    continue
                aa_idx = chain_id + str(res.get_id()[1]) + res.get_id()[2].strip()
                if aa_idx in aa_idx_list:  # for NUMMDL_0723
                    meet_aa_list.append(aa_idx)

                CA_coords = [atom.get_coord() for atom in res if atom.get_name() == "CA"]
                assert len(CA_coords) == 1

                aa_idx_list.append(aa_idx)
                aa_CAcoord_list.append(CA_coords[0])
                aa_name_list.append(three_to_one(res.get_resname()))

    # check meet
    if len(meet_aa_list) != 0:
        print("error meet_aa_list",len(meet_aa_list),meet_aa_list)
    '''
    # cutoff residue len: default = 630(90%)
    if len(aa_idx_list) > cutoff:
        print("cutoff > 630", pdbid)
        aa_less90PercentResLen = cutOff_resLen(aa_CAcoord_list,hetatm_atom_coord_list,cutoff)  # boolean llist
        aa_idx_list = list(compress(aa_idx_list, aa_less90PercentResLen))
        aa_CAcoord_list = list(compress(aa_CAcoord_list, aa_less90PercentResLen))
        aa_name_list = list(compress(aa_name_list, aa_less90PercentResLen))
    '''

    # get conacts_mat
    res_len = len(aa_idx_list)
    aa_CAcoord_array = np.array(aa_CAcoord_list)
    conacts_mat = []

    for res_idx in range(res_len):
        new_dis = get_distance(res_idx,aa_CAcoord_array,neighbour_threshold,weightEdge_flag)
        conacts_mat.append(new_dis)
    conacts_mat = np.array(conacts_mat)
    
    # topu resLen
    if topu_threshold != "all":
        aa_topu_ResLen = topu_resLen(dfs_InterOut_dict,pdbid, conacts_mat, aa_idx_list, topu_threshold)
        aa_idx_list = list(compress(aa_idx_list, aa_topu_ResLen))
        aa_name_list = list(compress(aa_name_list, aa_topu_ResLen))
        conacts_mat = np.array(list(compress(conacts_mat, aa_topu_ResLen))).T
        conacts_mat = np.array(list(compress(conacts_mat, aa_topu_ResLen)))
        # features_mat = np.array(list(compress(features_mat, aa_topu_ResLen)))
        
    # del those aa that without neighbor
    if False:
        aa_topu_ResLen = np.max(conacts_mat, axis=1) != 0
        aa_idx_list = list(compress(aa_idx_list, aa_topu_ResLen))
        aa_name_list = list(compress(aa_name_list, aa_topu_ResLen))
        conacts_mat = np.array(list(compress(conacts_mat, aa_topu_ResLen))).T
        conacts_mat = np.array(list(compress(conacts_mat, aa_topu_ResLen)))
        # features_mat = np.array(list(compress(features_mat, aa_topu_ResLen)))      

        assert (np.max(conacts_mat, axis=1) == 0).any() == False
    # get node_feature_mat aaindex1_fea_dict

    node_features_mat = []
    for res_name in aa_name_list:
        node_features_mat.append(aaindex1_fea_dict[res_name])
    node_features_mat = np.array(node_features_mat)

    # turn csr_matrix/ normalize: feature_mat; weight_conacts_mat; unweight_conacts_mat

    node_features = sp.csr_matrix(node_features_mat, dtype=np.float32)
    node_features = normalize(node_features)

    adj = sp.coo_matrix(conacts_mat, dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    if halfNormalizeFlag:
        adj = normalize_half(adj + sp.eye(adj.shape[0]))
    else:
        adj = normalize(adj + sp.eye(adj.shape[0]))

    # get edge_feature_mat aaindex3_fea_dict ; edge_normalize
    bond_features_mat_list,bond_concats_mat_list,max_neighbor_num = get_edge_features_mat(aa_name_list, adj, aaindex3_fea_dict)
    return aa_idx_list, aa_name_list, adj, node_features, bond_features_mat_list,bond_concats_mat_list,max_neighbor_num


def get_profea_pickle(prex_path,ori_dataset,dfs_InterOut_dict,cutoff = 630,neighbour_threshold = 6,topu_threshold=2,weightEdge_flag = True, halfNormalizeFlag= False):
    out_feaPath = prex_path + "step4_proteinFeature/output/" + "out4_3_protein_fea_dict_" + "cutOff"+str(cutoff)+"_neighbour"+str(neighbour_threshold)+ \
                  "_topu" + str(topu_threshold) + "_weight" +str(weightEdge_flag) +"_halfNormalize"+ str(halfNormalizeFlag) +".pickle"
    if os.path.exists(out_feaPath):
        return pickle.load(open(out_feaPath,"rb"))
    else:
        pdb_path = "../step1/pdbs_Bio_firstMODEL_firstLigand_nodisorder/4_delHydrogen/"
        print(ori_dataset.shape)

        # 2. read pd
        aaindex1_fea_dict = preprocess_df_aaIndex1()
        aaindex3_fea_dict = preprocess_df_aaIndex3()

        count = 0
        max_aa_len = 0
        max_aa_neighbor_num = 0
        protein_fea_dict = {}
        for idx in range(ori_dataset.shape[0]):
            count += 1
            pdb_name = ori_dataset[idx,0]
            lig_name = ori_dataset[idx,1]
            print(count,pdb_name,lig_name)
            pdb_file = pdb_path + pdb_name + "_delHydrogen.pdb"

            # get conacts_map; aaindex1_fea_dict

            aa_idx_list,aa_name_list,adj,node_features_mat,\
            bond_features_mat_list,bond_concats_mat_list,cur_neighbor_num = \
                get_aa_idx(pdb_name,pdb_file,lig_name,aaindex1_fea_dict,dfs_InterOut_dict,\
                                                        cutoff,neighbour_threshold,topu_threshold,weightEdge_flag,halfNormalizeFlag,aaindex3_fea_dict)
            
            temp_aa_len = len(aa_idx_list)
            if max_aa_len < temp_aa_len:
                max_aa_len = temp_aa_len
            if max_aa_neighbor_num < cur_neighbor_num:
                max_aa_neighbor_num = cur_neighbor_num

            protein_fea_dict[pdb_name] = {}
            protein_fea_dict[pdb_name]["resID"] = aa_idx_list      # "A60D"
            protein_fea_dict[pdb_name]["resName"] = aa_name_list         # "S"
            protein_fea_dict[pdb_name]["conactsMat"] = adj
            protein_fea_dict[pdb_name]["nodeFeaturesMat"] = node_features_mat
            protein_fea_dict[pdb_name]["bondFeaturesMat_list"] = bond_features_mat_list
            protein_fea_dict[pdb_name]["bondConcatsMat_list"] = bond_concats_mat_list

        protein_fea_dict["max_aa_len"] = max_aa_len
        protein_fea_dict["max_aa_neighbor_num"] = max_aa_neighbor_num

        print("protein_fea_dict", len(protein_fea_dict.keys()))
        print("max_aa_len", max_aa_len)
        print("max_aa_neighbor_num", max_aa_neighbor_num)

        # saving
        with open(out_feaPath, 'wb') as f1:
            pickle.dump(protein_fea_dict, f1)
        f1.close()
        return protein_fea_dict


