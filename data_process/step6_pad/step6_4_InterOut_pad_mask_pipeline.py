import pickle
import os
import gc
import numpy as np
import scipy.sparse as sp
"""
1. protein_fea
2. concat_map
3. Inter_out

"""
# max_aa_num = 630 + 1 # add 1 so everything can pad that empty res
max_pocket_len = 54
max_atom_len = 48
def get_InterOut_mask(sorted_InterOut,protein_fea_dict,sorted_InterOut_dict_Padded_path):
    # output
    if os.path.exists(sorted_InterOut_dict_Padded_path):
        with open(sorted_InterOut_dict_Padded_path, "rb") as f:
            sorted_InterOut_dict_Padded = pickle.load(f)
        return sorted_InterOut_dict_Padded

    else:
        sorted_InterOut_dict_Padded = {}
        count = 0
        max_aa_num = protein_fea_dict["max_aa_len"] + 1
        for pdb_name in sorted_InterOut.keys():
            count += 1
            InterOut_aa_idx = sorted_InterOut[pdb_name]["aa_idx"]
            InterOutVal = sorted_InterOut[pdb_name]["InterOutVal"]

            expandLen = len(InterOutVal)
            #print("expandLen",expandLen)

            pad_pocket_mask = np.zeros(max_pocket_len, dtype=np.float32)
            pad_InterOutVals = np.zeros(max_pocket_len, dtype=np.float32)
            pad_chainRes_idx = np.zeros(max_pocket_len, dtype=np.float32)

            pad_pocket_mask[:expandLen] = 1.0
            pad_InterOutVals[:expandLen] = InterOutVal

            pad_chainRes_idx.fill(max_aa_num - 1)  # fill the last one empty residue
            proteinFea_aaIdx_list = protein_fea_dict[pdb_name]["resID"]
            for i in range(len(InterOut_aa_idx)):
                aa_idx = InterOut_aa_idx[i]
                index = proteinFea_aaIdx_list.index(str(aa_idx))
                pad_chainRes_idx[i] = index

            sorted_InterOut_dict_Padded[pdb_name] = {}
            sorted_InterOut_dict_Padded[pdb_name]["pocketMask_Pad"] = pad_pocket_mask
            sorted_InterOut_dict_Padded[pdb_name]["InterOutVal_Pad"] = pad_InterOutVals
            sorted_InterOut_dict_Padded[pdb_name]["chainRes_idx_Pad"] = pad_chainRes_idx

            if False:
                np.savetxt("pad_chainRes_idx.csv", pad_chainRes_idx, delimiter=",")
                np.savetxt("expand_InterOut_dict_Padded.csv", pad_InterOutVals, delimiter=",")
                np.savetxt("pocket_mask_Padded.csv", pad_pocket_mask, delimiter=",")

        print("saving pad_InterOut:", len(sorted_InterOut_dict_Padded.keys()))

        with open(sorted_InterOut_dict_Padded_path, "wb") as f:
            pickle.dump(sorted_InterOut_dict_Padded, f)
        f.close()

        return sorted_InterOut_dict_Padded


def get_pairwise_mask(pad_dataset_path,model_dataset):

    if os.path.exists(pad_dataset_path):
        with open(pad_dataset_path, "rb") as f:
            model_dataset = pickle.load(f)
        return model_dataset
    else:
        for idx in range(model_dataset.shape[0]):
            pairwise_mat = model_dataset[idx, -2]
            pad_pairwise_mat = np.zeros((max_atom_len,max_pocket_len))
            pad_pairwise_mat[:pairwise_mat.shape[0], :pairwise_mat.shape[1]] = pairwise_mat

            model_dataset[idx,-2] = pad_pairwise_mat

        with open(pad_dataset_path, "wb") as f:
            pickle.dump(model_dataset, f)
        f.close()
        return model_dataset


def get_Protein_mask(Padded_protein_path,protein_fea_dict):
    if os.path.exists(Padded_protein_path):
        with open(Padded_protein_path, "rb") as f:
            protein_fea_dict_Padded = pickle.load(f)
        return protein_fea_dict_Padded
    else:
        count = 0
        max_aa_num = protein_fea_dict["max_aa_len"] + 1  # add 1 so everything can pad that empty res
        max_aa_neighbor_num = 15 if protein_fea_dict["max_aa_neighbor_num"] <= 15 else protein_fea_dict["max_aa_neighbor_num"]
        print("max_aa_neighbor_num",max_aa_neighbor_num,protein_fea_dict["max_aa_neighbor_num"])
        protein_fea_dict_Padded = {}
        for pdb_name in protein_fea_dict.keys():
            if pdb_name == "max_aa_len" or pdb_name == "max_aa_neighbor_num":
                continue
            count += 1
            print(count, pdb_name)

            # get origin
            aa_idx_list = protein_fea_dict[pdb_name]["resID"]  # "A60D"
            aa_name_list = protein_fea_dict[pdb_name]["resName"]  # "S"
            conacts_mat = protein_fea_dict[pdb_name]["conactsMat"]
            node_features_mat = protein_fea_dict[pdb_name]["nodeFeaturesMat"]
            bond_features_mat_list = protein_fea_dict[pdb_name]["bondFeaturesMat_list"]
            bond_concats_mat_list = protein_fea_dict[pdb_name]["bondConcatsMat_list"]

            # get mask
            chainRes_mask = np.zeros(max_aa_num)
            row, node_col = node_features_mat.shape
            _, bond_col = bond_features_mat_list[0].shape
            chainRes_mask[:row] = 1.0

            # padding
            pad_res_num = max_aa_num - row
            pad_node_featureMat = sp.vstack([node_features_mat, sp.csr_matrix(np.zeros((pad_res_num, node_col)))])

            temp_conacts_mat = sp.coo_matrix(np.zeros((pad_res_num, pad_res_num)))
            pad_conacts_mat = sp.bmat([[conacts_mat, None], [None, temp_conacts_mat]])

            '''
            pad_bond_featureMat = np.zeros((max_aa_num,max_aa_num,bond_col))
            for i in range(row_1):
                pad_bond_featureMat[i,:row_2] = edge_features_mat[i]
            '''

            # bondFeature - conact_mat
            bond_concats_mat = np.zeros((max_aa_num,1,max_aa_neighbor_num))
            assert row == len(bond_concats_mat_list)
            for cml_idx in range(len(bond_concats_mat_list)):
                val = bond_concats_mat_list[cml_idx]
                bond_concats_mat[cml_idx,0,:len(val)] = val

            # bondFeature - bond_feture
            bond_features_mat = np.zeros((max_aa_num, max_aa_neighbor_num,bond_col))
            bfml_count = 0
            for bfml in bond_features_mat_list:
                pad_res_num = max_aa_neighbor_num - bfml.shape[0]
                # print(pad_res_num, bfml.shape, sp.csr_matrix(np.zeros((pad_res_num, bond_col))).shape)
                pad_bond_featureMat = sp.vstack([bfml, sp.csr_matrix(np.zeros((pad_res_num, bond_col)))]).toarray()
                bond_features_mat[bfml_count] = pad_bond_featureMat

            protein_fea_dict_Padded[pdb_name] = {}
            protein_fea_dict_Padded[pdb_name]["resID"] = aa_idx_list  # "A60D"
            protein_fea_dict_Padded[pdb_name]["resName"] = aa_name_list  # "S"
            protein_fea_dict_Padded[pdb_name]["conactsMat_Pad"] = pad_conacts_mat.toarray()
            protein_fea_dict_Padded[pdb_name]["nodeFeaturesMat_Pad"] = pad_node_featureMat.toarray()
            protein_fea_dict_Padded[pdb_name]["bondFeaturesMat_Pad"] = bond_features_mat # max_aa_num, max_aa_neighbor_num,bond_col
            protein_fea_dict_Padded[pdb_name]["bondConnectMat_Pad"] = bond_concats_mat  # max_aa_num,1,max_aa_neighbor_num
            protein_fea_dict_Padded[pdb_name]["chainResMask"] = chainRes_mask

        with open(Padded_protein_path, "wb") as f:
            pickle.dump(protein_fea_dict_Padded, f)
        f.close()
        # save the result
        print("saving pad protein dict:", len(protein_fea_dict_Padded.keys()))

        return protein_fea_dict_Padded

def pad_mask(model_dataset,sorted_InterOut,protein_fea_dict,\
             output_InterOut_Padded_path,output_dataset_Padded_path,output_protein_Padded_path):
    print("===========Pad InterOut==========================")
    sorted_InterOut_dict_Padded  = get_InterOut_mask(sorted_InterOut,protein_fea_dict,output_InterOut_Padded_path)

    print("===========Pad pairwise==========================")
    model_dataset = get_pairwise_mask(output_dataset_Padded_path,model_dataset)

    print("===========Pad Protein==========================")
    protein_fea_dict_Padded = get_Protein_mask(output_protein_Padded_path,protein_fea_dict)

    print("done pad")

    return model_dataset,protein_fea_dict_Padded,sorted_InterOut_dict_Padded