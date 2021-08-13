'''
look for the max_env_num = 2319,90% = 630 and then pad protein features with 0
'''

import pickle
import os
import numpy as np
import torch
import pandas as pd

# ================================================================================================== #


# ==============================================
# to delete the model_dataset's pdbid which has no related ligand pdb
def get_model_dataset_afterMatch(model_dataset,ligand_ligs_list,protein_pdbs):
    # delete miss lig in ligand_feature_dicts and miss pro in protein_feature_dicts
    model_dataset_new=[]
    miss_both_in=[]
    miss_in_ligands = []
    miss_in_proteins = []
    for i in range(model_dataset.shape[0]):
        if (model_dataset[i,1] in ligand_ligs_list) and (model_dataset[i,0] in protein_pdbs):  # the ligandid is in ligand_fea.pickle  & the pdbid is in dict_Vx
            model_dataset_new.append(model_dataset[i,:])
        elif (model_dataset[i,1] not in ligand_ligs_list) and (model_dataset[i,0] not in protein_pdbs):
            miss_both_in.append([model_dataset[i,0],model_dataset[i,1] ])
        elif model_dataset[i,0] not in protein_pdbs:
            miss_in_ligands.append([model_dataset[i,0],model_dataset[i,1]])
        else:
            miss_in_proteins.append([model_dataset[i,0],model_dataset[i,1]])

    model_dataset = np.array(model_dataset_new)
    print(model_dataset.shape[0])
    print("miss_both_in",len(miss_both_in), miss_both_in)
    print("miss_in_ligands",len(miss_in_ligands), miss_in_ligands)
    print("miss_in_proteins",len(miss_in_proteins), miss_in_proteins)

    return model_dataset
# ================================================================================================== #
# to expand the cols(aa) of InterOut to match proteinFea_aaIdx_list
def pad_interOut(InterOut_dict, model_dataset,protein_fea_dict):
    print("expanding InterOut...")
    sorted_InterOut = dict()
    proteinFea_notContain_dfDone_aaIdx = []
    max_pocket_len = 0
    for idx in range(model_dataset.shape[0]):
        pdb_name = model_dataset[idx,0]
        print("no.",idx,pdb_name)
        df_done = InterOut_dict[pdb_name]
        dfDone_aaIdx_list = np.array(df_done["pdbid1"])
        proteinFea_aaIdx_list = protein_fea_dict[pdb_name]["resID"]

        if not set(proteinFea_aaIdx_list) >= set(dfDone_aaIdx_list):
            print(proteinFea_aaIdx_list)
            print(dfDone_aaIdx_list)
            print("error: proteinFea_aaIdx notContain dfDone_aaIdx")
            proteinFea_notContain_dfDone_aaIdx.append(pdb_name)

        assert len(dfDone_aaIdx_list) == len(set(dfDone_aaIdx_list))
        assert len(proteinFea_aaIdx_list) == len(set(proteinFea_aaIdx_list))

        # expand
        df_done['pdbid1'] = df_done['pdbid1'].astype('category')
        df_done['pdbid1'].cat.set_categories(proteinFea_aaIdx_list, inplace=True)
        df_done = df_done.sort_values('pdbid1', ascending=True)
        if df_done.shape[0] > max_pocket_len:
            max_pocket_len = df_done.shape[0]

        sorted_InterOut[pdb_name] = {}
        sorted_InterOut[pdb_name]["aa_idx"] = np.array(df_done["pdbid1"])
        sorted_InterOut[pdb_name]["InterOutVal"] = np.array(df_done["faAtrRepElec_hbondSc"])

    print("max_pocket_len",max_pocket_len)

    print("saving sorted_InterOut:", len(sorted_InterOut.keys()))

    '''
    with open(sorted_InterOut_path, "wb") as f:
        pickle.dump(sorted_InterOut, f)
    f.close()
    '''

    print("done")
    return sorted_InterOut
#########################################################################


# ================================================================================================== #
# to delete the cols(aa) which has no Env feas in pairwise_babel
def pad_nonCovalent(model_dataset,nonCovalent_dict,sorted_InterOut):

    miss_babel_pdbids = []
    nonCovalent_notContain_proteinFea_aaIdx = []
    no_pairwise_mat_pdbs = []
    for idx in range(model_dataset.shape[0]):
        
        pdb_name = model_dataset[idx,0]
        print("===", idx,pdb_name)
        pairwise_mat = model_dataset[idx, -2]
        pairwise_mask = model_dataset[idx, -1]
        if not pairwise_mask:
            no_pairwise_mat_pdbs.append(pdb_name)
            continue
        nonCovalent_aaIdx_list = nonCovalent_dict[pdb_name]["aa_idx_list"]

        InterOut_aaIdx_list = sorted_InterOut[pdb_name]["aa_idx"]
        if not set(nonCovalent_aaIdx_list) >= set(InterOut_aaIdx_list):
            print(nonCovalent_aaIdx_list)
            print(InterOut_aaIdx_list)
            print("error: nonCovalent_aaIdx notContain InterOut_aaIdx")
            nonCovalent_notContain_proteinFea_aaIdx.append(pdb_name)

        assert len(nonCovalent_aaIdx_list) == len(set(nonCovalent_aaIdx_list))
        assert len(InterOut_aaIdx_list) == len(set(InterOut_aaIdx_list))
        assert len(nonCovalent_aaIdx_list) == pairwise_mat.shape[1]

        aa_used_list = np.zeros(len(nonCovalent_aaIdx_list),dtype=bool)
        match_aa_idx_list = []
        for i_aa in range(len(nonCovalent_aaIdx_list)):
            if nonCovalent_aaIdx_list[i_aa] in InterOut_aaIdx_list:
                aa_used_list[i_aa] = True
                match_aa_idx_list.append(nonCovalent_aaIdx_list[i_aa])
        # assert (np.array(match_aa_idx_list) == proteinFea_aaIdx_list).all()

        aa_used_list = aa_used_list.astype(np.bool).reshape(-1)
        pairwise_mat = pairwise_mat[:,aa_used_list]

        if len(np.unique(pairwise_mat.reshape(-1))) != 2:
            miss_babel_pdbids.append(pdb_name)
            pairwise_mask = False
            no_pairwise_mat_pdbs.append(pdb_name)
        else:
            # make aa_idx of pairwise_mat order
            df_temp = pd.DataFrame(pairwise_mat,columns=match_aa_idx_list)
            df_temp= df_temp[InterOut_aaIdx_list]
            assert (np.array(df_temp.columns) == InterOut_aaIdx_list).all()
            pairwise_mat = df_temp.values

        model_dataset[idx, -2] = pairwise_mat
        model_dataset[idx, -1] = pairwise_mask
        # break

    print("miss_babel_pdbids",len(miss_babel_pdbids),miss_babel_pdbids)
    print("nonCovalent_notContain_proteinFea_aaIdx",len(nonCovalent_notContain_proteinFea_aaIdx),nonCovalent_notContain_proteinFea_aaIdx)
    print("no_pairwise_mat_pdbs",len(no_pairwise_mat_pdbs),no_pairwise_mat_pdbs)

    # ==============================================================
    # save the result
    print("saving model dataset:", model_dataset.shape)

    '''
    with open(afterMatch_dataset_path, "wb") as f:
        pickle.dump(model_dataset, f)
    f.close()
    '''
    return model_dataset

def pad_ff(model_dataset,InterOut_dict,ligand_ligs_list,nonCovalent_dict,protein_fea_dict):
    protein_pdbs = protein_fea_dict.keys()
    model_dataset = get_model_dataset_afterMatch(model_dataset, ligand_ligs_list,protein_pdbs)
    sorted_InterOut = pad_interOut(InterOut_dict,model_dataset,protein_fea_dict)
    model_dataset = pad_nonCovalent(model_dataset,nonCovalent_dict,sorted_InterOut)

    return model_dataset, sorted_InterOut


