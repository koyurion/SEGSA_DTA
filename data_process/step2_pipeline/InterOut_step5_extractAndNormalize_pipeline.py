import os
import pandas as pd
import numpy as np
import pickle


def judgeInterOutResNum(pdb_path,df):
    # 1. get all res(resName,chain,resNum)
    map_dict = dict()
    miss_match_residue_list = []
    same_resNum_flag = True
    atomLines = [line.strip() for line in open(pdb_path).readlines() if line.strip().startswith("ATOM")]
    for line in atomLines:
        resNum = line[22:27].strip()
        chainID = line[21]
        chain_resNum = chainID + resNum
        resName = line[17:20]
        map_dict[chain_resNum] = resName
    # 2. match
    df_new =  df.reset_index(drop=True)
    for i in range(df_new.shape[0]):
        df_chain_resNum = df_new.at[i,"pdbid1"]
        if map_dict[df_chain_resNum] == df_new.at[i,"restype1"]:
            pass
        else:
            same_resNum_flag = False
            miss_match_residue_list.append([df_chain_resNum,map_dict[df_chain_resNum],df_new.at[i,"restype1"]])
    if not same_resNum_flag:
        print("miss_match_residue_list",miss_match_residue_list)
    return same_resNum_flag

def normalnize(df_col):
    K = 1.380649 * 10e-23
    T = 298.15
    NA = 6.02 * 10e23
    factor = 4180
    df_col_new = - (df_col * factor) / (K * T * NA)
    exp_df_col_new = df_col_new.apply(np.exp)
    exp_df_col_new = exp_df_col_new / exp_df_col_new.sum()
    return exp_df_col_new

def normalnize_fu(df_col):
    df_col_new = - df_col 
    exp_df_col_new = df_col_new.apply(np.exp)
    exp_df_col_new = exp_df_col_new / exp_df_col_new.sum()
    return exp_df_col_new
    
def normalnize_abs(df_col):
    df_col_new = abs(df_col)
    exp_df_col_new = df_col_new.apply(np.exp)
    exp_df_col_new = exp_df_col_new / exp_df_col_new.sum()
    return exp_df_col_new

def renamePDBID(pdbids):
    new_pdb_list = []
    for pdb in pdbids:
        if ":" in pdb:
            prex, icode = pdb.split(":")
            new_pdb_list.append(prex[-1]+prex[:-1]+icode)
        else:
            new_pdb_list.append(pdb[-1]+pdb[:-1])
    return new_pdb_list

def getResNumChain(pdb,L_pdb_path):
    chainID = "X"
    L_pdb_file = L_pdb_path + pdb + "_L.pdb"
    L_pdb_resNum = list(set([i[22:27] for i in open(L_pdb_file).readlines()]))
    assert len(L_pdb_resNum) == 1
    resNum = L_pdb_resNum[0].strip()
    # print(resNum+chainID)
    return resNum+chainID



def get_InterOut_pickle(out_InterOut_filepath,screenFlag = False,normalnizeFlag = True):

    if os.path.exists(out_InterOut_filepath):
        return pickle.load(open(out_InterOut_filepath,"rb"))
    else:
        result_files_path = "step2_pipeline/output/ALL_InterOut_output/"
        prex_path = "step2_pipeline/output/InterOut_step2_splitPDB/P_pdbs/"
        L_pdb_path = "step2_pipeline/output/InterOut_step2_splitPDB/L_pdbs/"
        result_files = [i for i in os.listdir(result_files_path) if i.endswith(".out")]
        print(len(result_files))

        dfs_dict = {}
        max_sum = -9e8
        min_sum = 9e8
        max_aa_num = -9e8
        sum_values = []
        empty_df_1x_1a4_pdbs = []
        big_tmp_sum = []
        changed_resNum_pdbs = []

        result_count = 0
        for file in result_files:
            result_count += 1
            print("===",result_count,file,"===")

            pdb_name = file.split(".")[0]
            prex = pdb_name[:-1]  # "1a4h" : "1a4"
            text = [i.split() for i in open(result_files_path + file).readlines()]  # read file

            # make df and transfer into numberic
            df = pd.DataFrame(text[1:], columns=text[0])
            df_1 = df.apply(pd.to_numeric, errors='ignore')

            # select 1X and prex("1a4")
            pdbid1_resNumChain = getResNumChain(pdb_name,L_pdb_path)
            print("pdbid1_resNumChain",pdbid1_resNumChain)

            if file == "6agg.out":
                df_1X_1a4 = df_1[(df_1.pdbid1.isin([pdbid1_resNumChain]) & df_1.restype1.isin([prex]))]
            else:
                df_1X_1a4 = df_1[(df_1.pdbid2.isin([pdbid1_resNumChain]) & df_1.restype2.isin([prex]))]

            # build sum-col and select target cols and update "pdbid1"-col
            df_1X_1a4["faAtrRepElec_hbondSc"] = df_1X_1a4["fa_atr"] + 0.55 * df_1X_1a4["fa_rep"] + df_1X_1a4["fa_elec"] + \
                                                df_1X_1a4["hbond_sc"]

            # df_done
            if file == "6agg.out":
                df_done = df_1X_1a4[["pdbid2", "restype2", "faAtrRepElec_hbondSc"]]
                df_done["pdbid2"] = renamePDBID(df_done["pdbid2"])
                df.rename(columns={"pdbid2": "pdbid1"})
            else:
                df_done = df_1X_1a4[["pdbid1", "restype1", "faAtrRepElec_hbondSc"]]
                df_done["pdbid1"] = renamePDBID(df_done["pdbid1"])


            if len(df_1X_1a4) == 0:
                empty_df_1x_1a4_pdbs.append(pdb_name)
                continue


            tmp_max = max(df_1X_1a4["faAtrRepElec_hbondSc"])
            tmp_min = min(df_1X_1a4["faAtrRepElec_hbondSc"])
            if tmp_max > max_sum:
                max_sum = tmp_max
            if tmp_min < min_sum:
                min_sum = tmp_min
            if df_done.shape[0] > max_aa_num:
                max_aa_num = df_done.shape[0]
            sum_values.extend(df_done["faAtrRepElec_hbondSc"])

            if screenFlag:
                # tmp_max > 1.5:
                print(file, tmp_max, tmp_min)
                big_tmp_sum.append(file)
            else:
                if normalnizeFlag == "Energy":
                    df_done["faAtrRepElec_hbondSc"] = normalnize(df_done["faAtrRepElec_hbondSc"])
                elif normalnizeFlag == "Fu":
                    df_done["faAtrRepElec_hbondSc"] = normalnize_fu(df_done["faAtrRepElec_hbondSc"])
                elif normalnizeFlag == "Abs":
                    df_done["faAtrRepElec_hbondSc"] = normalnize_abs(df_done["faAtrRepElec_hbondSc"])
                else:
                    print("unknow normalize method")
                    input()
                dfs_dict[pdb_name] = df_done

            pdb_path = prex_path + file.split(".")[0] + "_P.pdb"
            same_resNum_flag = judgeInterOutResNum(pdb_path, df_done)
            if not same_resNum_flag:
                changed_resNum_pdbs.append(file.split(".")[0])
            #break
        #
        print()
        print("=== program result summary ===")
        print("changed_resNum_pdbs",len(changed_resNum_pdbs),changed_resNum_pdbs)
        print("empty_df_1x_1a4_pdbs",len(empty_df_1x_1a4_pdbs),empty_df_1x_1a4_pdbs)
        print("big_tmp_sum",len(big_tmp_sum),big_tmp_sum)
        # save
        pickle.dump(dfs_dict,open(out_InterOut_filepath,"wb"))
        return dfs_dict