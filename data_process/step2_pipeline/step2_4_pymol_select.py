#!/usr/bin/python

import os
import __main__
import pickle
__main__.pymol_argc = ['pymol', '-qc']

import pymol

pymol.finish_launching()

out_pdb_path = "F:/LLL-doc/new_data_0501/pdbs_Bio_firstMODEL_firstLigand_nodisorder/"
out_5_pdb_path = out_pdb_path + "5_pymolSelect/"

in_pdb_path = out_pdb_path + "4_delHydrogen/"
in_pdbs = os.listdir(in_pdb_path)
print(len(in_pdbs))

fixed_path = r'E:\0430\code_DTA_0430\step2\output\func2_2_dataset_noEmptyHetatm.pickle'
dataset_numpy = pickle.load(open(fixed_path, "rb"))
pdb_lig_array = dataset_numpy[:, 0:2]
pdb_lig_dict= {}
for i in pdb_lig_array:
    pdb_lig_dict[i[0]] = i[1]


for idx in range(dataset_numpy.shape[0]):
    pdb = dataset_numpy[idx,0]
    # pdb = "3zzf"
    print(pdb)
    # 1. load
    pdb_file = in_pdb_path + pdb + "_delHydrogen.pdb"
    pymol.cmd.load(pdb_file)

    pro_name = pdb.split("_")[0]
    ligand_name = pdb_lig_dict[pro_name]  # 3PC
    # cmd.select("%s_%s" % (prefix, stretch), "none")
    '''
        select NAP3A, resn NAP around 2.6        
        select NAP3Aress,byres NAP3A
    '''
    # 2. select within 10A
    # cmd.select("negative","resn ASP+Glu and name OD*+OE*")
    atom10A_name = ligand_name + "10A"
    res10A_name = ligand_name + "10Aress"
    chain10A_name = ligand_name + "10Achain"
    out_pdb_name = pro_name + "_10A"
    # 3_1. select res_atom by lig
    pymol.cmd.select(atom10A_name, "resn " + ligand_name + " around 10")
    # 3_2. select resn by res_atom
    pymol.cmd.select(res10A_name, "byres " + atom10A_name)
    # 3_3. select chain by resn
    pymol.cmd.select(chain10A_name, "bychain " + res10A_name)

    # 4. select with ligand: # select aa,resn 3PC+ 3PC10A
    pymol.cmd.select(out_pdb_name, "resn " + ligand_name + " + " + chain10A_name)

    # 5. save
    pymol.cmd.save(out_5_pdb_path + pro_name + "_pymolSelect.pdb", out_pdb_name)
    # 6. delete
    pymol.cmd.delete("all")

print("done")
pymol.cmd.quit()
"""
done
"""