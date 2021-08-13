# work on Server 3
import os
import shutil

ppath = "step1_screen/PLIP_results_0607_delHydrogenWithHOH/"
ori_pdb_path = "step1_screen/94_4_delHydrogenWithHOH/"
outpath = "step1_screen/after_plip_pdbs_0607/"
dirs = os.listdir(ppath)

num_1 = 0
num_3 = 0


for i in dirs:
    pdb_name = i.split("_")[0]
    i_path = ppath+i
    i_dirs_num = len(os.listdir(i_path))
    if i_dirs_num == 1:
        num_1 += 1 
        shutil.copy(ori_pdb_path + pdb_name + "_delHydrogenWithHOH.pdb",outpath+ pdb_name +"_notFixedPlip.pdb")
    elif i_dirs_num == 3:
        num_3 += 1
        fixedPdb = [i_path +"/"+ i for i in os.listdir(i_path) if i.startswith("plipfixed")][0]
        shutil.copy(fixedPdb,outpath+ pdb_name +"_fixedPlip.pdb")
    else:
        assert False
        
print("num_1", num_1)
print("num_3",num_3)
print("total",num_1 + num_3)