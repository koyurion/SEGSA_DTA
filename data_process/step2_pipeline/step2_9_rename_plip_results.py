import os
import shutil

ppath = "step1_screen/PLIP_results_0607_delHydrogenWithHOH/"
outpath = "step1_screen/after_rename_plip_results_0607/"
dirs = os.listdir(ppath)

miss_list = []
empty_list = []

for i in dirs:
    pdb_name = i.split("_")[0]
    ifile=ppath+i+"/report.txt"
    if os.path.exists(ifile):
        shutil.copy(ifile,outpath+"output_"+ pdb_name +".txt")
    else:
        miss_list.append(pdb_name)
    if os.path.getsize(ifile) == 0:
        empty_list.append(i)
        
print("miss_list", miss_list)
print("empty_list",empty_list)
print("done")