import os
root = "step1_screen/after_plip_pdbs_0607/"
delFiles = [i for i in os.listdir(root) if i.endswith("_fixedPlip.pdb")]
print(len(delFiles))
for i in delFiles:
    os.remove(os.path.join(root,i))


root = "step1_screen/extract_bonds_pdbs/"
delFiles = [i for i in os.listdir(root) if i.endswith(".pdb")]
print(len(delFiles))
for i in delFiles:
    pdb_name = i.split("_")[0]
    os.rename(root+i,root + pdb_name + "_extractBonds.pdb" )
