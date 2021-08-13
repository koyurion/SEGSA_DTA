import Bio
import os
import shutil
import pickle
import numpy as np
from Bio.PDB import PDBParser, Selection
from Bio.PDB import PDBParser,Select,PDBIO


def bio_delHydrogen(pdb,in_pdb_file):
    class DelHydrogen(Select):
        def accept_atom(self, atom):
            return atom.element not in ["H"]

    p = PDBParser()
    pro_structures = p.get_structure(pdb,in_pdb_file)
    firstModel = pro_structures[0]
    io = PDBIO()
    io.set_structure(firstModel)
    io.save(out_4_pdb_path + pdb + "_delHydrogen.pdb", DelHydrogen())


# 1.load dataset to get pro & lig
fixed_path = r'E:\0430\code_DTA_0430\step2\output\func2_2_dataset_noEmptyHetatm.pickle'
dataset_numpy = pickle.load(open(fixed_path, "rb"))
pdb_lig_array = dataset_numpy[:, 0:2]
pdb_lig_dict= {}
for i in pdb_lig_array:
    pdb_lig_dict[i[0]] = i[1]

# 2.  set outpath and inpath
out_pdb_path = "F:/LLL-doc/new_data_0501/pdbs_Bio_firstMODEL_firstLigand_nodisorder/"
out_4_pdb_path = out_pdb_path + "4_delHydrogen/"

in_pdb_path = out_pdb_path + "3_delALtLoc_delNoHetatm_addTER/"
print(dataset_numpy.shape)

# 3. main
count = 0
error_pdbs = []
for pdb in pdb_lig_dict.keys():
    count += 1
    pdb = pdb.split(".")[0]
    lig = pdb_lig_dict[pdb]
    print(count, pdb)
    lig_count = 0
    in_pdb_file = in_pdb_path + pdb + "_delAltLocDelNoHetatmAddTER.pdb"
    try:
        bio_delHydrogen(pdb,in_pdb_file)
    except:
        error_pdbs.append(pdb)

print("done")
print("error_pdbs",len(error_pdbs),error_pdbs)

'''
done
error_pdbs 0 []
'''