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
    io.save(out_4_pdb_path + pdb + "_delHydrogenWithHOH.pdb", DelHydrogen())


# 1.load dataset to get pro & lig
dataset_path = r"E:\0430\code_DTA_0430\step2\output\dataset_afterInterOut_5489.pickle"
ori_dataset = pickle.load(open(dataset_path,"rb"))

# 2.  set outpath and inpath
out_pdb_path = "F:/LLL-doc/new_data_0501/pdbs_Bio_firstMODEL_firstLigand_nodisorder/9_addWaterBased4/"
out_4_pdb_path = out_pdb_path + "94_4_delHydrogenWithHOH/"
in_pdb_path = out_pdb_path + "93_delALtLoc_delNoHetatm_addTERWithHOH/"

# 3. main
count = 0
error_pdbs = []
for idx in range(ori_dataset.shape[0]):
    count += 1
    pdb = ori_dataset[idx,0]
    lig = ori_dataset[idx,1]
    print(count, pdb)
    lig_count = 0
    in_pdb_file = in_pdb_path + pdb + "_delAltLocDelNoHetatmAddTERWithHOH.pdb"
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

'''
done
error_pdbs 0 []
'''