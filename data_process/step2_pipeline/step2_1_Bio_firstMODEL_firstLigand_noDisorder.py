import Bio
import os
import shutil
import pickle
import numpy as np
from Bio.PDB import PDBParser, Selection
from Bio.PDB import PDBParser,Select,PDBIO


def bio_get_firstModel_01(pdb,in_pdb_file):

    class FirstModelFirstLigand01(Select):
        def accept_residue(self, residue):
            global lig_count
            # 1. 小分子
            if residue.get_id()[0] != ' ':
                if lig == residue.get_resname() and lig_count == 0:
                    lig_count = 1
                    return True
            else:
                return True

    class NoDisorder(Select):
        def accept_atom(self, atom):
            return atom.get_serial_number() in atoms_serial_number

    p = PDBParser()
    io = PDBIO()
    # 01
    pro_structures = p.get_structure(pdb, in_pdb_file)
    firstModel = pro_structures[0]
    io.set_structure(firstModel)
    out_01_pdb_file = out_1_pdb_path + pdb + "_firstModelFirstLigand.pdb"
    io.save(out_01_pdb_file, FirstModelFirstLigand01())

    # 02
    pro_structures = p.get_structure(pdb, out_01_pdb_file)
    firstModel = pro_structures[0]
    atoms_serial_number = [a.get_serial_number() for a in firstModel.get_atoms()]
    io.set_structure(firstModel)
    out_02_pdb_file = out_2_pdb_path + pdb + "_noDisorder.pdb"
    io.save(out_02_pdb_file, NoDisorder())


# 1.load dataset to get pro & lig
fixed_path = r'E:\0430\code_DTA_0430\step1\output\func1_5_dataset_withoutNanCharge.pickle'
dataset_numpy = pickle.load(open(fixed_path, "rb"))
pdb_lig_array = dataset_numpy[:, 0:2]
pdb_lig_dict= {}
for i in pdb_lig_array:
    pdb_lig_dict[i[0]] = i[1]

# 2.  set outpath and inpath
out_pdb_path = "F:/LLL-doc/new_data_0501/pdbs_Bio_firstMODEL_firstLigand_nodisorder/"
out_1_pdb_path = out_pdb_path + "1_firstMODEL_firstLigand/"
out_2_pdb_path = out_pdb_path + "2_noDisorder/"

in_pdb_path = "F:/LLL-doc/new_data_0501/pdbs/"

# 3. main
count = 0
error_pdbs = []
for pdb in pdb_lig_dict.keys():
    count += 1
    pdb = pdb.split(".")[0]

    lig = pdb_lig_dict[pdb]
    print(count, pdb)
    lig_count = 0
    in_pdb_file = in_pdb_path + pdb + ".pdb"
    try:
        bio_get_firstModel_01(pdb,in_pdb_file)
    except:
        error_pdbs.append(pdb)

print("done")
print("error_pdbs",len(error_pdbs),error_pdbs)

'''
done
error_pdbs 0 []


D:\anaconda\envs\my-rdkit-env\lib\site-packages\Bio\PDB\StructureBuilder.py:92: PDBConstructionWarning: WARNING: Chain D is discontinuous at line 2722.
  PDBConstructionWarning,
'''