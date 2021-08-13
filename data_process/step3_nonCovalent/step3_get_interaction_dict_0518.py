from rdkit import Chem
from Bio.PDB import PDBParser, Selection
from Bio.PDB.Polypeptide import three_to_one
import os
import pickle
import numpy as np
import warnings

warnings.filterwarnings("ignore")
standard_aa_names = [
    "ALA",
    "CYS",
    "ASP",
    "GLU",
    "PHE",
    "GLY",
    "HIS",
    "ILE",
    "LYS",
    "LEU",
    "MET",
    "ASN",
    "PRO",
    "GLN",
    "ARG",
    "SER",
    "THR",
    "VAL",
    "TRP",
    "TYR",
]

# S1
def get_atoms_from_pdb(ligand, pdbid):
    # from pdb protein structure, get ligand index list for bond extraction
    p = PDBParser()
    atom_idx_list = []
    atom_name_list = []
    disorder_atom_error = []
    structure = p.get_structure(pdbid, pdb_path + pdbid + '_extractBonds.pdb')
    seq_dict = {}
    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            id_list = []
            for res in chain:
                if ligand == res.get_resname():
                    if res.get_id()[0] == ' ':  # remove ATOM: res.get_id()[0] == ' ' 代表ATOM行
                        continue
                    for atom in res:
                        if atom.is_disordered():  # if disorder, choose "A"
                            # print(pdbid,atom.get_serial_number(),atom.get_id())
                            assert not atom.is_disordered()
                        atom_idx_list.append(atom.get_serial_number())  # pdb文件中HETATM行的第二列： 即 非标准基团 原子序号： 2618
                        atom_name_list.append(atom.get_id())  # name: 原子的id, 即O1 C1 C2等

    if len(atom_idx_list) != 0:
        return atom_idx_list, atom_name_list
    else:
        return None, None


# S2
def get_bonds(pdbid, ligand, atom_idx_list):
    bond_list = []
    plip_file = plip_path+'output_' + pdbid + '.txt'
    if not os.path.exists(plip_file):
        return None
    f = open(plip_file)
    isheader = False
    for line in f.readlines():
        if line[0] == '*':  # 1 **Hydrogen Bonds**
            bond_type = line.strip().replace('*', '')
            isheader = True
        if line[0] == '|':
            if isheader:
                header = line.replace(' ', '').split(
                    '|')  # 2 ['', 'RESNR', 'RESTYPE', 'RESCHAIN', 'RESNR_LIG', ...,'PROTCOO', '']
                isheader = False
                continue
            lines = line.replace(' ', '').split('|')
            if (ligand not in lines[5]) and (ligand not in lines[6]):
                print("error no target ligand",lines)
                continue
            if bond_type in ['Hydrogen Bonds', 'Water Bridges']:
                # print(atom_idx_list)
                aa_id, aa_name, aa_chain, ligand_id, ligand_name, ligand_chain = int(lines[1]), lines[2], lines[3], int(
                    lines[4]), lines[5], lines[6]
                atom_idx1, atom_idx2 = int(lines[12]), int(lines[14])  # DONORIDX ACCEPTORIDX(321 ,2627)
                if atom_idx1 in atom_idx_list and atom_idx2 in atom_idx_list:  # discard ligand-ligand interaction
                    continue
                if atom_idx1 in atom_idx_list:
                    atom_idx_ligand, atom_idx_protein = atom_idx1, atom_idx2
                elif atom_idx2 in atom_idx_list:
                    atom_idx_ligand, atom_idx_protein = atom_idx2, atom_idx1
                else:
                    print(pdbid, ligand, bond_type, 'error: atom index in plip result not in atom_idx_list')
                    print(atom_idx1, atom_idx2)
                    # return None
                    continue

                bond_list.append((bond_type + '_' + str(len(bond_list)), aa_chain, aa_name, aa_id, [atom_idx_protein],
                                  ligand_chain, ligand_name, ligand_id, [atom_idx_ligand]))
            elif bond_type == 'Hydrophobic Interactions':
                aa_id, aa_name, aa_chain, ligand_id, ligand_name, ligand_chain = int(lines[1]), lines[2], lines[3], int(
                    lines[4]), lines[5], lines[6]
                atom_idx_ligand, atom_idx_protein = int(lines[8]), int(lines[9])
                if atom_idx_ligand in atom_idx_list and atom_idx_protein in atom_idx_list:  # discard ligand-ligand interaction
                    print('error: atom_idx_ligand and atom_idx_protein are in atom_idx_list')
                    print('Hydrophobic Interactions', atom_idx_ligand, atom_idx_protein)
                    continue
                elif atom_idx_ligand not in atom_idx_list:
                    print('error: atom index in plip result not in atom_idx_list')
                    print('Hydrophobic Interactions', atom_idx_ligand, atom_idx_protein)
                    # return None
                    continue
                bond_list.append((bond_type + '_' + str(len(bond_list)), aa_chain, aa_name, aa_id, [atom_idx_protein],
                                  ligand_chain, ligand_name, ligand_id, [atom_idx_ligand]))
            elif bond_type in ['pi-Stacking']:
                aa_id, aa_name, aa_chain, ligand_id, ligand_name, ligand_chain = int(lines[1]), lines[2], lines[3], int(
                    lines[4]), lines[5], lines[6]
                atom_idx_ligand_list = list(
                    map(int, lines[12].split(',')))
                atom_idx_protein_list = list(map(int, lines[7].split(',')))
                if len(set(atom_idx_ligand_list).intersection(set(atom_idx_list))) != len(
                        atom_idx_ligand_list):
                    print(bond_type, 'error: atom index in plip result not in atom_idx_list')
                    print(atom_idx_ligand_list)
                    # return None
                    continue
                bond_list.append((bond_type + '_' + str(len(bond_list)), aa_chain, aa_name, aa_id, atom_idx_protein_list, ligand_chain,
                                  ligand_name, ligand_id, atom_idx_ligand_list))
            elif bond_type in ['pi-Cation Interactions']:
                aa_id, aa_name, aa_chain, ligand_id, ligand_name, ligand_chain = int(lines[1]), lines[2], lines[3], int(
                    lines[5]), lines[6], lines[7]
                atom_idx_ligand_list = list(
                    map(int, lines[12].split(',')))
                atom_idx_protein_list = list(map(int, lines[4].split(',')))
                if len(set(atom_idx_ligand_list).intersection(set(atom_idx_list))) != len(
                        atom_idx_ligand_list):
                    print(bond_type, 'error: atom index in plip result not in atom_idx_list')
                    print(atom_idx_ligand_list)
                    # return None
                    continue
                bond_list.append((bond_type + '_' + str(len(bond_list)), aa_chain, aa_name, aa_id, atom_idx_protein_list, ligand_chain,
                          ligand_name, ligand_id, atom_idx_ligand_list))

            elif bond_type == 'Salt Bridges':
                aa_id, aa_name, aa_chain, ligand_id, ligand_name, ligand_chain = int(lines[1]), lines[2], lines[3], int(
                    lines[5]), lines[6], lines[7]
                atom_idx_ligand_list = list(
                    set(map(int, lines[11].split(','))))
                atom_idx_protein_list = list(map(int, lines[4].split(',')))
                if len(set(atom_idx_ligand_list).intersection(set(atom_idx_list))) != len(atom_idx_ligand_list):
                    print('error: atom index in plip result not in atom_idx_list')
                    print('Salt Bridges', atom_idx_ligand_list,
                          set(atom_idx_ligand_list).intersection(set(atom_idx_list)))
                    # return None
                    continue
                bond_list.append((bond_type + '_' + str(len(bond_list)), aa_chain, aa_name, aa_id, atom_idx_protein_list, ligand_chain,
                                  ligand_name, ligand_id, atom_idx_ligand_list))
            elif bond_type == 'Halogen Bonds':
                aa_id, aa_name, aa_chain, ligand_id, ligand_name, ligand_chain = int(lines[1]), lines[2], lines[3], int(
                    lines[4]), lines[5], lines[6]
                atom_idx1, atom_idx2 = int(lines[11]), int(lines[13])
                if atom_idx1 in atom_idx_list and atom_idx2 in atom_idx_list:  # discard ligand-ligand interaction
                    continue
                if atom_idx1 in atom_idx_list:
                    atom_idx_ligand, atom_idx_protein = atom_idx1, atom_idx2
                elif atom_idx2 in atom_idx_list:
                    atom_idx_ligand, atom_idx_protein = atom_idx2, atom_idx1
                else:
                    print('error: atom index in plip result not in atom_idx_list')
                    print('Halogen Bonds', atom_idx1, atom_idx2)
                    continue
                    # return None
                bond_list.append((bond_type + '_' + str(len(bond_list)), aa_chain, aa_name, aa_id, [atom_idx_protein],
                                  ligand_chain, ligand_name, ligand_id, [atom_idx_ligand]))
            else:
                print("error")
                print('bond_type', bond_type)
                print(header)
                print(lines)
                # return None
                continue
    f.close()
    if len(bond_list) != 0:
        return bond_list
    else:
        return None


# S3
def get_interact_atom_name(atom_idx_list, atom_name_list, bond_list):
    interact_atom_name_list = []
    interact_bond_type_list = []
    interact_atom_name_set = set()
    assert len(atom_idx_list) == len(atom_name_list)
    for bond in bond_list:
        for atom_idx in bond[-1]:
            atom_name = atom_name_list[atom_idx_list.index(atom_idx)]  # 如，2618
            # if atom_name not in interact_atom_name_set:
            interact_atom_name_set.add(atom_name)
            interact_atom_name_list.append(atom_name)  # O1
            interact_bond_type_list.append((atom_name, bond[0]))  # （O1，Hydrogen Bonds_0）
    return interact_atom_name_list, interact_bond_type_list


# S4
def get_mol_from_ligandpdb(ligand):
    lig_file = lig_path + ligand + '_ideal.pdb'
    if not os.path.exists(lig_file):
        return None, None, None
    if os.path.getsize(lig_file) == 0:
        return None, None, None
    name_order_list = []
    name_to_idx_dict, name_to_element_dict = {}, {}

    p = PDBParser()
    structure = p.get_structure(ligand, lig_file)

    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            for res in chain:
                if ligand == res.get_resname():
                    # print(ligand,res.get_resname(),res.get_full_id())
                    for atom in res:
                        name_order_list.append(atom.get_id())
                        name_to_element_dict[atom.get_id()] = atom.element
                        name_to_idx_dict[atom.get_id()] = atom.get_serial_number() - 1
    # print('check', name_to_idx_dict.items())
    if len(name_to_idx_dict) == 0:
        return None, None, None
    return name_order_list, name_to_idx_dict, name_to_element_dict


# S5
def get_interact_atom_list(name_order_list, atom_name_to_idx_dict, atom_name_to_element_dict, interact_atom_name_list):
    atom_idx_list = []
    atom_name_list = []
    atom_element_list = []
    atom_interact_list = []
    for name in name_order_list:
        idx = atom_name_to_idx_dict[name]
        atom_idx_list.append(idx)
        atom_name_list.append(name)
        atom_element_list.append(atom_name_to_element_dict[name])
        # 'O1' in ['O1','C1',...]
        atom_interact_list.append(
            int(name in interact_atom_name_list))
    return atom_idx_list, atom_name_list, atom_element_list, atom_interact_list


### second part
def get_aa_idx(pdbid):
    p = PDBParser()
    structure = p.get_structure(pdbid, pdb_path + pdbid + '_extractBonds.pdb')
    aa_idx_list = []
    aaidx_atomidx_dict = dict()
    meet_aa_list = []
    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            if chain_id == ' ':
                print("warning chainId empty:",pdbid)
                continue
            seq = ''
            id_list = []
            for res in chain:

                # print(res.get_id())
                if res.get_id()[0] != ' ' or res.get_resname().upper() not in standard_aa_names:# or res.get_id()[2] != ' ':  # remove HETATM
                    continue
                if "CA" not in [atom.get_name() for atom in res]:
                    continue
                aa_idx = chain_id + str(res.get_id()[1]) + res.get_id()[2].strip()
                if aa_idx in aa_idx_list:  # for NUMMDL_0723
                    meet_aa_list.append(aa_idx)
                aa_idx_list.append(aa_idx)
                aaidx_atomidx_dict[aa_idx] = [atom.get_serial_number() for atom in res]
    if len(meet_aa_list) != 0:
        print("error meet_aa_list",len(meet_aa_list),meet_aa_list)
    return aa_idx_list,aaidx_atomidx_dict


def get_interact_residue(aa_idx_list, bond_list,aaidx_atomidx_dict):
    interact_residue_list = []
    for bond in bond_list:  # bond[1]: aa_chain; bond[2]: aa_name; bond[3]: aa_id  # 'A','SER'，'37'
        aa_name = bond[1] + str(bond[3])
        if aa_name not in aa_idx_list:
            for key,resAtoms in aaidx_atomidx_dict.items():
                if len(set(bond[4]).intersection(set(resAtoms))) == len(bond[4]):
                    aa_name = key
        if aa_name not in aa_idx_list:
            print("warning aa_name not in aa_idx_list:",aa_name,aa_idx_list)
            continue
        interact_residue_list.append((aa_name, bond[0]))  # [("A1","Hydrogen Bonds_0"),...]
    if len(interact_residue_list) != 0:
        return interact_residue_list
    else:
        return None


####   third  part ###
def get_mol_dict():
    if os.path.exists(mol_dict_path):
        with open(mol_dict_path, "rb") as f:
            mol_dict = pickle.load(f)
    else:
        mol_dict = {}
        ComponentsPub_path  = "./step1_screen/lig_mol_data/Components - pub.sdf"
        mols = Chem.SDMolSupplier(ComponentsPub_path)
        for m in mols:
            if m is None:
                continue
            name = m.GetProp("_Name")
            mol_dict[name] = m
        with open(mol_dict_path, 'wb') as f:
            pickle.dump(mol_dict, f)
    # print('mol_dict',len(mol_dict))
    return mol_dict


def get_pairwise_label(pdbid, interaction_dict):
    if pdbid in interaction_dict:
        sdf_element = np.array([atom.GetSymbol().upper() for atom in mol.GetAtoms()])
        atom_element = np.array(interaction_dict[pdbid]['atom_element'], dtype=str)
        atom_name_list = np.array(interaction_dict[pdbid]['atom_name'], dtype=str)
        atom_interact = np.array(interaction_dict[pdbid]['atom_interact'], dtype=int)
        nonH_position = np.where(atom_element != ('H'))[0]
        # print(atom_element[nonH_position],sdf_element)
        assert sum(atom_element[nonH_position] != sdf_element) == 0

        atom_name_list = atom_name_list[nonH_position].tolist()
        aa_name_list = interaction_dict[pdbid]['aa_idx_list']
        pairwise_mat = np.zeros((len(nonH_position), len(aa_name_list)), dtype=np.int32)
        # print(len(nonH_position), len(aa_name_list))
        for atom_name, bond_type in interaction_dict[pdbid]['atom_bond_type']:
            atom_idx = atom_name_list.index(str(atom_name))
            assert atom_idx < len(nonH_position)

            aa_idx_list = []
            for aa_name, bond_type_seq in interaction_dict[pdbid]['residue_bond_type']:
                if bond_type == bond_type_seq:
                    aa_idx = aa_name_list.index(str(aa_name))
                    aa_idx_list.append(aa_idx)

                    # print(bond_type,atom_name,aa_name,atom_idx,aa_idx)
                    pairwise_mat[atom_idx, aa_idx] = 1

        if len(np.where(pairwise_mat != 0)[0]) != 0:
            pairwise_mask = True
            return pairwise_mask, pairwise_mat

    return False, np.zeros((1, 1))


log_file = "output/log_step3_getInteractionDict.txt"
miss_ligand_in_pdb_list = []
empty_bond_list = []
miss_ideal_pdb_error_list = []
empty_atom_interact_list = []
protein_aa_error_list = []
miss_lig_in_moldict_list = []
miss_smiles_list = []


dataset_path = "../step2_pipeline/output/dataset_afterInterOut_5489.pickle"
ori_dataset = pickle.load(open(dataset_path,"rb"))

pdb_path = "../step1_screen/extract_bonds_pdbs/"
plip_path = "./after_rename_plip_results_0607/"
lig_path = '../step1_screen/ligands/'
mol_dict_path = "./step1_screen/lig_mol_data/mol_dict"

interaction_dict = {}
model_dataset = []
count = 0
if True:
    for idx_dataset in range(ori_dataset.shape[0]):
        if True:
            # first part: ligand, atom_bond_type
            # pdb_id,lig_name,lig_chain,lig_id=line.strip().split()
            pdb_id, lig_name, measurement, aff_value = ori_dataset[idx_dataset,:]
            # pdb_id, lig_name, measurement, aff_value = "1tok","MAE","KD","5"
            count += 1

            print(count, pdb_id, lig_name)

            # S1
            atom_idx_list, atom_name_list = get_atoms_from_pdb(lig_name, pdb_id)  # for bond atom identification

            # break
            if atom_idx_list is None:
                miss_ligand_in_pdb_list.append([pdb_id, lig_name])
                continue
            # S2
            bond_list = get_bonds(pdb_id, lig_name, atom_idx_list)
            """for bl in bond_list:
                print(bl)
            input()"""
            if bond_list is None:
                empty_bond_list.append([pdb_id, lig_name, len(atom_idx_list)])
                continue
            # S3
            interact_atom_name_list, interact_bond_type_list = get_interact_atom_name(atom_idx_list, atom_name_list,
                                                                                      bond_list)
            """print(len(interact_atom_name_list), interact_atom_name_list)
            print(len(interact_bond_type_list), interact_bond_type_list)
            input()"""
            # S4
            name_order_list, atom_name_to_idx_dict, atom_name_to_element_dict = get_mol_from_ligandpdb(lig_name)
            """print(name_order_list)
            print(atom_name_to_idx_dict)
            input()"""
            if atom_name_to_idx_dict == None:
                miss_ideal_pdb_error_list.append([pdb_id, lig_name])
                continue
            # S5

            atom_idx_list, atom_name_list, atom_element_list, atom_interact_list \
                = get_interact_atom_list(name_order_list, atom_name_to_idx_dict, atom_name_to_element_dict,
                                         interact_atom_name_list)
            """print()
            print(len(atom_idx_list), atom_idx_list)
            print(len(atom_name_list), atom_name_list)
            print(len(atom_element_list), atom_element_list)
            print(len(atom_interact_list), atom_interact_list)
            input()"""

            if len(atom_idx_list) == 0:
                empty_atom_interact_list.append([pdb_id, lig_name.len(name_order_list)])
                continue

        ### secoond part: protein,residue_bond_type
        aa_idx_list,aaidx_atomidx_dict = get_aa_idx(pdb_id)
        # print("aa_idx_list",len(aa_idx_list),aa_idx_list)

        interact_residue_list = get_interact_residue(aa_idx_list, bond_list,aaidx_atomidx_dict)
        """print(interact_residue_list)
        input()"""
        if interact_residue_list is None:
            if interact_residue_list is None:
                protein_aa_error_list.append([pdb_id, lig_name])
                continue

        #### thrid part: pairwise_label

        interaction_dict[pdb_id] = {}
        interaction_dict[pdb_id]["ligand"] = lig_name
        interaction_dict[pdb_id]["measurement"] = measurement
        interaction_dict[pdb_id]["aff_value"] = float(aff_value)
        interaction_dict[pdb_id]['bond'] = bond_list
        interaction_dict[pdb_id]['atom_idx'] = atom_idx_list
        interaction_dict[pdb_id]['atom_name'] = atom_name_list
        interaction_dict[pdb_id]['atom_element'] = atom_element_list
        interaction_dict[pdb_id]['atom_interact'] = atom_interact_list
        interaction_dict[pdb_id][
            'atom_bond_type'] = interact_bond_type_list
        interaction_dict[pdb_id]["aa_idx_list"] = aa_idx_list
        interaction_dict[pdb_id]["residue_bond_type"] = interact_residue_list

        mol_dict = get_mol_dict()
        if lig_name not in mol_dict.keys():
            miss_lig_in_moldict_list.append([pdb_id, lig_name])
            del interaction_dict[pdb_id]
            continue
        mol = mol_dict[lig_name]

        pairwise_mask, pairwise_label = get_pairwise_label(pdb_id, interaction_dict)
        # print(pairwise_mask)
        # print(pairwise_label)

        interaction_dict[pdb_id]["pairwise_mask"] = pairwise_mask
        interaction_dict[pdb_id]["pairwise_label"] = pairwise_label

        model_dataset.append([pdb_id, lig_name, measurement, aff_value, pairwise_label, pairwise_mask])
        # break
        # print(smiles)
        # print(model_dataset)


# input()
### saving data
with open('./out3_1_interaction_dict_0607.pickle', 'wb') as f1:
    pickle.dump(interaction_dict, f1)
f1.close()
with open("./out3_2_model_dataset_0607.pickle", "wb") as f2:
    pickle.dump(model_dataset, f2)
f2.close()

print("S1", len(miss_ligand_in_pdb_list), miss_ligand_in_pdb_list)  # ,file=log_file)
print("S2", len(empty_bond_list), empty_bond_list)  # ,file=log_file)
print("S3", "NONE")
print("S4", len(miss_ideal_pdb_error_list), miss_ideal_pdb_error_list)
print("S5", len(empty_atom_interact_list), empty_atom_interact_list)
print("P2", len(protein_aa_error_list), protein_aa_error_list)
print("P3", len(miss_lig_in_moldict_list), miss_lig_in_moldict_list)
print("P4", len(miss_smiles_list), miss_smiles_list)
print("orignal_dataset", count)
print("Total dataset", len(model_dataset))
