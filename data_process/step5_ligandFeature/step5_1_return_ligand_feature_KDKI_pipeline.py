import os
import numpy as np
from rdkit import Chem
import pickle
from data_process.step5_ligandFeature.getFeatures_ParticalCharge_0608 import save_smiles_dicts


def get_mol_dict():
    mol_dict_path = "../step1/lig_mol_data/mol_dict"
    if os.path.exists(mol_dict_path):
        with open(mol_dict_path, "rb") as f:
            mol_dict = pickle.load(f)
    else:
        mol_dict = {}
        sdf_path = "../step1/lig_mol_data/Components-pub.sdf"
        mols = Chem.SDMolSupplier(sdf_path)
        for m in mols:
            if m is None:
                continue
            name = m.GetProp("_Name")
            mol_dict[name] = m
        with open(mol_dict_path, 'wb') as f:
            pickle.dump(mol_dict, f)
    # print('mol_dict',len(mol_dict))
    return mol_dict


def get_ligand_feature(prex_path,model_dataset,halfNormalizeFlag):
    """
    mol to SMILES,then use SMILES to "save_smiles_dicts"
    """
    feature_filename = prex_path + "out_5_1_ligand_fea_KDKI" +"_halfNormalize"+ str(halfNormalizeFlag) + ".pickle"
    gcn_data_filename = prex_path + "out5_2_Padded_ligand_fea_dict" +"_halfNormalize"+ str(halfNormalizeFlag)+ ".pickle"
    if os.path.isfile(feature_filename):
        feature_dicts = pickle.load(open(feature_filename, "rb"))
        gcn_data_file = pickle.load(open(gcn_data_filename, "rb"))
        print("done")
        return feature_dicts,gcn_data_file
    else:
        model_dataset = np.array(model_dataset)
        lig_list = set(model_dataset[:, 1])
        count = len(lig_list)
        feature_dicts,lig_fea_gcn_Padded = save_smiles_dicts(lig_list, feature_filename,gcn_data_filename,halfNormalizeFlag)
        print("ori:",count,"after:",len(feature_dicts["ligmol_to_atom_mask"].keys()))

    return feature_dicts,lig_fea_gcn_Padded


if __name__ == '__main__' :
    get_ligand_feature()



