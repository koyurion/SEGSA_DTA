import os
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import shutil
import pandas as pd
from rdkit.Chem import AllChem
import math


# 1.1 Screening complexes and small molecules
def func_step1_1(filename):
    pdbid_to_ligand = {}
    count_uncertain_error,count_mer_error,count_or_warning,count_unequal_error =[],[],[],[]
    count_underline_error,count_plus_error = [],[]
    IC50_count = 0
    pdb_set = set()
    ligand_set = set()
    with open(filename) as f:
        for line in f.readlines():
            if line[0] != '#':
                measurement = line.strip().split()[4]
                if "IC50" in measurement:
                    IC50_count += 1
                    continue
                if '<' in measurement or '>' in measurement or '~' in measurement:
                    # print lines[3]
                    count_uncertain_error.append(ligand)
                    continue

                measurement = measurement.split("=")[0]

                ligand = line.strip().split('(')[1].split(')')[0]
                if '-mer' in ligand:
                    count_mer_error.append(ligand)
                    continue
                elif '/' in ligand:
                    count_or_warning.append(ligand)
                    ligand = ligand.split('/')[0]

                if len(ligand) != 3:
                    # print(line[:4], ligand)
                    count_unequal_error.append(ligand)
                    continue
                if len(ligand.strip("_")) != 3:
                    count_underline_error.append(ligand)
                    continue
                if len(ligand.strip("+")) != 3:
                    count_plus_error.append(ligand)
                    continue
                affinity_value = line.strip().split()[3]
                pdbid_to_ligand[line[:4]] = [ligand, measurement, affinity_value]
                pdb_set.add(line[:4])
                ligand_set.add(ligand)
    print('pdbid_to_ligand', len(pdbid_to_ligand))
    print("IC50_count",IC50_count)

    print("count_uncertain_error",len(count_uncertain_error),count_uncertain_error)
    print("count_mer_error", len(count_mer_error), count_mer_error)
    print("count_or_warning", len(count_or_warning), count_or_warning)
    print("count_unequal_error", len(count_unequal_error), count_unequal_error)
    print("count_underline_error", len(count_underline_error), count_underline_error)
    print("count_plus_error", len(count_plus_error), count_plus_error)

    print("pdb_set",len(pdb_set))
    print("ligand_set", len(ligand_set))

    fw1 = open('./output/out1.1_pdbdata_KDKI_list.txt', 'w')
    fw2 = open('./output/out1.2_pdbbind_wget_KDKI_complex.txt', 'w')
    fw3 = open('./output/out1.3_pdbbind_wget_KDKI_ligand.txt', 'w')
    dataset = []
    for pdbid, data in pdbid_to_ligand.items():
        fw1.write((" ").join([pdbid, data[0], data[1], data[2]]) + '\n')
        fw2.write('https://files.rcsb.org/download/' + pdbid + '.pdb\n')
        dataset.append([pdbid, data[0], data[1], data[2]])
    for lig in ligand_set:
        fw3.write('https://files.rcsb.org/ligands/download/' + lig + '_ideal.pdb\n')
    fw1.close()
    fw2.close()
    fw3.close()
    pickle.dump(np.array(dataset),open("./output/out1.4_pdbbind_KDKI_dataset.pickle",'wb'))
    print("step1.1 done")
    return pdbid_to_ligand


'''
pdbid_to_ligand 8728
IC50_count 6555
count_uncertain_error 237 ['SIN', 'SIN', 'DGN', 'DGN', 'NHE', 'MES', '3B1', '3B1', '3HP', '3HP', 'THM', 'THM', 'F12', 'FB2', 'FB2', '3HB', 'ATP', 'FB5', 'FB5', 'FB5', '11H', '11H', '3QD', '3QD', 'FGV', 'GYP', 'ZKD', 'BZ2', 'NAG-MBG', 'NAG-MBG', 'HSX', 'HSX', '6-mer', 'E7E', '5ON', 'FYM', 'ABV', '39O', 'ARU', 'S45', 'S45', 'MS0', 'MS0', '2L2', '39U', '12-mer', '12-mer', '12-mer', '12-mer', '12-mer', '12-mer', '12-mer', '12-mer', '12-mer', 'JW1', 'Q2S', 'ANP', 'B3U', 'NZ9', 'NZ9', 'NZ9', 'NZ9', 'NZ9', 'NZ9', 'NZ9', 'NZ9', 'NZ9', '5IM', '2-mer', 'Q2R', 'Q2R', 'Q2R', '6-mer', '17-mer', 'KLV', 'KOT', '5R8', '313', '10-mer', '1AN', '10-mer', 'ANP', '4NZ', '9-mer', '10P', 'GOX', '10-mer', '11-mer', '18-mer', 'TYR', 'TEP', '1DE', '5F3', '5F3', '5F3', '5TQ', '5TQ', '5TQ', '7MK', '14-mer', '14-mer', '17-mer', '15-mer', 'A4G', 'APF', '147', 'RPI', 'AU8', 'AU8', '5FW', '9-mer', 'ZEN', '13-mer', '322', '8-mer', '10-mer', '10-mer', '5-mer', '5-mer', '13-mer', 'WL3', '9-mer', '57G', 'DUP', '10-mer', 'DCK', '0N9', '10-mer', '10-mer', 'H8H', 'E67', '10-mer', 'EAT', '10-mer', '8-mer', '6NH', '9HS', '2AW', 'SAM', '10-mer', '19-mer', '19-mer', 'I5I', 'I5I', 'DAO', 'C2E', 'MVL', '14-mer', 'FM2', 'NVX', 'PAN', '33V', '33V', 'ATP', '11-mer', '4J0', 'MS8', '4-mer', '4-mer', '128', 'SG4', '11-mer', '18-mer', '18-mer', '0C5', 'NHT', 'FM0', '13-mer', '10-mer', '12-mer', 'GAC', 'NGH', '455', 'P43', 'P43', '2VR', '5L2', 'KU5', 'FH1', 'CJC', 'XNH', 'NTX', '42G', '19X', '6U1', '3C0', '5-mer', 'BAT', 'BAT', '1N1', '1N1', '1N1', '14K', '14K', '14K', '14K', '10O', '3XN', '3XN', '60W', '60W', '60W', '9EY', '9EY', '9EY', '9EY', 'FIN', 'RGJ', 'AWJ', 'AWJ', 'ROC', 'T44', '8-mer', '0QK', 'EUI', '8V8', 'IMH', 'ZX6', 'PPG', '92H', '92H', 'G79', 'G79', '4UX', 'TDI', 'TDI', 'TDI', 'TDI', 'TDI', 'TDI', '6-mer', 'TDI', 'ACR', 'ACR', '2HX', 'K2E', 'CAP']
count_mer_error 2045 ['2-mer', '2-mer', '3-mer', '2-mer', '3-mer', '2-mer', '11-mer', '8-mer', '3-mer', '2-mer', '2-mer', '5-mer', '10-mer', '7-mer', '3-mer', '12-mer', '3-mer', '5-mer', '17-mer', '7-mer', '5-mer', '3-mer', '2-mer', '6-mer', '10-mer', '2-mer', '12-mer', '4-mer', '3-mer', '15-mer', '3-mer', '13-mer', '3-mer', '14-mer', '6-mer', '4-mer', '11-mer', '12-mer', '11-mer', '11-mer', '2-mer', '13-mer', '5-mer', '11-mer', '5-mer', '4-mer', '11-mer', '7-mer', '6-mer', '6-mer', '2-mer', '19-mer', '7-mer', '7-mer', '2-mer', '2-mer', '4-mer', '13-mer', '6-mer', '5-mer', '12-mer', '15-mer', '10-mer', '4-mer', '14-mer', '19-mer', '13-mer', '5-mer', '12-mer', '8-mer', '10-mer', '3-mer', '15-mer', '4-mer', '4-mer', '13-mer', '14-mer', '9-mer', '8-mer', '15-mer', '12-mer', '10-mer', '10-mer', '9-mer', '15-mer', '4-mer', '5-mer', '6-mer', '2-mer', '13-mer', '10-mer', '13-mer', '8-mer', '2-mer', '6-mer', '9-mer', '10-mer', '3-mer', '3-mer', '6-mer', '12-mer', '12-mer', '11-mer', '11-mer', '15-mer', '7-mer', '17-mer', '15-mer', '2-mer', '11-mer', '2-mer', '10-mer', '7-mer', '6-mer', '12-mer', '11-mer', '3-mer', '15-mer', '6-mer', '2-mer', '3-mer', '11-mer', '2-mer', '10-mer', '11-mer', '11-mer', '2-mer', '15-mer', '6-mer', '7-mer', '13-mer', '17-mer', '11-mer', '18-mer', '3-mer', '6-mer', '11-mer', '15-mer', '3-mer', '17-mer', '13-mer', '16-mer', '13-mer', '17-mer', '9-mer', '8-mer', '9-mer', '12-mer', '4-mer', '10-mer', '18-mer', '7-mer', '12-mer', '10-mer', '11-mer', '18-mer', '7-mer', '4-mer', '9-mer', '6-mer', '12-mer', '7-mer', '2-mer', '3-mer', '9-mer', '8-mer', '15-mer', '9-mer', '10-mer', '15-mer', '5-mer', '8-mer', '11-mer', '9-mer', '6-mer', '14-mer', '8-mer', '5-mer', '6-mer', '2-mer', '16-mer', '15-mer', '17-mer', '17-mer', '10-mer', '11-mer', '11-mer', '7-mer', '18-mer', '8-mer', '9-mer', '10-mer', '4-mer', '7-mer', '7-mer', '10-mer', '17-mer', '11-mer', '7-mer', '2-mer', '3-mer', '12-mer', '14-mer', '9-mer', '9-mer', '12-mer', '11-mer', '6-mer', '12-mer', '7-mer', '10-mer', '5-mer', '6-mer', '10-mer', '14-mer', '18-mer', '18-mer', '14-mer', '3-mer', '3-mer', '10-mer', '9-mer', '9-mer', '8-mer', '8-mer', '6-mer', '7-mer', '9-mer', '8-mer', '11-mer', '4-mer', '2-mer', '12-mer', '5-mer', '8-mer', '2-mer', '5-mer', '6-mer', '12-mer', '8-mer', '4-mer', '15-mer', '13-mer', '15-mer', '16-mer', '16-mer', '4-mer', '11-mer', '15-mer', '15-mer', '19-mer', '3-mer', '14-mer', '9-mer', '13-mer', '10-mer', '18-mer', '2-mer', '13-mer', '12-mer', '3-mer', '15-mer', '3-mer', '12-mer', '11-mer', '11-mer', '3-mer', '15-mer', '2-mer', '10-mer', '4-mer', '5-mer', '10-mer', '10-mer', '18-mer', '14-mer', '4-mer', '2-mer', '11-mer', '9-mer', '9-mer', '13-mer', '4-mer', '2-mer', '9-mer', '15-mer', '5-mer', '4-mer', '13-mer', '10-mer', '16-mer', '16-mer', '15-mer', '6-mer', '8-mer', '2-mer', '17-mer', '9-mer', '19-mer', '5-mer', '10-mer', '3-mer', '9-mer', '9-mer', '6-mer', '15-mer', '15-mer', '10-mer', '9-mer', '2-mer', '13-mer', '6-mer', '17-mer', '17-mer', '11-mer', '5-mer', '18-mer', '6-mer', '9-mer', '3-mer', '12-mer', '12-mer', '18-mer', '7-mer', '15-mer', '12-mer', '11-mer', '8-mer', '5-mer', '5-mer', '5-mer', '2-mer', '16-mer', '5-mer', '9-mer', '14-mer', '17-mer', '13-mer', '10-mer', '11-mer', '5-mer', '16-mer', '6-mer', '13-mer', '6-mer', '10-mer', '10-mer', '7-mer', '9-mer', '12-mer', '4-mer', '18-mer', '8-mer', '11-mer', '14-mer', '7-mer', '5-mer', '13-mer', '6-mer', '6-mer', '4-mer', '9-mer', '3-mer', '9-mer', '9-mer', '10-mer', '11-mer', '9-mer', '10-mer', '10-mer', '6-mer', '4-mer', '9-mer', '13-mer', '19-mer', '4-mer', '8-mer', '11-mer', '12-mer', '9-mer', '14-mer', '11-mer', '12-mer', '11-mer', '9-mer', '16-mer', '12-mer', '13-mer', '13-mer', '10-mer', '17-mer', '7-mer', '12-mer', '10-mer', '10-mer', '9-mer', '16-mer', '5-mer', '2-mer', '10-mer', '10-mer', '12-mer', '13-mer', '10-mer', '3-mer', '9-mer', '15-mer', '3-mer', '3-mer', '10-mer', '11-mer', '9-mer', '12-mer', '7-mer', '7-mer', '12-mer', '9-mer', '9-mer', '3-mer', '13-mer', '12-mer', '14-mer', '6-mer', '11-mer', '15-mer', '12-mer', '12-mer', '11-mer', '12-mer', '9-mer', '11-mer', '10-mer', '6-mer', '7-mer', '8-mer', '11-mer', '7-mer', '7-mer', '8-mer', '11-mer', '8-mer', '9-mer', '9-mer', '7-mer', '4-mer', '5-mer', '12-mer', '17-mer', '18-mer', '13-mer', '2-mer', '5-mer', '5-mer', '11-mer', '11-mer', '7-mer', '12-mer', '17-mer', '18-mer', '11-mer', '18-mer', '13-mer', '13-mer', '15-mer', '12-mer', '10-mer', '15-mer', '6-mer', '18-mer', '19-mer', '18-mer', '15-mer', '12-mer', '13-mer', '9-mer', '5-mer', '11-mer', '6-mer', '10-mer', '11-mer', '15-mer', '14-mer', '15-mer', '6-mer', '17-mer', '2-mer', '17-mer', '9-mer', '12-mer', '12-mer', '5-mer', '17-mer', '15-mer', '11-mer', '15-mer', '11-mer', '8-mer', '11-mer', '15-mer', '12-mer', '11-mer', '10-mer', '13-mer', '4-mer', '8-mer', '6-mer', '11-mer', '4-mer', '9-mer', '14-mer', '10-mer', '8-mer', '6-mer', '17-mer', '12-mer', '3-mer', '3-mer', '10-mer', '10-mer', '11-mer', '6-mer', '8-mer', '6-mer', '4-mer', '11-mer', '14-mer', '16-mer', '9-mer', '17-mer', '2-mer', '6-mer', '16-mer', '12-mer', '9-mer', '6-mer', '14-mer', '6-mer', '16-mer', '16-mer', '8-mer', '9-mer', '10-mer', '15-mer', '8-mer', '6-mer', '10-mer', '14-mer', '2-mer', '2-mer', '11-mer', '7-mer', '5-mer', '19-mer', '10-mer', '16-mer', '4-mer', '10-mer', '14-mer', '15-mer', '9-mer', '7-mer', '9-mer', '6-mer', '16-mer', '12-mer', '10-mer', '10-mer', '7-mer', '11-mer', '6-mer', '8-mer', '13-mer', '4-mer', '19-mer', '5-mer', '17-mer', '3-mer', '6-mer', '10-mer', '4-mer', '9-mer', '10-mer', '15-mer', '7-mer', '11-mer', '19-mer', '16-mer', '4-mer', '13-mer', '13-mer', '13-mer', '9-mer', '5-mer', '5-mer', '8-mer', '6-mer', '18-mer', '11-mer', '7-mer', '5-mer', '4-mer', '9-mer', '10-mer', '13-mer', '11-mer', '17-mer', '9-mer', '12-mer', '14-mer', '10-mer', '18-mer', '2-mer', '10-mer', '10-mer', '6-mer', '8-mer', '3-mer', '11-mer', '4-mer', '3-mer', '17-mer', '8-mer', '10-mer', '6-mer', '4-mer', '13-mer', '10-mer', '11-mer', '13-mer', '14-mer', '13-mer', '13-mer', '5-mer', '15-mer', '11-mer', '7-mer', '11-mer', '12-mer', '9-mer', '7-mer', '13-mer', '10-mer', '8-mer', '7-mer', '6-mer', '10-mer', '8-mer', '8-mer', '10-mer', '5-mer', '11-mer', '5-mer', '11-mer', '4-mer', '9-mer', '9-mer', '13-mer', '11-mer', '12-mer', '5-mer', '13-mer', '12-mer', '8-mer', '9-mer', '10-mer', '8-mer', '12-mer', '11-mer', '14-mer', '18-mer', '8-mer', '3-mer', '11-mer', '10-mer', '12-mer', '2-mer', '17-mer', '4-mer', '4-mer', '7-mer', '7-mer', '11-mer', '15-mer', '5-mer', '17-mer', '17-mer', '5-mer', '16-mer', '9-mer', '3-mer', '7-mer', '8-mer', '8-mer', '15-mer', '11-mer', '11-mer', '6-mer', '12-mer', '11-mer', '6-mer', '9-mer', '9-mer', '9-mer', '14-mer', '9-mer', '8-mer', '4-mer', '9-mer', '9-mer', '10-mer', '12-mer', '15-mer', '12-mer', '11-mer', '17-mer', '8-mer', '9-mer', '13-mer', '9-mer', '11-mer', '11-mer', '12-mer', '9-mer', '14-mer', '2-mer', '15-mer', '14-mer', '10-mer', '12-mer', '15-mer', '9-mer', '9-mer', '13-mer', '9-mer', '14-mer', '5-mer', '17-mer', '15-mer', '18-mer', '8-mer', '11-mer', '14-mer', '14-mer', '15-mer', '10-mer', '12-mer', '9-mer', '3-mer', '8-mer', '12-mer', '14-mer', '8-mer', '9-mer', '6-mer', '3-mer', '10-mer', '10-mer', '4-mer', '16-mer', '7-mer', '10-mer', '5-mer', '17-mer', '9-mer', '8-mer', '8-mer', '13-mer', '11-mer', '5-mer', '3-mer', '16-mer', '16-mer', '14-mer', '11-mer', '9-mer', '16-mer', '12-mer', '12-mer', '8-mer', '13-mer', '13-mer', '3-mer', '6-mer', '8-mer', '2-mer', '16-mer', '16-mer', '11-mer', '10-mer', '9-mer', '9-mer', '11-mer', '7-mer', '12-mer', '13-mer', '12-mer', '14-mer', '14-mer', '5-mer', '17-mer', '12-mer', '11-mer', '16-mer', '13-mer', '9-mer', '9-mer', '13-mer', '3-mer', '3-mer', '10-mer', '16-mer', '3-mer', '13-mer', '14-mer', '4-mer', '13-mer', '12-mer', '13-mer', '18-mer', '15-mer', '16-mer', '10-mer', '15-mer', '10-mer', '12-mer', '8-mer', '7-mer', '15-mer', '2-mer', '10-mer', '13-mer', '13-mer', '17-mer', '11-mer', '12-mer', '18-mer', '10-mer', '3-mer', '11-mer', '19-mer', '9-mer', '10-mer', '11-mer', '15-mer', '15-mer', '5-mer', '13-mer', '15-mer', '13-mer', '8-mer', '10-mer', '13-mer', '8-mer', '18-mer', '10-mer', '15-mer', '11-mer', '4-mer', '17-mer', '8-mer', '10-mer', '4-mer', '9-mer', '5-mer', '6-mer', '4-mer', '18-mer', '9-mer', '4-mer', '4-mer', '11-mer', '14-mer', '15-mer', '11-mer', '12-mer', '16-mer', '4-mer', '15-mer', '14-mer', '17-mer', '6-mer', '13-mer', '5-mer', '13-mer', '13-mer', '13-mer', '8-mer', '11-mer', '8-mer', '10-mer', '9-mer', '9-mer', '16-mer', '16-mer', '9-mer', '4-mer', '17-mer', '2-mer', '18-mer', '19-mer', '3-mer', '10-mer', '12-mer', '12-mer', '10-mer', '4-mer', '2-mer', '15-mer', '13-mer', '17-mer', '14-mer', '11-mer', '6-mer', '17-mer', '6-mer', '7-mer', '12-mer', '5-mer', '5-mer', '12-mer', '10-mer', '20-mer', '8-mer', '3-mer', '12-mer', '10-mer', '12-mer', '9-mer', '6-mer', '6-mer', '13-mer', '15-mer', '9-mer', '15-mer', '12-mer', '11-mer', '6-mer', '13-mer', '12-mer', '19-mer', '8-mer', '13-mer', '10-mer', '11-mer', '12-mer', '8-mer', '15-mer', '6-mer', '5-mer', '11-mer', '18-mer', '10-mer', '6-mer', '16-mer', '17-mer', '14-mer', '9-mer', '8-mer', '8-mer', '11-mer', '13-mer', '3-mer', '12-mer', '12-mer', '12-mer', '10-mer', '12-mer', '2-mer', '10-mer', '2-mer', '4-mer', '17-mer', '15-mer', '15-mer', '15-mer', '8-mer', '17-mer', '11-mer', '8-mer', '15-mer', '8-mer', '3-mer', '12-mer', '8-mer', '11-mer', '15-mer', '5-mer', '11-mer', '8-mer', '15-mer', '5-mer', '4-mer', '4-mer', '16-mer', '15-mer', '12-mer', '16-mer', '11-mer', '4-mer', '8-mer', '10-mer', '9-mer', '13-mer', '16-mer', '6-mer', '5-mer', '15-mer', '4-mer', '11-mer', '8-mer', '7-mer', '15-mer', '14-mer', '10-mer', '10-mer', '15-mer', '9-mer', '14-mer', '11-mer', '7-mer', '11-mer', '11-mer', '11-mer', '11-mer', '11-mer', '19-mer', '10-mer', '13-mer', '9-mer', '8-mer', '16-mer', '12-mer', '15-mer', '9-mer', '18-mer', '10-mer', '9-mer', '8-mer', '11-mer', '14-mer', '12-mer', '6-mer', '9-mer', '9-mer', '12-mer', '10-mer', '7-mer', '11-mer', '8-mer', '10-mer', '8-mer', '15-mer', '10-mer', '5-mer', '14-mer', '10-mer', '18-mer', '8-mer', '6-mer', '15-mer', '15-mer', '15-mer', '9-mer', '12-mer', '6-mer', '4-mer', '11-mer', '15-mer', '9-mer', '9-mer', '17-mer', '13-mer', '13-mer', '4-mer', '12-mer', '16-mer', '6-mer', '17-mer', '15-mer', '9-mer', '12-mer', '13-mer', '18-mer', '6-mer', '7-mer', '9-mer', '11-mer', '14-mer', '12-mer', '11-mer', '3-mer', '4-mer', '12-mer', '10-mer', '13-mer', '19-mer', '7-mer', '4-mer', '3-mer', '13-mer', '16-mer', '11-mer', '8-mer', '19-mer', '4-mer', '18-mer', '7-mer', '6-mer', '11-mer', '12-mer', '16-mer', '11-mer', '16-mer', '16-mer', '18-mer', '19-mer', '4-mer', '14-mer', '5-mer', '2-mer', '11-mer', '14-mer', '5-mer', '7-mer', '12-mer', '19-mer', '19-mer', '13-mer', '2-mer', '6-mer', '4-mer', '12-mer', '13-mer', '12-mer', '13-mer', '6-mer', '4-mer', '12-mer', '3-mer', '9-mer', '11-mer', '12-mer', '6-mer', '10-mer', '6-mer', '9-mer', '11-mer', '3-mer', '19-mer', '9-mer', '3-mer', '16-mer', '6-mer', '5-mer', '15-mer', '15-mer', '11-mer', '7-mer', '19-mer', '10-mer', '15-mer', '13-mer', '9-mer', '15-mer', '16-mer', '9-mer', '7-mer', '6-mer', '12-mer', '8-mer', '16-mer', '3-mer', '12-mer', '3-mer', '13-mer', '9-mer', '10-mer', '2-mer', '18-mer', '13-mer', '6-mer', '15-mer', '15-mer', '16-mer', '12-mer', '3-mer', '5-mer', '15-mer', '7-mer', '12-mer', '10-mer', '15-mer', '5-mer', '4-mer', '11-mer', '13-mer', '7-mer', '5-mer', '19-mer', '14-mer', '5-mer', '13-mer', '18-mer', '11-mer', '11-mer', '12-mer', '18-mer', '9-mer', '14-mer', '14-mer', '4-mer', '16-mer', '7-mer', '5-mer', '14-mer', '9-mer', '10-mer', '9-mer', '11-mer', '19-mer', '6-mer', '7-mer', '16-mer', '16-mer', '13-mer', '15-mer', '6-mer', '3-mer', '10-mer', '11-mer', '5-mer', '12-mer', '14-mer', '15-mer', '16-mer', '2-mer', '19-mer', '13-mer', '14-mer', '14-mer', '12-mer', '10-mer', '18-mer', '11-mer', '10-mer', '18-mer', '8-mer', '8-mer', '3-mer', '11-mer', '13-mer', '8-mer', '10-mer', '11-mer', '5-mer', '16-mer', '15-mer', '3-mer', '9-mer', '18-mer', '9-mer', '5-mer', '14-mer', '12-mer', '12-mer', '5-mer', '9-mer', '5-mer', '4-mer', '4-mer', '6-mer', '9-mer', '13-mer', '10-mer', '9-mer', '10-mer', '9-mer', '14-mer', '7-mer', '3-mer', '13-mer', '5-mer', '9-mer', '4-mer', '13-mer', '10-mer', '11-mer', '11-mer', '19-mer', '16-mer', '18-mer', '17-mer', '15-mer', '6-mer', '9-mer', '9-mer', '7-mer', '4-mer', '14-mer', '15-mer', '5-mer', '8-mer', '18-mer', '15-mer', '8-mer', '5-mer', '19-mer', '15-mer', '8-mer', '11-mer', '15-mer', '11-mer', '8-mer', '3-mer', '13-mer', '11-mer', '6-mer', '5-mer', '11-mer', '15-mer', '9-mer', '12-mer', '12-mer', '5-mer', '12-mer', '13-mer', '4-mer', '3-mer', '15-mer', '8-mer', '12-mer', '11-mer', '19-mer', '13-mer', '4-mer', '12-mer', '13-mer', '11-mer', '5-mer', '5-mer', '11-mer', '6-mer', '11-mer', '19-mer', '9-mer', '5-mer', '13-mer', '10-mer', '11-mer', '14-mer', '13-mer', '6-mer', '3-mer', '7-mer', '13-mer', '12-mer', '13-mer', '10-mer', '9-mer', '9-mer', '12-mer', '11-mer', '11-mer', '11-mer', '14-mer', '15-mer', '3-mer', '19-mer', '9-mer', '12-mer', '15-mer', '11-mer', '14-mer', '5-mer', '9-mer', '11-mer', '12-mer', '7-mer', '6-mer', '4-mer', '6-mer', '5-mer', '5-mer', '15-mer', '15-mer', '9-mer', '13-mer', '12-mer', '14-mer', '8-mer', '8-mer', '11-mer', '8-mer', '9-mer', '4-mer', '2-mer', '19-mer', '4-mer', '3-mer', '3-mer', '3-mer', '8-mer', '16-mer', '12-mer', '17-mer', '10-mer', '11-mer', '10-mer', '15-mer', '10-mer', '14-mer', '13-mer', '15-mer', '9-mer', '9-mer', '10-mer', '16-mer', '12-mer', '10-mer', '2-mer', '11-mer', '13-mer', '9-mer', '13-mer', '5-mer', '12-mer', '14-mer', '16-mer', '12-mer', '5-mer', '11-mer', '4-mer', '16-mer', '18-mer', '15-mer', '10-mer', '10-mer', '8-mer', '7-mer', '14-mer', '13-mer', '14-mer', '8-mer', '10-mer', '12-mer', '19-mer', '11-mer', '8-mer', '9-mer', '3-mer', '11-mer', '10-mer', '15-mer', '10-mer', '11-mer', '15-mer', '13-mer', '12-mer', '10-mer', '15-mer', '17-mer', '13-mer', '15-mer', '16-mer', '17-mer', '9-mer', '14-mer', '11-mer', '3-mer', '9-mer', '10-mer', '10-mer', '9-mer', '15-mer', '6-mer', '7-mer', '13-mer', '17-mer', '9-mer', '3-mer', '6-mer', '12-mer', '11-mer', '15-mer', '8-mer', '5-mer', '5-mer', '5-mer', '2-mer', '3-mer', '5-mer', '12-mer', '10-mer', '12-mer', '10-mer', '11-mer', '12-mer', '3-mer', '2-mer', '8-mer', '11-mer', '15-mer', '8-mer', '10-mer', '4-mer', '10-mer', '12-mer', '12-mer', '14-mer', '5-mer', '5-mer', '12-mer', '11-mer', '14-mer', '11-mer', '12-mer', '10-mer', '18-mer', '5-mer', '2-mer', '3-mer', '15-mer', '4-mer', '3-mer', '3-mer', '12-mer', '5-mer', '9-mer', '18-mer', '3-mer', '3-mer', '19-mer', '16-mer', '8-mer', '3-mer', '14-mer', '16-mer', '2-mer', '19-mer', '18-mer', '18-mer', '8-mer', '8-mer', '10-mer', '16-mer', '6-mer', '2-mer', '5-mer', '18-mer', '19-mer', '15-mer', '17-mer', '17-mer', '19-mer', '12-mer', '8-mer', '12-mer', '16-mer', '7-mer', '14-mer', '18-mer', '12-mer', '5-mer', '16-mer', '17-mer', '5-mer', '3-mer', '17-mer', '11-mer', '11-mer', '10-mer', '3-mer', '9-mer', '14-mer', '7-mer', '2-mer', '12-mer', '3-mer', '15-mer', '15-mer', '3-mer', '10-mer', '8-mer', '9-mer', '13-mer', '17-mer', '12-mer', '11-mer', '4-mer', '13-mer', '9-mer', '15-mer', '7-mer', '5-mer', '12-mer', '8-mer', '10-mer', '10-mer', '14-mer', '17-mer', '3-mer', '2-mer', '3-mer', '10-mer', '6-mer', '8-mer', '9-mer', '12-mer', '7-mer', '15-mer', '11-mer', '19-mer', '13-mer', '14-mer', '4-mer', '9-mer', '16-mer', '13-mer', '6-mer', '6-mer', '16-mer', '9-mer', '16-mer', '7-mer', '17-mer', '17-mer', '14-mer', '14-mer', '4-mer', '16-mer', '2-mer', '8-mer', '9-mer', '13-mer', '12-mer', '3-mer', '11-mer', '16-mer', '15-mer', '4-mer', '2-mer', '6-mer', '11-mer', '4-mer', '5-mer', '3-mer', '11-mer', '6-mer', '7-mer', '3-mer', '9-mer', '13-mer', '6-mer', '18-mer', '4-mer', '14-mer', '18-mer', '13-mer', '10-mer', '13-mer', '3-mer', '9-mer', '3-mer', '5-mer', '15-mer', '4-mer', '12-mer', '15-mer', '14-mer', '3-mer', '3-mer', '9-mer', '10-mer', '14-mer', '9-mer', '16-mer', '13-mer', '15-mer', '4-mer', '11-mer', '7-mer', '5-mer', '8-mer', '9-mer', '15-mer', '14-mer', '11-mer', '12-mer', '11-mer', '4-mer', '6-mer', '12-mer', '15-mer', '4-mer', '10-mer', '6-mer', '5-mer', '10-mer', '9-mer', '5-mer', '19-mer', '19-mer', '16-mer', '18-mer', '19-mer', '12-mer', '11-mer', '13-mer', '11-mer', '6-mer', '3-mer', '12-mer', '7-mer', '8-mer', '6-mer', '7-mer', '10-mer', '3-mer', '3-mer', '8-mer', '3-mer', '7-mer', '6-mer', '16-mer', '15-mer', '15-mer', '14-mer', '15-mer', '11-mer', '3-mer', '12-mer', '12-mer', '12-mer', '15-mer', '17-mer', '13-mer', '19-mer', '15-mer', '10-mer', '19-mer', '5-mer', '5-mer', '7-mer', '10-mer', '14-mer', '5-mer', '3-mer', '19-mer', '14-mer', '9-mer', '3-mer', '3-mer', '13-mer', '10-mer', '13-mer', '8-mer', '7-mer', '19-mer', '13-mer', '10-mer', '11-mer', '8-mer', '13-mer', '5-mer', '14-mer', '3-mer', '8-mer', '4-mer', '15-mer', '15-mer', '13-mer', '11-mer', '9-mer', '3-mer', '10-mer', '19-mer', '11-mer', '16-mer', '2-mer', '17-mer', '13-mer', '9-mer', '9-mer', '12-mer', '14-mer', '17-mer', '5-mer', '15-mer', '18-mer', '8-mer', '16-mer', '12-mer', '18-mer', '10-mer', '8-mer', '16-mer', '13-mer', '14-mer', '8-mer', '14-mer', '16-mer', '9-mer', '13-mer', '18-mer', '16-mer', '11-mer', '13-mer', '12-mer', '12-mer', '18-mer', '2-mer', '9-mer', '12-mer', '18-mer', '3-mer', '9-mer', '7-mer', '2-mer', '8-mer', '9-mer', '19-mer', '17-mer', '14-mer', '12-mer', '7-mer', '15-mer', '2-mer', '19-mer', '10-mer', '6-mer', '8-mer', '15-mer', '16-mer', '16-mer', '10-mer', '16-mer', '6-mer', '16-mer', '11-mer', '16-mer', '15-mer', '8-mer', '14-mer', '4-mer', '9-mer', '9-mer', '11-mer', '12-mer', '16-mer', '13-mer', '16-mer', '9-mer', '19-mer', '14-mer', '3-mer', '11-mer', '7-mer', '7-mer', '4-mer', '11-mer', '8-mer', '3-mer', '11-mer', '9-mer', '11-mer', '11-mer', '17-mer', '2-mer', '4-mer', '5-mer', '11-mer', '5-mer', '9-mer', '7-mer', '3-mer', '16-mer', '19-mer', '8-mer', '11-mer', '12-mer', '5-mer', '5-mer', '5-mer', '2-mer', '10-mer', '12-mer', '16-mer', '19-mer', '3-mer', '9-mer', '9-mer', '13-mer', '12-mer', '6-mer', '10-mer', '19-mer', '19-mer', '13-mer', '7-mer', '8-mer', '5-mer', '12-mer', '12-mer', '16-mer', '15-mer', '9-mer', '11-mer', '8-mer', '16-mer', '16-mer', '5-mer', '5-mer', '6-mer', '13-mer', '6-mer', '12-mer', '16-mer', '16-mer', '12-mer', '11-mer', '15-mer', '18-mer', '15-mer', '15-mer', '4-mer', '5-mer', '11-mer', '8-mer', '19-mer', '19-mer', '10-mer', '13-mer', '15-mer', '6-mer', '3-mer', '9-mer', '4-mer', '11-mer', '2-mer', '4-mer', '14-mer', '5-mer', '18-mer', '14-mer', '18-mer', '16-mer', '12-mer', '10-mer', '15-mer', '11-mer', '12-mer', '7-mer', '8-mer', '4-mer', '8-mer', '10-mer', '7-mer', '17-mer', '16-mer', '8-mer', '5-mer', '15-mer', '13-mer', '5-mer', '4-mer', '17-mer', '2-mer', '16-mer', '3-mer', '17-mer', '15-mer', '8-mer', '5-mer', '7-mer', '17-mer', '13-mer', '5-mer', '12-mer', '8-mer', '5-mer', '5-mer', '7-mer', '10-mer', '8-mer', '5-mer', '10-mer', '5-mer', '11-mer', '4-mer', '19-mer', '10-mer', '8-mer', '16-mer', '5-mer', '5-mer', '2-mer', '12-mer', '3-mer', '8-mer', '8-mer', '5-mer', '10-mer', '9-mer', '10-mer', '13-mer', '6-mer', '12-mer', '12-mer', '8-mer', '4-mer', '11-mer', '12-mer', '5-mer', '18-mer', '7-mer', '6-mer', '11-mer', '6-mer', '11-mer', '9-mer', '5-mer', '6-mer', '19-mer', '14-mer', '8-mer', '14-mer', '11-mer', '11-mer', '17-mer', '4-mer', '6-mer', '10-mer', '6-mer', '11-mer', '10-mer', '14-mer', '5-mer', '5-mer', '7-mer', '6-mer', '5-mer', '6-mer', '4-mer', '15-mer', '14-mer']
count_or_warning 20 ['RVA/VAE', 'RVC/VCE', 'FCA/FCB', 'HBS/HBR', '5ND/5NL', 'FCA/FCB', 'P59/P69', 'ARA/ARB', 'FCA/FCB', 'GLA/GLB', 'ARA/ARB', 'DQ6/DQ7', 'GLA/GLB', 'GLA/GLB', '32Q/32R', '53I/53J', '52I/52J', '2Y7/2Y8', '34R/34S', 'A2G/NGA']
count_unequal_error 106 ['BR', 'SIA-GAL', 'GLC-FRU', 'MA2-MA3', 'SIA-GAL', 'SIA-GAL', 'PC', 'SGA-BGC', 'NAG-MBG', 'SIA-MAG-GAL-FUC', 'NOJ-SO4', 'GLC-GAL', 'NAG-BDP', 'U', 'FUC-LAT', 'ET', 'U', 'GAL-GLC', 'Ca2+', '98', 'NAG-GAL', '1P0-SIA', 'MYI ', '1P4-SIA', 'GLC-NOJ', 'GAL-NGA', 'JKE&EMC', 'BGC-OXZ', 'GLA-MBG', 'FMN hq', 'FMN hq', 'PC', 'GLC-DMJ', 'PC', 'SGN-IDY', 'PC', 'GAL-MHD', 'DH6-MMA', 'GNS-IDY', 'SGN-IDY', 'ET', 'AJD&AKD', 'SUS-IDY', 'SUS-IDY', 'XDL-XYP', 'XDN-XYP', 'XDN-XYP', 'IFM-BGC', 'GAL-LRD', 'CP', 'GNX-IDY', 'GLC-GLC', 'NGA-AMG', 'F1A-MMA', 'EDG-AHR', 'ZN', 'ET', 'GNX-IDY', 'GAL-MGC', 'MNT-BEF', 'ET', 'ALX-BNX', 'MAN-IFM', 'ET', 'NOY-BGC', 'SIA-NAG-GAL', 'A', 'IFM-SO4', 'FMN hq', '45T-3QE', 'GLC-IFM', 'SLE-TYR', 'FMN hq', '035-036', 'SIA-GAL', 'KDA&KDO', 'KDA&KDO', 'MG5&SG5', 'UDP-BGC', 'KDA&KDO', 'T3', 'TMX-CTO', '1N', 'KDA&KDO', 'XIF-XYP', 'XIF-XYP', 'ZN', 'SER-DNF', 'AI', 'GDP-7MG', 'GDP-7MG', 'U', '3CU+GLC', 'FMN ox', 'FMN ox', 'FMN ox', 'F6Y-T44', 'FMN ox', 'FMN ox', 'FMN ox', 'FMN ox', 'FMN sq', 'FMN ox', '9D9&9DG', 'T3', 'FMN sq']
count_underline_error 6 ['__U', '__N', '_MC', '_VX', '_VX', '_T3']
count_plus_error 2 ['Na+', 'MN+']
pdb_set 8728
ligand_set 6198
step1.1 done
    Total ERROR: 237+2045+106+6+2 = 2396
'''
# filename = './INDEX_PDBBind/INDEX_general_PL_data.2019'
# pdbid_to_ligand_data = func_step1_1(filename)


# 1.2 Check data volume and completeness
# finished by S3: check4_dataVolume_dataEnd.py
def func_step1_2():
    '''
    del: miss_ligand and empty ideal.pdb,and get dataset
    :return:
    '''
    dataset = pickle.load(open("./output/out1.4_pdbbind_KDKI_dataset.pickle",'rb'))
    ligands_path = "step1_screen/ligands/"
    miss_ligand_11 = ["ARY","8VQ","MGU","72V","UNF","XLM","MFS","5HV","CBS","2SD","KTS"]
    assert len(miss_ligand_11) == 11
    new_dataset = []
    for i in range(dataset.shape[0]):
        ligand = dataset[i,1]
        if ligand not in miss_ligand_11:
            if os.path.getsize(ligands_path + ligand + '_ideal.pdb') != 0:
                new_dataset.append(dataset[i,:])
    new_dataset = np.array(new_dataset)
    pickle.dump(new_dataset,open("./output/func1_2_dataset_withoutEmptyIdealPdb.pickle","wb"))
    print(new_dataset.shape[0])


'''
new_dataset.shape[0] : 8671
ligand: 6151
'''
# func_step1_2()


# 1.3 Resolution screening of complexes
def func_step1_3():
    pdb_path = "step1_screen/pdbs/"  # 8736
    # pdbs = os.listdir(pdb_path)

    pro_include_pdb = []
    exclude_NOT_pdb = []
    exclude_BIGGER_pdb = []

    count = 0
    dataset = pickle.load(open("./output/func1_2_dataset_withoutEmptyIdealPdb.pickle", 'rb'))
    new_dataset = []
    for i in range(dataset.shape[0]):
        pdb = dataset[i,0]
        pdb_file = pdb_path + pdb + ".pdb"
        lines = open(pdb_file,"r").readlines()
        for line in lines:
            if line.startswith("REMARK   2 RESOLUTION. "):
                count += 1
                flag = line.split()[3]
                if flag == "NOT":
                    exclude_NOT_pdb.append(pdb)
                else:
                    if float(flag) <= 2.5:
                        pro_include_pdb.append(pdb)
                        new_dataset.append(dataset[i,:])
                    else:
                        exclude_BIGGER_pdb.append(pdb)
                break

    print("include_pdb", len(pro_include_pdb))
    print("exclude_NOT_pdb", len(exclude_NOT_pdb))
    print("exclude_BIGGER_pdb", len(exclude_BIGGER_pdb))
    pickle.dump(pro_include_pdb,open("./output/func1_3_include_KDKI_complexPdb.pickle","wb"))
    new_dataset = np.array(new_dataset)
    pickle.dump(new_dataset, open("./output/func1_3_dataset_withoutResolution25.pickle", "wb"))
    print("func1_3: done")


'''
        include_pdb 7261
        exclude_NOT_pdb 80
        exclude_BIGGER_pdb 1330
        func1_3: done
        total : 7261 + 80 + 1330 = 8671
'''
# func_step1_3()


# 1.4 Molecular weight screening of small molecules
def func_step1_4():

    ligand_include = []
    ligand_exclude = []
    empty_lig = []
    dataset = pickle.load(open("./output/func1_3_dataset_withoutResolution25.pickle","rb"))

    if os.path.exists('./lig_mol_data/mol_dict'):
        with open('./lig_mol_data/mol_dict', "rb") as f:
            mol_dict = pickle.load(f)
    new_dataset = []
    for i in range(dataset.shape[0]):
        lig =  dataset[i,1]
        try:
            mol = mol_dict[lig]
        except:
            empty_lig.append(lig)
            continue
        if Descriptors.ExactMolWt(mol) <= 500:
            ligand_include.append(lig)
            new_dataset.append(dataset[i,:])
        else:
            ligand_exclude.append(lig)
        if Descriptors.ExactMolWt(mol) == 500:
            print(lig)
    new_dataset = np.array(new_dataset)
    pickle.dump(new_dataset,open("./output/func1_4_dataset_withoutMolWt500.pickle",'wb'))
    print("ligand_include", len(ligand_include))
    print("ligand_exclude", len(ligand_exclude))
    print("empty_lig", len(empty_lig))
    print("new_dataset",new_dataset.shape)
'''
ligand_include 5720
ligand_exclude 1527
empty_lig 14
new_dataset (5720, 4)
'''

# func_step1_4()


# 1.5 Screening of charge properties of small molecules
def func_step1_5():
    dataset = pickle.load(open("./output/func1_4_dataset_withoutMolWt500.pickle",'rb'))
    if os.path.exists('./lig_mol_data/mol_dict_0501.pickle'):
        with open('./lig_mol_data/mol_dict_0501.pickle', "rb") as f:
            mol_dict = pickle.load(f)
    new_dataset = []
    nan_charge_list = []
    inf_charge_list = []
    for i in range(dataset.shape[0]):
        lig = dataset[i,1]

        mol = mol_dict[lig]
        AllChem.ComputeGasteigerCharges(mol)
        Chem.AssignStereochemistry(mol, flagPossibleStereoCenters=True)
        nan_flag = False
        for atom in mol.GetAtoms():
            G_charge = atom.GetDoubleProp('_GasteigerCharge')
            G_Hcharge = atom.GetDoubleProp('_GasteigerHCharge')
            # if G_charge == np.nan or G_Hcharge == np.nan: # not work
            # if G_charge != G_charge or G_Hcharge != G_Hcharge:
            if math.isnan(G_charge) or math.isnan(G_Hcharge):
                print(lig,G_charge, G_Hcharge)
                nan_charge_list.append(lig)
                nan_flag = True
                break
            if math.isinf(G_charge) or math.isinf(G_Hcharge):
                print(lig,G_charge,G_Hcharge)
                inf_charge_list.append(lig)
                nan_flag = True
                break
        if not nan_flag:
            new_dataset.append(dataset[i,:])
    new_dataset = np.array(new_dataset)
    pickle.dump(new_dataset,open("./output/func1_5_dataset_withoutNanCharge.pickle","wb"))
    print("nan_charge_list",len(set(nan_charge_list)),set(nan_charge_list))
    print("inf_charge_list", len(set(inf_charge_list)), set(inf_charge_list))
    print("new_dataset",new_dataset.shape)
'''
nan_charge_list 22 {'B9F', 'CFQ', 'UVC', 'SFU', 'EZ1', 'VO3', '498', '6YR', '84A', 'GXE', 'FQY', '6ZJ', 'B1R', 'B30', '1PT', '670', '3Q7', 'HGB', '1Y8', 'FQV', 'H57', '067'}
inf_charge_list 3 {'H58', 'B70', '830'}
new_dataset (5693, 4)
'''

# func_step1_5()

