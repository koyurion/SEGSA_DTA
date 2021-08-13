import os
import pickle
from Bio.PDB import PDBParser,Select,PDBIO

ori_pdbs_path = "F:/LLL-doc/new_data_0501/pdbs/"

dataset_path = r"E:\0430\code_DTA_0430\step2\output\dataset_afterInterOut_5489.pickle"
ori_dataset = pickle.load(open(dataset_path,"rb"))

def check_altLoc_HOHs():
    altLoc_HOHs_pdbs = []
    for i in range(ori_dataset.shape[0]):
        pdb_name = ori_dataset[i,0]
        altLoc_HOHs = [line for line in open(ori_pdbs_path + pdb_name + ".pdb").readlines() \
                    if line.startswith("HETATM") and (line[17:20] == "HOH") and (line[16] != " ")]
        if len(altLoc_HOHs) != 0:
            altLoc_HOHs_pdbs.append(pdb_name)
    print("altLoc_HOHs_pdbs",len(altLoc_HOHs_pdbs),altLoc_HOHs_pdbs)

# check_altLoc_HOHs()
"""
altLoc_HOHs_pdbs 492 ['5mnc', '4x8s', '2hds', '4x8t', '4y3j', '4aci', '5m9w', '1ew8', '2hdu', '4y3y', '5fsc', '4y38', '4x8u', '4y4j', '4ze6', '5mn1', '5mnb', '4kz7', '5lvd', '4ndu', '4kz3', '4ykk', '4kz8', '6mvu', '3c2u', '4i7k', '4guj', '4kz6', '2ya7', '1ew9', '2ri9', '4n5d', '3d52', '4i7j', '4wyp', '3w07', '4lhm', '5aqv', '5ma7', '3nyd', '1c5o', '4i7p', '5mby', '3ff3', '3dx1', '3sk2', '3ex6', '5kh7', '5nzn', '4kza', '4kov', '4w52', '1mpl', '4klv', '6f5w', '5mnn', '5mno', '5dhp', '5g4m', '3wjw', '2qw1', '3fgc', '2flh', '4i7l', '4w53', '4umb', '5g4n', '5wcm', '1c5z', '5nzf', '1gj4', '1c5t', '4kxb', '4whr', '4whs', '3ozt', '4i7m', '1m2x', '3c8e', '3d51', '4kz4', '4w54', '3sv2', '1c5y', '1ghw', '1n3z', '5kh3', '6gnm', '3ijg', '1gx4', '5l4j', '5mfr', '4kow', '1ghv', '3feg', '4ykj', '3u81', '3jqf', '4gki', '4cu8', '4bvb', '5yto', '1gi7', '2zz1', '2zz2', '3rr4', '5oss', '6gnp', '4yrd', '3t2c', '4dfb', '3lgs', '5ulp', '3u90', '5n34', '2z3h', '4psb', '3ddf', '4gkh', '5jsq', '1c5p', '5mng', '5mnh', '1c5n', '3nkk', '5lvn', '4w55', '3d50', '4kox', '5mkr', '5mks', '1ghz', '4aq6', '1o2p', '4w57', '4q0k', '6apu', '3d4z', '4cd1', '1km3', '5ytu', '1gj5', '1o5g', '2vjx', '6gnw', '1o35', '3smq', '4io5', '5n3y', '1gi8', '4rqv', '4knz', '4xx9', '5n2t', '5fhm', '6ce6', '3ozr', '6b95', '6ar9', '2qdt', '4a6s', '1o2r', '4few', '5oje', '1gjd', '1o5c', '3iof', '3qxt', '3qxh', '4y59', '5b5f', '5b5g', '2vcb', '3bbf', '4e70', '5ect', '2oyk', '6d2o', '5n2z', '5g1z', '3ozs', '5l4m', '2gkl', '6afe', '6cb5', '2f35', '5o1i', '6aff', '1gja', '4fev', '6afd', '1h0a', '3u10', '5tp0', '5uxm', '6gnr', '4kot', '5mnr', '1gi1', '1o2s', '4qb3', '2cbv', '6grp', '3bmq', '4w9f', '6af9', '6gr7', '3iog', '5lpm', '4tkb', '3qce', '3nes', '6gon', '6goo', '2vr4', '3oil', '3d4y', '4bs0', '4l2l', '3wvm', '1o5e', '2b1v', '2pgz', '4tkh', '2ya8', '1o5b', '5n31', '6eog', '4rqk', '5dw2', '1o2u', '1o2v', '1o2y', '1qbn', '4azc', '3h1x', '5l4f', '5lvl', '4y5d', '1o36', '5hv0', '4tjz', '4yx4', '4az5', '1c5s', '1hn2', '2izl', '3ddg', '5f1r', '6f90', '6afc', '4tkj', '4kqp', '5n0f', '4io3', '5n2x', '6ge7', '1o5f', '1o2n', '6eaa', '6eed', '2flr', '5nai', '1o2z', '6b4d', '1o2g', '6afh', '4zx1', '3oik', '4aq4', '4wrb', '6afi', '4ymb', '5ttf', '1gi6', '3okv', '5wa8', '5lwd', '5nz4', '6afj', '2vrj', '4d1j', '5meh', '4zvi', '5nu5', '1gjb', '1c5q', '1o2o', '3oku', '4yyt', '5nya', '5nze', '6csq', '6gf9', '5ihh', '5jxn', '4fmu', '5ny1', '5jss', '5mfs', '4yxi', '3cj4', '5jvi', '5t8o', '3dx3', '4qme', '1o3l', '6gfs', '6ea2', '3sax', '5l4i', '3cwk', '5jt9', '5nu3', '1o3p', '1pxp', '5nxi', '6aps', '5fho', '6b1k', '6ee6', '1c5x', '5fs5', '6ee4', '4io2', '6ghh', '4i73', '4i74', '5ng9', '4r5b', '5js3', '3oim', '4f9y', '1o30', '1o3j', '3fvn', '5elv', '1o37', '1o38', '3dx2', '3s9t', '6csr', '5nih', '5tci', '2f34', '4g4p', '3cow', '1o2j', '1o2k', '2g78', '2g79', '4f9w', '6apv', '1gj6', '3m2u', '3mhm', '4ayr', '4r3c', '4riv', '5edb', '5sz6', '6afl', '3igp', '4yxo', '1xug', '2ate', '5hz5', '2hnx', '3dx4', '1o39', '1o3d', '1pxn', '5u28', '1gi4', '2g8n', '3mz6', '2psv', '4yxu', '5nxw', '5hvt', '4itp', '4ryl', '5i7x', '5i7y', '4iwz', '1o3h', '3t0w', '4riu', '4ayq', '5ttg', '5hdz', '5tcj', '2hzy', '4na9', '5ne5', '1qb9', '3dx0', '2g71', '3oe4', '5mft', '5ny3', '3u92', '3p58', '1c1r', '1c1v', '5el9', '4e5w', '5edc', '1o2q', '2yln', '3th9', '5sz5', '1qbo', '4i71', '5iu8', '4z2b', '5hz6', '1o2t', '5ny6', '1gj7', '6en5', '6en6', '1o3e', '4yt6', '5nxp', '3hs4', '4zx0', '4rfd', '4his', '1ghy', '4z1j', '4pop', '3odu', '4buq', '4q07', '5u4x', '4hiq', '4x5p', '5k8s', '4z1n', '4z1e', '1c1u', '6gzm', '3tt4', '5nxo', '5eef', '4z1k', '2vc9', '4css', '5nxv', '3b28', '4m2r', '3t3v', '4xoc', '6csp', '3ejp', '5nxg', '3ejq', '3ejr', '3v7x', '3t3u', '5sz3', '1pxo', '2fr3', '4r5a', '4zwz', '5sz2', '6b1f', '4cst', '4pxm', '5sz1', '4xo8', '5iu7', '4x5q', '5ef7', '5m7s', '5eek', '5sz0', '5eei', '3nzk', '5iu4', '3rbu', '3pr0', '6b8y', '5sz4', '5sz7', '4ear', '4eb8', '1flr', '6h5w', '6h5x', '4f3c']
"""
# 2 like "step2_1_Bio_firstMODEL_firstLigand_noDisorder.py"
def bio_get_firstModel_01(pdb,in_pdb_file):

    class FirstModelFirstLigand01(Select):
        def accept_residue(self, residue):
            global lig_count
            # 1. 小分子
            if residue.get_id()[0] != ' ':
                if lig == residue.get_resname() and lig_count == 0:
                    lig_count = 1
                    return True
                elif "HOH" == residue.get_resname():
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
    out_01_pdb_file = out_1_pdb_path + pdb + "_firstModelFirstLigandWithHOH.pdb"
    io.save(out_01_pdb_file, FirstModelFirstLigand01())

    # 02
    pro_structures = p.get_structure(pdb, out_01_pdb_file)
    firstModel = pro_structures[0]
    atoms_serial_number = [a.get_serial_number() for a in firstModel.get_atoms()]
    io.set_structure(firstModel)
    out_02_pdb_file = out_2_pdb_path + pdb + "_noDisorderWithHOH.pdb"
    io.save(out_02_pdb_file, NoDisorder())

ori_pdbs_path = "F:/LLL-doc/new_data_0501/pdbs/"
out_pdb_path = "F:/LLL-doc/new_data_0501/pdbs_Bio_firstMODEL_firstLigand_nodisorder/9_addWaterBased4/"
out_1_pdb_path = out_pdb_path + "91_firstModelFirstLigandWithHOH/"
out_2_pdb_path = out_pdb_path + "92_noDisorderWithHOH/"

for i in range(ori_dataset.shape[0]):
    pdb_name = ori_dataset[i, 0]
    lig = ori_dataset[i, 1]
    lig_count = 0
    in_pdb_file = ori_pdbs_path + pdb_name + ".pdb"
    bio_get_firstModel_01(pdb_name, in_pdb_file)
print("done")

'''
done
'''



