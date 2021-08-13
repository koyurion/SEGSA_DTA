from functools import partial
import os
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import scipy.sparse as sp
from data_process.step5_ligandFeature.Featurizer_ParticalCharge_0608 import *
import time
from rdkit.Chem import AllChem
from rdkit import Chem
import pickle
import numpy as np
smilesList = ['CC']
# TODO: [1,2,3,4]
# degrees = [0, 1, 2, 3, 4, 5]
degrees = [1, 2, 3, 4]
# todo


class MolGraph(object):
    def __init__(self):
        self.nodes = {}  # dict of lists of nodes, keyed by node type

    def new_node(self, ntype, features=None, rdkit_ix=None):
        new_node = Node(ntype, features, rdkit_ix)
        # dict.setdefault(key, default=None)
        self.nodes.setdefault(ntype, []).append(new_node)
        return new_node

    def add_subgraph(self, subgraph):
        old_nodes = self.nodes
        new_nodes = subgraph.nodes
        for ntype in set(old_nodes.keys()) | set(new_nodes.keys()):
            old_nodes.setdefault(ntype, []).extend(new_nodes.get(ntype, []))
    # todo: 1
    def sort_nodes_by_degree(self, ntype):
        nodes_by_degree = {i: [] for i in degrees}
        for node in self.nodes[ntype]:
            nodes_by_degree[len(node.get_neighbors(ntype))].append(node)

        new_nodes = []
        for degree in degrees:
            cur_nodes = nodes_by_degree[degree]
            self.nodes[(ntype, degree)] = cur_nodes
            #new_nodes.extend(cur_nodes)

        #self.nodes[ntype] = new_nodes

    def feature_array(self, ntype):
        assert ntype in self.nodes
        return np.array([node.features for node in self.nodes[ntype]])

    def rdkit_ix_array(self):
        return np.array([node.rdkit_ix for node in self.nodes['atom']])

    def neighbor_list(self, self_ntype, neighbor_ntype):
        assert self_ntype in self.nodes and neighbor_ntype in self.nodes
        neighbor_idxs = {n: i for i, n in enumerate(self.nodes[neighbor_ntype])}
        return [[neighbor_idxs[neighbor]
                 for neighbor in self_node.get_neighbors(neighbor_ntype)]
                for self_node in self.nodes[self_ntype]]

    def order_neighbor_list(self, self_ntype, neighbor_ntype):
        assert self_ntype in self.nodes and neighbor_ntype in self.nodes
        neighbor_idxs = {n: i for i, n in enumerate(self.nodes[neighbor_ntype])}
        return [[neighbor_idxs[neighbor]
                 for neighbor in self_node.get_neighbors(neighbor_ntype)]
                for self_node in self.nodes[self_ntype]]

class Node(object):
    __slots__ = ['ntype', 'features', '_neighbors', 'rdkit_ix']

    def __init__(self, ntype, features, rdkit_ix):
        self.ntype = ntype
        self.features = features
        self._neighbors = []
        self.rdkit_ix = rdkit_ix

    def add_neighbors(self, neighbor_list):
        for neighbor in neighbor_list:
            self._neighbors.append(neighbor)
            neighbor._neighbors.append(self)

    def get_neighbors(self, ntype):
        return [n for n in self._neighbors if n.ntype == ntype]


class memoize(object):
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if args in self.cache:
            return self.cache[args]
        else:
            result = self.func(*args)
            self.cache[args] = result
            return result

    def __get__(self, obj, objtype):
        return partial(self.__call__, obj)


def graph_from_smiles_tuple(smiles_tuple):
    graph_list = [graph_from_lig(s) for s in smiles_tuple]
    big_graph = MolGraph()
    for subgraph in graph_list:
        big_graph.add_subgraph(subgraph)

    # This sorting allows an efficient (but brittle!) indexing later on.
    big_graph.sort_nodes_by_degree('atom')
    return big_graph


def graph_from_lig(mol_dict,lig):

    try:
        mol = mol_dict[lig]
    except:
        return None

    if not mol:
        raise ValueError("Could not parse SMILES string:", lig)
    graph = MolGraph()
    atoms_by_rd_idx = {}
    AllChem.ComputeGasteigerCharges(mol)
    Chem.AssignStereochemistry(mol, flagPossibleStereoCenters=True)
    for atom in mol.GetAtoms():
        # print(lig,atom.GetSymbol())
        new_atom_node = graph.new_node('atom', features=atom_features(atom), rdkit_ix=atom.GetIdx())
        atoms_by_rd_idx[atom.GetIdx()] = new_atom_node

    for bond in mol.GetBonds():
        #print(bond.GetIdx(), end='\t')
        #print(bond.GetBeginAtomIdx(), end='\t')
        #print(bond.GetEndAtomIdx())

        atom1_node = atoms_by_rd_idx[bond.GetBeginAtom().GetIdx()]
        atom2_node = atoms_by_rd_idx[bond.GetEndAtom().GetIdx()]
        new_bond_node = graph.new_node('bond', features=bond_features(bond))
        new_bond_node.add_neighbors((atom1_node, atom2_node))
        atom1_node.add_neighbors((atom2_node,))

    mol_node = graph.new_node('molecule')  # super node :used to get all nodes(atom list)
    mol_node.add_neighbors(graph.nodes['atom'])
    return graph


def array_rep_from_mol(molgraph):
    """Precompute everything we need from MolGraph so that we can free the memory asap."""
    # molgraph = graph_from_smiles_tuple(tuple(smiles))

    arrayrep = {'atom_features': molgraph.feature_array('atom'),
                'bond_features': molgraph.feature_array('bond'),
                'atom_list': molgraph.neighbor_list('molecule', 'atom'),
                'rdkit_ix': molgraph.rdkit_ix_array()}

    arrayrep['atom_neighbors'] = molgraph.order_neighbor_list('atom', 'atom')
    arrayrep['bond_neighbors'] = molgraph.order_neighbor_list('atom', 'bond')
    return arrayrep


def num_atom_features():
    # Return length of feature vector using a very simple molecule.
    m = Chem.MolFromSmiles('CC')
    alist = m.GetAtoms()
    a = alist[0]
    return len(atom_features(a))


def num_bond_features():
    # Return length of feature vector using a very simple molecule.
    simple_mol = Chem.MolFromSmiles('CC')
    Chem.SanitizeMol(simple_mol)
    return len(bond_features(simple_mol.GetBonds()[0]))


def gen_descriptor_data(Lig_List):
    ligmol_to_fingerprint_array = {}
    miss_gen_descriptor_data_list=[]
    miss_ligand_pdb_list=[]
    # print(Lig_List)
    mol_dict = get_mol_dict()
    
    for i, lig in enumerate(Lig_List):
        #         if i > 5:
        #             print("Due to the limited computational resource, submission with more than 5 molecules will not be processed")
        #             break
        #smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True)
        try:
            molgraph = graph_from_lig(mol_dict, lig)
            if molgraph is None:
                miss_ligand_pdb_list.append(lig)
                continue
            # molgraph.sort_nodes_by_degree('atom')
            arrayrep = array_rep_from_mol(molgraph)

            ligmol_to_fingerprint_array[lig] = arrayrep

        except:
            print(lig)
            miss_gen_descriptor_data_list.append(lig)
            time.sleep(1)
    print("miss_ligand_pdb_list",len(miss_ligand_pdb_list),miss_ligand_pdb_list)
    print("miss_gen_descriptor_data_list",len(miss_gen_descriptor_data_list),miss_gen_descriptor_data_list)
    return ligmol_to_fingerprint_array

def get_mol_dict():
    mol_dict_path = "step1_screen/lig_mol_data/mol_dict"
    if os.path.exists(mol_dict_path):
        with open(mol_dict_path, "rb") as f:
            mol_dict = pickle.load(f)
    else:
        mol_dict = {}
        sdf_path = "step1_screen/lig_mol_data/Components-pub.sdf"
        mols = Chem.SDMolSupplier(sdf_path)
        for m in mols:
            if m is None:
                continue
            name = m.GetProp("_Name")
            mol_dict[name] = m
        with open(mol_dict_path, 'wb') as f:
            pickle.dump(mol_dict, f)
    print('mol_dict',len(mol_dict))
    return mol_dict

def normalize_half(adj):
    """TF_version adj: Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocsr()

def normalize(mx):
    """Pytorch_version fea/adj;TF_version adj"""
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(mx)

def printPercentile(resLen_array):
    print("mean",np.mean(resLen_array))
    # print("25%",np.percentile(resLen_array, 25))
    print("50%",np.median(resLen_array))
    print("75%",np.percentile(resLen_array, 75))
    print("90%",np.percentile(resLen_array, 90))
    print("95%",np.percentile(resLen_array, 95))
    print("99%",np.percentile(resLen_array, 99))

def get_gcn_data(ligmol_to_fingerprint_features,gcn_data_filename,halfNormalizeFlag):
    max_atom_len = 48
    lig_fea_gcn_Padded = {}
    atomLen_bondLen_list = []
    for lig, arrayrep in ligmol_to_fingerprint_features.items():

        atom_features = arrayrep['atom_features']
        bond_features = arrayrep['bond_features']
        atom_len, num_atom_features = atom_features.shape
        bond_len, num_bond_features = bond_features.shape
        atomLen_bondLen_list.append([lig,atom_len,bond_len])

        mask = np.zeros((max_atom_len))
        mask[:atom_len] = 1.0
        atom_features = sp.csr_matrix(atom_features, dtype=np.float32)
        atom_features = normalize(atom_features)
        neighbor_bond_features = []

        atom_neighbors_list = arrayrep['atom_neighbors']
        bond_neighbors_list = arrayrep['bond_neighbors']
        edges = []
        for i, degree_array in enumerate(atom_neighbors_list):
            curAtom_neighbor_bond_features = np.zeros((max_atom_len,num_bond_features))
            curAtom_bond_neighbors = bond_neighbors_list[i]
            for j in range(len(degree_array)):
                neighbor_atom_idx = degree_array[j]
                edges.append([i,neighbor_atom_idx])
                curAtom_neighbor_bond_features[neighbor_atom_idx] = curAtom_bond_neighbors[j]
            curAtom_neighbor_bond_features = normalize(sp.csr_matrix(curAtom_neighbor_bond_features)).toarray()
            neighbor_bond_features.append(curAtom_neighbor_bond_features)
            
        neighbor_bond_features = np.array(neighbor_bond_features)
        pad_bond_featureMat = np.zeros((max_atom_len,max_atom_len,num_bond_features))
        pad_bond_featureMat[:neighbor_bond_features.shape[0]] = neighbor_bond_features

        edges = np.array(edges)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(atom_len, atom_len),
                            dtype=np.float32)

        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        if halfNormalizeFlag:
            adj = normalize_half(adj + sp.eye(adj.shape[0]))
        else:
            adj = normalize(adj + sp.eye(adj.shape[0]))

        pad_atom_num = max_atom_len - atom_len
        pad_atom_featureMat = sp.vstack([atom_features, sp.csr_matrix(np.zeros((pad_atom_num, num_atom_features)))])

        temp_conacts_mat = sp.coo_matrix(np.zeros((pad_atom_num, pad_atom_num)))
        pad_adj = sp.bmat([[adj, None], [None, temp_conacts_mat]])

        lig_fea_gcn_Padded[lig] = {}
        lig_fea_gcn_Padded[lig]["mask"] = mask
        lig_fea_gcn_Padded[lig]["atomFeatureMat_Pad"] = pad_atom_featureMat.toarray()
        lig_fea_gcn_Padded[lig]["bondFeatureMat_Pad"] = pad_bond_featureMat
        lig_fea_gcn_Padded[lig]["conactsMat_Pad"] = pad_adj.toarray()

    print("saving pad ligand dict:", len(lig_fea_gcn_Padded.keys()))
    with open(gcn_data_filename, "wb") as f:
        pickle.dump(lig_fea_gcn_Padded, f)
    f.close()

    atomLen_bondLen_list = np.array(atomLen_bondLen_list)
    df_atomLen = pd.DataFrame(atomLen_bondLen_list)
    df_atomLen.to_csv(gcn_data_filename.split(".")[0] + "_df_atomLen_bondLen.csv")

    atomLen_list = atomLen_bondLen_list[:, 1].astype(np.int)
    bondLen_list = atomLen_bondLen_list[:, -1].astype(np.int)
    print("=== atom_len")
    printPercentile(atomLen_list)
    print("=== bond_len")
    printPercentile(bondLen_list)
    print("done. pad ligand dict")

    return lig_fea_gcn_Padded


def save_smiles_dicts(Lig_List, filename, gcn_data_filename,halfNormalizeFlag):
    # filename: '../data/delaney-processed'
    # first need to get the max atom length
    max_atom_len = 0
    max_bond_len = 0
    num_atom_features = 0
    num_bond_features = 0
    ligmol_to_rdkit_list = {}

    ligmol_to_fingerprint_features = gen_descriptor_data(Lig_List)

    lig_fea_gcn_Padded = get_gcn_data(ligmol_to_fingerprint_features,gcn_data_filename,halfNormalizeFlag)

    for lig, arrayrep in ligmol_to_fingerprint_features.items():
        atom_features = arrayrep['atom_features']
        bond_features = arrayrep['bond_features']

        rdkit_list = arrayrep['rdkit_ix']
        ligmol_to_rdkit_list[lig] = rdkit_list

        atom_len, num_atom_features = atom_features.shape
        bond_len, num_bond_features = bond_features.shape

        if atom_len > max_atom_len:
            max_atom_len = atom_len
        if bond_len > max_bond_len:
            max_bond_len = bond_len

    # 0608 according to the statistic over all lig datas
    print("cur max_atom_len & max_bond_len:",max_atom_len, max_bond_len)
    max_atom_len = 48
    max_bond_len = 53
    print("statistic max_atom_len & max_bond_len:", max_atom_len, max_bond_len)

    # then add 1 so I can zero pad everything
    max_atom_index_num = max_atom_len
    max_bond_index_num = max_bond_len

    max_atom_len += 1
    max_bond_len += 1

    ligmol_to_atom_info = {}
    ligmol_to_bond_info = {}

    ligmol_to_atom_neighbors = {}
    ligmol_to_bond_neighbors = {}

    ligmol_to_atom_mask = {}


    # then run through our numpy array again
    for lig, arrayrep in ligmol_to_fingerprint_features.items():

        mask = np.zeros(max_atom_len)

        # get the basic info of what
        #    my atoms and bonds are initialized
        atoms = np.zeros((max_atom_len, num_atom_features))
        bonds = np.zeros((max_bond_len, num_bond_features))

        # then get the arrays initlialized for the neighbors
        atom_neighbors = np.zeros((max_atom_len, len(degrees)))
        bond_neighbors = np.zeros((max_atom_len, len(degrees)))

        # now set these all to the last element of the list, which is zero padded
        atom_neighbors.fill(max_atom_index_num)
        bond_neighbors.fill(max_bond_index_num)

        atom_features = arrayrep['atom_features']
        bond_features = arrayrep['bond_features']

        for i, feature in enumerate(atom_features):  # TODO: mask
            mask[i] = 1.0
            atoms[i] = feature

        for j, feature in enumerate(bond_features):
            bonds[j] = feature

        atom_neighbor_count = 0
        bond_neighbor_count = 0
        working_atom_list = []
        working_bond_list = []

        atom_neighbors_list = arrayrep['atom_neighbors']
        bond_neighbors_list = arrayrep['bond_neighbors']

        for i, degree_array in enumerate(atom_neighbors_list):
            for j, value in enumerate(degree_array):
                atom_neighbors[i, j] = value

        for i, degree_array in enumerate(bond_neighbors_list):
            for j, value in enumerate(degree_array):
                bond_neighbors[i, j] = value

        # then add everything to my arrays
        ligmol_to_atom_info[lig] = atoms
        ligmol_to_bond_info[lig] = bonds

        ligmol_to_atom_neighbors[lig] = atom_neighbors
        ligmol_to_bond_neighbors[lig] = bond_neighbors

        ligmol_to_atom_mask[lig] = mask

    del ligmol_to_fingerprint_features

    feature_dicts = {
        'ligmol_to_atom_mask': ligmol_to_atom_mask,
        'ligmol_to_atom_info': ligmol_to_atom_info,
        'ligmol_to_bond_info': ligmol_to_bond_info,
        'ligmol_to_atom_neighbors': ligmol_to_atom_neighbors,
        'ligmol_to_bond_neighbors': ligmol_to_bond_neighbors,
        'ligmol_to_rdkit_list': ligmol_to_rdkit_list
    }
    pickle.dump(feature_dicts, open(filename, "wb"))
    print('feature dicts file saved as ' + filename)
    return feature_dicts,lig_fea_gcn_Padded


def get_smiles_array(Lig_List, feature_dicts):
    x_mask = []
    x_atom = []
    x_bonds = []
    x_atom_index = []
    x_bond_index = []
    for lig in Lig_List:
        x_mask.append(feature_dicts['ligmol_to_atom_mask'][lig])
        x_atom.append(feature_dicts['ligmol_to_atom_info'][lig])
        x_bonds.append(feature_dicts['ligmol_to_bond_info'][lig])
        x_atom_index.append(feature_dicts['ligmol_to_atom_neighbors'][lig])
        x_bond_index.append(feature_dicts['ligmol_to_bond_neighbors'][lig])
    return np.asarray(x_atom), np.asarray(x_bonds), np.asarray(x_atom_index), \
           np.asarray(x_bond_index), np.asarray(x_mask), feature_dicts['ligmol_to_rdkit_list']


def get_dictvx_array(pdbids_list,dict_Vx):
    Vx_list=[]
    Vn_list=[]
    ff_mask=[]
    degree_list=[]
    for pdbid in pdbids_list:
        vx,vn,ori_degree,mask=dict_Vx[pdbid]
        Vx_list.append(vx)
        Vn_list.append(vn)
        degree_list.append(ori_degree)
        ff_mask.append(mask)
    degree_list=np.asarray(degree_list)

    return np.asarray(Vx_list),np.asarray(Vn_list),degree_list,np.asarray(ff_mask)



