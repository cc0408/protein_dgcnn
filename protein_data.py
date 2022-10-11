import os
import jsonlines
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

def load_data_protein(partition):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'serialized-collections')
    dim = 3 + 59 + 2
    all_label = []
    num_points = 606
    all_data = []
    all_atnum = []
    s = 0  #数据的数量
    mx = 0  #最多的box有原子数
    aa2id = {'ALA': 0,'ARG': 1,'ASN': 2,'ASP': 3,'CYS': 4,'GLN': 5,'GLU': 6,'GLY': 7,'HIS': 8,'ILE': 9,'LEU': 10,'LYS': 11,'MET': 12,'PHE': 13,'PRO': 14,'SER': 15,'THR': 16,'TRP': 17,'TYR': 18,'VAL': 19}
    atom2id = {'Ag': 0,'Al': 1,'As': 2,'Au': 3,'B': 4,'Ba': 5,'Be': 6,'Br': 7,'C': 8,'Ca': 9,'Cd': 10,'Cl': 11,'Co': 12,'Cs': 13,'Cu': 14,'Dy': 15,'Eu': 16,'F': 17,'Fe': 18,'Ga': 19,'Gd': 20,'Hg': 21,'Ho': 22,'I': 23,'Ir': 24,'K': 25,'La': 26,'Li': 27,'Lu': 28,'Mg': 29,'Mn': 30,'Mo': 31,'N': 32,'Na': 33,'Ni': 34,'O': 35,'Os': 36,'P': 37,'Pb': 38,'Pr': 39,'Pt': 40,'Rb': 41,'Ru': 42,'S': 43,'Se': 44,'Si': 45,'Sm': 46,'Sn': 47,'Sr': 48,'Ta': 49,'Tb': 50,'Te': 51,'U': 52,'V': 53,'W': 54,'Xe': 55,'Y': 56,'Yb': 57,'Zn': 58}
    file_re = 'AlkaliEarthMetalProtein-10000-snapshots.jsonl' if partition == 'test' else 'collections-0.jsonl'
    for file_name in glob.glob(os.path.join(DATA_DIR, partition, file_re)):
        with open(file_name,'r+',encoding='utf-8') as f:
            for item in jsonlines.Reader(f):
                a = aa2id[item['label']]
                bool_p = (np.array(item['atomic_collection']['element']) != 'H')
                num_atom = np.sum(bool_p)
                #itcl = np.array(item['atomic_collection'][:])[bool_p]
                tmp = []
                for coord in ['x','y','z']:
                    tmp.append(np.pad(np.array(item['atomic_collection'][coord])[bool_p],(0,num_points-num_atom),'constant'))
                ele = np.array([atom2id[key] for key in np.array(item['atomic_collection']['element'])[bool_p]])
                for i in range(59):
                    tmp.append(np.pad((ele == i),(0,num_points-num_atom),'constant'))
                tmp.append(np.pad(np.array(item['atomic_collection']['_atom_site.fw2_charge'])[bool_p],(0,num_points-num_atom),'constant'))
                tmp.append(np.pad(np.array(item['atomic_collection']['_atom_site.FreeSASA_value'])[bool_p],(0,num_points-num_atom),'constant'))
                all_data.append(tmp)
                all_label.append(a)
                all_atnum.append(num_atom)
    #s = len(all_label)
    #ept = np.zeros((s,20))
    #ept[np.arange(s),all_label] = 1
    all_label = np.array(all_label).astype('int64')
    all_data = np.transpose(np.array(all_data),(0,2,1)).astype('float32')
    return all_data, all_atnum, all_label
    
    # #                                                               train       test        val
    #                         # min number of atom                    15          19          51
    # print(len(dt))          # the number of types of animo acids    20          20          20
    # print(mx)               # max number of atom                    605         514         438
    #                         # the number of items                   1654067     264389      9999
    # print(len(all_data))    # (num_box, num_point, dim)             

class ProteinData(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.atnum, self.label = load_data_protein(partition)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item]
        label = self.label[item]
        atnum = self.atnum[item]
        # if self.partition == 'train':
        #     randomize = np.append(np.random.permutation(atnum),np.arange(atnum,self.num_points))
        #     pointcloud = pointcloud[randomize]
        atnum = np.array([1]*atnum + [0]*(self.num_points-atnum)).astype('float32')
        return pointcloud, atnum, label

    def __len__(self):
        return self.data.shape[0]

