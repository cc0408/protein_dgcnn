import os
import jsonlines
import glob
import math
import numpy as np
import torch
from torch.utils.data import Dataset

def load_data_protein(partition):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'serialized-collections')
    dim = 3
    all_label = []
    num_points = 606
    all_data = []
    all_atnum = []
    s = 0  #数据的数量
    mx = 0  #最多的box有原子数
    atset = ['C', 'N', 'O', 'P', 'S']
    aa2id = {'ALA': 0,'ARG': 1,'ASN': 2,'ASP': 3,'CYS': 4,'GLN': 5,'GLU': 6,'GLY': 7,'HIS': 8,'ILE': 9,'LEU': 10,'LYS': 11,'MET': 12,'PHE': 13,'PRO': 14,'SER': 15,'THR': 16,'TRP': 17,'TYR': 18,'VAL': 19}
    atom2id = {'Ag': 0,'Al': 1,'As': 2,'Au': 3,'B': 4,'Ba': 5,'Be': 6,'Br': 7,'C': 8,'Ca': 9,'Cd': 10,'Cl': 11,'Co': 12,'Cs': 13,'Cu': 14,'Dy': 15,'Eu': 16,'F': 17,'Fe': 18,'Ga': 19,'Gd': 20,'Hg': 21,'Ho': 22,'I': 23,'Ir': 24,'K': 25,'La': 26,'Li': 27,'Lu': 28,'Mg': 29,'Mn': 30,'Mo': 31,'N': 32,'Na': 33,'Ni': 34,'O': 35,'Os': 36,'P': 37,'Pb': 38,'Pr': 39,'Pt': 40,'Rb': 41,'Ru': 42,'S': 43,'Se': 44,'Si': 45,'Sm': 46,'Sn': 47,'Sr': 48,'Ta': 49,'Tb': 50,'Te': 51,'U': 52,'V': 53,'W': 54,'Xe': 55,'Y': 56,'Yb': 57,'Zn': 58}
    file_re = 'collections-1.jsonl' if partition == 'test' else 'collections-0.jsonl'
    for file_name in glob.glob(os.path.join(DATA_DIR, 'train', file_re)):
        with open(file_name,'r+',encoding='utf-8') as f:
            for item in jsonlines.Reader(f):
                a = aa2id[item['label']]
                bool_p = np.array([k in atset for k in item['atomic_collection']['element']])
                num_atom = np.sum(bool_p)
                #itcl = np.array(item['atomic_collection'][:])[bool_p]
                tmp = []
                for coord in ['x','y','z']:
                    bb = np.array(item['atomic_collection'][coord])[bool_p]
                    am=(np.min(bb)+np.max(bb))/2
                    ad=(np.max(bb)-np.min(bb))/2
                    tmp.append(np.pad((bb-am),(0,num_points-num_atom),'constant'))
                #if tmp[0][-1]>1e-6 or tmp[0][-1]<-1e-6:
                #    print(tmp)
                all_data.append(tmp)
                all_label.append(a)
                all_atnum.append(num_atom)
    #s = len(all_label)
    #ept = np.zeros((s,20))
    #ept[np.arange(s),all_label] = 1
    all_label = np.array(all_label).astype('int64')
    all_data = np.transpose(np.array(all_data),(0,2,1)).astype('float32')
    all_atnum = np.array(all_atnum).astype('int64')
    #print(all_data.shape)
    randomize = np.random.permutation(all_data.shape[0])
    all_data = all_data[randomize]
    all_label = all_label[randomize]
    all_atnum = all_atnum[randomize]
    if partition == 'train':
        sap = 100000
    else:
        sap = 20000
    all_data = all_data[:sap]
    all_label = all_label[:sap]
    all_atnum = all_atnum[:sap]
    #print(math.sqrt((all_data**2).mean()/all_atnum.mean()))
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
        #print(pointcloud)
        return pointcloud, atnum, label

    def __len__(self):
        return self.data.shape[0]

