import mne
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils.data import Dataset
import torch.optim as optim
import torch
import torch.nn as nn
import random
import sklearn
import torch.nn.functional as F
from visualdl import LogWriter
from torch.autograd import Variable
import pickle

if __name__ == '__main__':
    

    file = '../data/'

    Lan_datalist=[]
    Lan_labellist=[]



    # 想象语音 中英文八分类
    for root, dirs, files in os.walk(file):
        for file in files:
            path = os.path.join(root, file)
            if '.set' in path and 'MI_read' in path:
                epo = mne.io.read_epochs_eeglab(path, uint16_codec='latin1')
                for i in range(64):
                    Lan_datalist.append(epo[i]._data[:,:,:])
                    # Lan_labellist.append(0 if 'CN' in path else 1)
                    if 'CN' in path:
                        Lan_labellist.append(list(epo[i].event_id.values())[0] - 1)
                    elif 'EN' in path:
                        Lan_labellist.append(list(epo[i].event_id.values())[0] + 3)




    Lan_datalist = np.array(Lan_datalist)
    Lan_labellist = np.array(Lan_labellist)
    Lan_labellist = Lan_labellist.reshape(Lan_labellist.shape[0], 1)


    # x_train, y_train = sklearn.utils.shuffle(Lan_datalist, Lan_labellist, random_state=0)
    seed=77
    random.seed(seed)
    random.shuffle(Lan_datalist)
    random.seed(seed)
    random.shuffle(Lan_labellist)

    with open('../data/八分类/data.pickle', 'wb') as f:
        pickle.dump(Lan_datalist, f)

    with open('../data/八分类/label.pickle', 'wb') as f:
        pickle.dump(Lan_labellist, f)
 
 

