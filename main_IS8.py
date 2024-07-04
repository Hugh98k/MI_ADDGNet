import pickle
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import ADDGNet_IS8
from torch.utils.data import Dataset
from visualdl import LogWriter
from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score,f1_score,auc,roc_curve
import scipy
from sklearn import preprocessing
import logging
import os
from torch.optim.lr_scheduler import StepLR,MultiStepLR

class dataReader(Dataset):
    def __init__(self, datas, labels, mode='test'):
        super(dataReader, self).__init__()
        assert mode in ['train', 'test'], "mode should be 'train' or 'test', but got {}".format(mode)
        self.datas = []
        self.labels = []
        if mode == 'train' :
            datas = datas[:int(datas.shape[0]*0.8), :, :, :].astype('float32')
            labels = labels[:int(labels.shape[0]*0.8)].astype('int32')
        elif mode == 'test' :
            datas = datas[int(datas.shape[0]*0.8):, :, :, :].astype('float32')
            labels = labels[int(labels.shape[0]*0.8):].astype('int32')
        self.datas = datas
        self.labels = labels
    
    def __getitem__(self, index):
        data = torch.from_numpy(self.datas[index].astype('float32'))
        label = torch.from_numpy(self.labels[index].astype('int64'))
        return data, label
    
    def __len__(self):
        return len(self.datas)

# def train(model,train_reader,test_reader,epoch_num = 700,learning_rate = 0.001,device='cuda'):
#     loss_function = nn.CrossEntropyLoss()
#     model.to(device) 
#     opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     with LogWriter(logdir='./log/train') as train_writer, LogWriter(logdir='./log/test') as test_writer:
#         for epoch in range(epoch_num):
#             model.train()
#             train_loss = 0.0
#             for batch_id, data in enumerate(train_reader):
#                 x_data = data[0]
#                 y_data = data[1].squeeze(dim=1).to(device)
#                 opt.zero_grad()

#                 x_predict,Lloss = model(x_data.to(device))

#                 loss = F.cross_entropy(x_predict, y_data) + Lloss
#                 loss = loss.requires_grad_()
                
#                 loss.backward()
#                 opt.step()

#                 # for var_name in opt.state_dict():
#                 #     print(var_name,'\t',opt.state_dict()[var_name])
    

#                 train_loss += loss
def train(model,train_reader,test_reader,adj,epoch_num = 4000,learning_rate = 0.01,device='cuda'):
    loss_function = nn.CrossEntropyLoss()
    model.to(device) 
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    PATH = './model5003.pt'
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']



    milestones = [1000,2000]
    scheduler = MultiStepLR(opt, milestones, gamma=0.1)
    maxacc = 0
    
    with LogWriter(logdir='./log/train') as train_writer, LogWriter(logdir='./log/test') as test_writer:
        # for epoch in range(epoch_num):
        #     model.train()
        #     train_loss = 0.0
        #     train_lloss = 0.0
        #     for batch_id, data in enumerate(train_reader):
        #         x_data = data[0]
        #         y_data = data[1].squeeze(dim=1).to(device)
        #         opt.zero_grad()

        #         x_predict,Lloss = model(x_data.to(device),adj)

        #         loss = F.cross_entropy(x_predict, y_data) + Lloss
        #         loss = loss.requires_grad_()
                
        #         loss.backward()
        #         opt.step()
        #         scheduler.step()

        #         # for var_name in opt.state_dict():
        #         #     print(var_name,'\t',opt.state_dict()[var_name])
    

        #         train_loss += loss
        #         train_lloss += Lloss

        #     #     rate = (batch_id+1)/len(train_reader)
        #     #     a = "*" * int(rate * 50)
        #     #     b = "." * int((1 - rate) * 50)
        #     #     print("\r{:^3.0f}%[{}->{}]".format(int(rate*100), a, b), end="")
        #     # print()
        #     print('train_loss: %.3f ' %
        #     (train_loss / batch_id))


			
		# 验证模式
        test_loss = 0.0
        model.eval()
        accs = []
        ypre=[]
        ytrue=[]
        for batch_id, data in enumerate(test_reader):
            x_data = data[0]
            y_data = data[1].squeeze(dim=1).to(device)
            outputs,_ = model(x_data.to(device),adj)

                
            predict_y = torch.max(outputs, dim=1)[1]
            ypre_=predict_y.cpu().detach().numpy()
            val_labels_=y_data.cpu().detach().numpy()
            ypre.extend(ypre_)
            ytrue.extend(val_labels_)



            # ypre_ = x_predict.cpu().detach().numpy()
            # acc = accuracy_score(ypre_, y_data)
            # accs.append(acc.numpy())
            # losses.append(loss.numpy())

        # avg_acc = np.mean(accs)
        # print("[test] accuracy : {}".format(avg_acc))
        # test_writer.add_scalar(tag='test/acc', step=epoch, value=avg_acc)

        acc=accuracy_score(ytrue,ypre)
        f1 = f1_score(ytrue,ypre , average='macro')
        class_names = np.array([0, 1, 2, 3])
        y_binarize = label_binarize(ytrue, classes=class_names)
        y_fit=label_binarize(ypre, classes = class_names)
        fpr, tpr, _= roc_curve(y_binarize.ravel(),y_fit.ravel())
        nauc = auc(fpr, tpr)




        # torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': model.state_dict(),
        #     'optimizer_state_dict': opt.state_dict(),
        # }, PATH)


                
        print('Accuracy: %.3f  F1: %.3f  AUC: %.3f' %
            (acc,f1,nauc))

            
def get_adjacency(x):
    corr_matrix = np.corrcoef(x) 
    abs_corr = np.abs(corr_matrix)
    res = abs_corr - np.eye(abs_corr.shape[0])
    return res


if __name__ == '__main__':


    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    DEVICE = 'cuda'

    # # with open('../data/想象语音八分类DATA/new128LongEEGdata.pickle', 'rb') as f:
    # Lan_datalist = np.load('../data/想象语音八分类DATA/new128LongEEGdata.npy')

    # with open('../data/想象语音八分类DATA/Lan_label.pickle', 'rb') as f:
    #     Lan_labellist = pickle.load(f)

    # with open('../data/想象语音八分类DATA/new128graphs.pickle', 'rb') as f:
    #     graphs = pickle.load(f)



    with open('../../../data/想象语音八分类DATA/Data.pickle', 'rb') as f:
        Lan_datalist = pickle.load(f)


    with open('../../../data/想象语音八分类DATA/Label.pickle', 'rb') as f:
        Lan_labellist = pickle.load(f)





    # 展平
    Lan_datalist = np.squeeze(Lan_datalist)



    meandata = np.mean(Lan_datalist,axis=0)
    adj = get_adjacency(meandata)
    adj = np.abs(adj)
    adj = torch.from_numpy(adj).to(DEVICE).to(torch.float32)








    n, c, m = Lan_datalist.shape
    res = []
    for i in range(c):
        tmp = Lan_datalist[:, i, :].reshape(n * m)
        res.append(tmp)
    Lan_datalist = np.array(res)


   






 
    # # 归一化
    # max_abs_scaler = preprocessing.MaxAbsScaler()
    # Lan_datalist = max_abs_scaler.fit_transform(Lan_datalist)

    # 转置为(127,1250*N) 然后按段划分为 (50*N,127,25) 并按段制作新的label  (7200*(1250//WINDOW),127,WINDOW)
    TIMEWINDOW = 100
    Lan_datalist = Lan_datalist.reshape(Lan_datalist.shape[0],-1,TIMEWINDOW)
    Lan_datalist = np.transpose(Lan_datalist, (1,0,2))
    Lan_labellist = np.repeat(Lan_labellist, 1000 // TIMEWINDOW)

    # 标准化
    # Lan_datalist = np.transpose(Lan_datalist, (0,2,1))
    # max_abs_scaler = preprocessing.MaxAbsScaler()
    # for i in range(Lan_datalist.shape[0]):
    #     Lan_datalist[i]=max_abs_scaler.fit_transform(Lan_datalist[i])
    # Lan_datalist = np.transpose(Lan_datalist, (0,2,1))

    max_abs_scaler = preprocessing.MaxAbsScaler()
    for i in range(Lan_datalist.shape[0]):
        Lan_datalist[i] = max_abs_scaler.fit_transform(Lan_datalist[i].T).T


    Lan_datalist = np.expand_dims(Lan_datalist,axis=1)
    Lan_labellist = np.expand_dims(Lan_labellist,axis=1)
 

    BATCH_SIZE = 32
    train_loader = dataReader(Lan_datalist, Lan_labellist, 'train')
    test_loader = dataReader(Lan_datalist, Lan_labellist, 'test')
    train_reader = torch.utils.data.DataLoader(train_loader, batch_size=BATCH_SIZE, shuffle=True)
    test_reader = torch.utils.data.DataLoader(test_loader, batch_size=BATCH_SIZE)

    model = ADDGNet_IS8.ADDGNet()

    

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # for var_name in optimizer.state_dict():
    #     print(var_name,'\t',optimizer.state_dict()[var_name])
    
    # print(type(model.named_parameters()))  # 返回的是一个generator
    
    # for para in model.named_parameters(): # 返回的每一个元素是一个元组 tuple 
    #     '''
    #     是一个元组 tuple ,元组的第一个元素是参数所对应的名称，第二个元素就是对应的参数值
    #     '''
    #     print(para[0],'\t',para[1].size())


    # summary(model.to('cuda'), ( 1,127, 1250))

    



    train(model,train_reader,test_reader,adj)

   