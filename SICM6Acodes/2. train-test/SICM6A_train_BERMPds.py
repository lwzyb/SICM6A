# -*- coding: utf-8 -*-
"""
Created on 2019-6-236

@author:liu_wenzhong
"""
import torch.nn as nn

import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import pylab as pl
import datetime 

from apex import amp
################ 
epochnum=51
Basestr="-AGCU"
##################################
####### 1.Full_transcript #########
#specieID=1
#seqlen=501
#findseqlen=501
#batchnum=400
#traindatapath="C:/M6Acv/DATA/BERMP/Full_transcript training data.txt"
#testdatapath="C:/M6Acv/DATA/BERMP/Full_transcript testing data.txt" 
#savepath="C:/M6Acv/Full_transcript/"
###############################
####### 2. Mature_mRNA #########
#specieID=2
#seqlen=251
#findseqlen=251
#batchnum=1000
#traindatapath="C:/M6Acv/DATA/BERMP/Mature_mRNA training data.txt" 
#testdatapath="C:/M6Acv/DATA/BERMP/Mature_mRNA testing data.txt"

#savepath="C:/M6Acv/Mature_mRNA/Mature_mRNA"
###############################  
####### 3. A.thaliana #########
#specieID=3
#seqlen=101
#findseqlen=101
#batchnum=200
#traindatapath="C:/M6Acv/DATA/BERMP/A.thaliana_train.txt"
#testdatapath="C:/M6Acv/DATA/BERMP/A.thaliana_test.txt" 
#savepath="C:/M6Acv/A.thaliana/"
###############################  
####### 4. S.cerevisiae #########
specieID=4
seqlen=51
findseqlen=51
batchnum=200
traindatapath="C:/M6Acv/DATA/BERMP/S.cerevisiae_train.txt"
testdatapath="C:/M6Acv/DATA/BERMP/S.cerevisiae_test.txt" 
savepath="C:/M6Acv/S.cerevisiae/S.cerevisiae"
############################### 
 
cuda = True if torch.cuda.is_available() else False
 
seq_dict_train=dict()
seq_label_train=dict()
seq_dict_test=dict()
seq_label_test=dict()
trainlist_dict=dict()
testlist_dict=dict()        
onekmercode=dict()
twokmercode=dict() 
threekmercode=dict()

def ROC( y_test,y_predicted):
    y = y_test
    pred =y_predicted
    fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
    v_auc=auc(fpr, tpr)
    print(v_auc)
    #x = [_v[0] for _v in xy_arr]
    #y = [_v[1] for _v in xy_arr]
    pl.title("ROC curve of %s (AUC = %.4f)" % ('M6A' , v_auc))
    pl.xlabel("False Positive Rate")
    pl.ylabel("True Positive Rate")
    pl.plot( fpr ,tpr)
    pl.show()

    return v_auc


def buildthreekmercode():
 
    encode=dict()
   
    len1=len(Basestr )

    for i in range(len1):
        s1=Basestr[i:i+1]
        for j in range(len1):            
            s2=Basestr[j:j+1]
            for k in range(len1):
                s3=Basestr[k:k+1]
                ss=s1+s2+s3
                kk=int(str(i)+str(j)+str(k))
                encode[ss]=kk

    return encode


 
        
def buildseqqtbin( seq):
    seq=seq.strip()
    lefthalf=int((seqlen-findseqlen)/2)
    righthalf=seqlen-lefthalf 
    seq=seq[lefthalf:righthalf]  
    len1=len(seq)
    mat = np.zeros((findseqlen ), dtype=np.longlong) 
    i=0      
    while i <len1 :#3-mer  
       
        basestr=""
        if i<len1-2:
            basestr=seq[i:i+3 ]
        elif i==len1-2:
            basestr=seq[i:i+2 ]+"-"
        else:
            basestr=seq[i:i+1 ]+"--"
        bin_k=threekmercode[basestr]    
        mat[i]=bin_k       
        i +=1

    return mat 

def buildseqcode( seq_dict , seq_label   ):     
  
    imgs = []
    global num_pos_train
    global num_neg_train
    for key in  seq_dict.keys():
        seq= seq_dict[key]       
        mat= buildseqqtbin( seq)         
        label=seq_label[key]
        imgs.append((mat,label))
 
    return imgs
        


 
def readseqtodict(  seqfile):

    fo=open( seqfile ,"r")
    ls=fo.readlines() 
    seq_dict =dict()
    seq_label =dict()
    k=0
    
    if specieID==4:
        for s in ls:
            k+=1
            if k==1:
                continue
            split=s.split("\t")
            key =  str(k) 
            seq_dict[key]= split[0]
            seq_label[key]= int(split[1])
    else:
        for s in ls:
            k+=1
            if k==1:
                continue#title
            split=s.split("\t")
            key =  split[1]+"_"+split[2]+"_"+split[3] 
            seq_dict[key]= split[0]
            seq_label[key]= int(split[3])
    return seq_dict , seq_label 

 
   
    return seq_dict , seq_label 

    
class SeqDataset(Dataset):
 
    def __init__(self,  seq_dict , seq_label   ): 
        self.imgs = buildseqcode( seq_dict , seq_label   ) 

    def __getitem__(self, index):
        img, label = self.imgs[index]
        return img,label

    def __len__(self):
        return len(self.imgs)
    
#ACGU
# (1,1,1), (0,1,0), (1,0,0) and (0,0,1),      
# A-7 C-2 G-4 U-1 --0    
#-UCGA
#0.904 125-125 51-64 8,8-64 64 ok best3 22
class SICNet(nn.Module):

   def __init__(self ):   
        super(SICNet, self).__init__()

        self.embed_3_1=nn.Embedding(1024,125)

        self.LSTM3_1 =nn.GRU(input_size=125,hidden_size=125 ,bidirectional=True,
                     num_layers=2,  dropout=0.5, batch_first=True   )  
        
        self.LSTM3_2 =nn.GRU(input_size=findseqlen,hidden_size=128 ,
                             num_layers=2,  dropout=0.5, batch_first=True    )  
  
        self.LSTM3_3 =nn.GRU(input_size=16 ,hidden_size=128,
                          num_layers=2,  dropout=0.5, batch_first=True   )

        
        self.fc_1= nn.Sequential( nn.Linear(128,16) ,  
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.BatchNorm1d(16, momentum=0.5),
                                 nn.Dropout(0.25),

                                   )
        self.fc_2= nn.Sequential( nn.Linear(16,2) ) 
        

   def forward(self, x):

       
       x_3_1=self.embed_3_1 (x)

       x_3_1,_=self.LSTM3_1( x_3_1) 
       x_3_1= x_3_1.contiguous() 
       
       x_3_1=  x_3_1.permute(0,2,1)
       x_3_1,_ =self.LSTM3_2( x_3_1 )
       x_3_1=x_3_1[:,-1,:]       
       x_3_1=x_3_1.contiguous() 
       x_3_1= x_3_1.view(x_3_1.shape[0],8,16)  
       
       
       x_3 ,_  =self.LSTM3_3(x_3_1) 
       x_3 =x_3 [:,-1,:]
       x_3 =x_3 .contiguous() 
       x_4= x_3 .view(x_3 .shape[0],128) 
          

   
       x_5=self.fc_1(x_4)
       x_5=self.fc_2(x_5)
      
        

        
       out=F.log_softmax(x_5,dim=1)
      
       return out
   
def savedict(savefilename,dict1):
    target = open(savefilename, 'a')
    for key in dict1:
        auc=dict1[key]
        s=str(key) + "\t"+str(auc)+"\n"

        target.write(s)
    target.close()
    
 
def saveauc(savefilename,cvnum,i,kk,  auc):
    target = open(savefilename, 'a')
    #s0= "cvnum\trepeatnum\tepcohnum\tauc\n"
    #target.write(s0)    
    s1=str(cvnum)+"\t"+str(i)+"\t"+str(kk)+"\t"+ str(auc)+"\n"
    target.write(s1)
    target.close()   

def saveprobdata(savefilename,  probdata):
    
    target = open(savefilename, 'a')
    len1=np.shape(probdata)[0]
    for i in range(len1):
        label=int(probdata[i,0])
        prob="%.7f" % probdata[i,1] 
        
        s=str(label) + "\t"+str(prob)+"\n"

        target.write(s)
    target.close()
    
def saveauchead(savefilename ):
    target = open(savefilename, 'a')
    s0= "cvnum\trepeatnum\tepcohnum\tauc\n"
    target.write(s0)    
    
    target.close() 
    
def saveteststr(savefilename,s ):
    target = open(savefilename, 'a')
    
    target.write(s)    
    
    target.close()  
        


    

#ACGU
# (1,1,1), (0,1,0), (1,0,0) and (0,0,1),      
# A-7 C-2 G-4 U-1 --0    


if __name__ == "__main__":

    auclist=dict()
   
    threekmercode=buildthreekmercode() 
    
    seq_dict_test , seq_label_test =   readseqtodict( testdatapath)
    train_loader_list =list()
    img_test=SeqDataset(seq_dict_test , seq_label_test  )
    test_loader= DataLoader(dataset= img_test, batch_size=1000, shuffle=False)
    len_test=len(seq_label_test)
    seq_dict_train , seq_label_train =  readseqtodict(traindatapath)
    img_train=SeqDataset(seq_dict_train , seq_label_train  )
    train_loader = DataLoader(dataset=img_train, batch_size=batchnum, shuffle=True)
    
    SEED = 0
    torch.manual_seed(SEED)
    if cuda:
        torch.cuda.manual_seed(SEED)#为当前GPU设置随机种子
        torch.cuda.manual_seed_all(SEED)#为所有GPU设置随机种子
        torch.backends.cudnn.deterministic = True
    np.random.seed(SEED)  
    model =SICNet() 
    optimizer = torch.optim.Adam (model.parameters(), lr=0.0001  )
    
    loss_func =  torch.nn.CrossEntropyLoss()
    if cuda:
        model.cuda()
        loss_func.cuda()
    
    model, optimizer = amp.initialize(model, optimizer, opt_level="O2" )
    model.train()
    train_acc=0
    for epoch in range( epochnum):
        now_time = datetime.datetime.now() 
        now_time=datetime.datetime.strftime(now_time,'%Y-%m-%d %H:%M:%S')
        print('epoch {}'.format(epoch ),now_time)
        #training----------------------------
        train_loss = 0.
        rain_acc = 0.
        for ttt,(batch_x, batch_y) in  enumerate(train_loader):  
            batch_x, batch_y = Variable( batch_x.cuda() ),   Variable(batch_y.cuda())
            optimizer.zero_grad()
            out = model(batch_x ) 
            loss  = loss_func( out   , batch_y  )
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = torch.max(out, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.item()
            del loss 
        now_time = datetime.datetime.now()
        now_time=datetime.datetime.strftime(now_time,'%Y-%m-%d %H:%M:%S')
        print( now_time,'Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
                        img_train)), train_acc / (len(img_train))))
        if   epoch>=0:
            model.eval()
            eval_loss = 0.
            eval_acc = 0.
            y_test_com = np.zeros((len_test,1),dtype=np.float32)
            y_predicted_com = np.zeros((len_test,1),dtype=np.float32)
            pos_test=0
            for ttt,(batch_x, batch_y) in  enumerate(test_loader):
                batch_x, batch_y = Variable(batch_x.cuda() ),   Variable(batch_y.cuda())
                out  = model(batch_x )
                y_predicted= out.cpu() .detach().numpy()[:,1]
                len1=len( batch_y)
                y_test=batch_y.reshape((len1,1))
                y_predicted=y_predicted.reshape((len1,1))
                y_test= y_test.cpu().numpy()
                for ll in range(len1):
                    if ll+pos_test>=len_test:
                        break
                    y_test_com[ll+pos_test,0]=y_test[ll]
                    y_predicted_com[ll+pos_test,0]=y_predicted[ll]
                pos_test +=len1
            v_auc=ROC(y_test_com,y_predicted_com )
            auclist[epoch]= v_auc
            np_data_full=np.hstack((y_test_com ,y_predicted_com ))
            vaucint=int( v_auc*10000)            
            modelfile=savepath+"_model_ind_"+Basestr+"_"+ str(batchnum)+"_"+str(vaucint)+"_"+str(epoch)+"_"+str(findseqlen)+ ".pth"
            torch.save(model.state_dict() ,modelfile )
            saveprobdata(savepath+"_prob_ind_"+Basestr+"_"+ str(batchnum)+"_"+str(vaucint)+"_"+str(epoch)+"_"+str(findseqlen)+"_prob.txt", np_data_full)                
            model.train()
            model.zero_grad()
            print(  auclist )    
            
