# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 12:14:44 2018

@author: Administrator
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

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda. FloatTensor if cuda else torch.FloatTensor  
#####################################
seqlen=101 
findseqlen=101
batchnum=90
epochnum=51
Basestr="NAGCT"     
deepm6adatapath="C:/M6Acv/DATA/DeepM6ASeq" 
savepath="C:/M6Acv/HepG2_humanbrain/HepG2"
#####################################

seq_dict_train=dict()
seq_label_train=dict()
seq_dict_test=dict()
seq_label_test=dict()
trainlist_dict=dict()
testlist_dict=dict()        
onekmercode=dict()
twokmercode=dict() 
threekmercode=dict()
torch.cuda.set_device(0)

def ROC( y_test,y_predicted,picname):
    y = y_test#np.array(y_test)
    pred =y_predicted# np.array(y_predicted)
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
                ss =s1+s2+s3 

                ss1=str(i )+str(j  )+str(k  )
                kk=int (ss1)
 
                  
                encode[ss]=kk
           
    return encode


        

def buildseqqtbin( seq):
    seq=seq.strip()
    lefthalf=int((seqlen-findseqlen)/2)
    righthalf=seqlen-lefthalf
 
    seq=seq[lefthalf:righthalf]  
    len1=len(seq) 
    mat   = np.zeros((findseqlen  ), dtype=np.longlong) 
    
    i=0 
    while i <len1 :#3-mer 
        
        basestr=""
        if i<len1-2:
            basestr=seq[i:i+3 ]
        elif i==len1-2:
            basestr=seq[i:i+2 ]+"N"
           
        else:
            basestr=seq[i:i+1 ]+"NN"
        tmp =threekmercode[basestr]
                 
        mat[i]=tmp
        i +=1  
  
    return mat  

def buildseqcode( seq_dict , seq_label  ):     
  
    imgs = []
    global num_pos_train
    global num_neg_train
    for key in  seq_dict.keys():
        seq= seq_dict[key]
       
        mat= buildseqqtbin( seq)
         
        label=seq_label[key]
        imgs.append((mat,label))
 
    return imgs
        



class SeqDataset(Dataset):
 
    def __init__(self,  seq_dict , seq_label  ): 
        self.imgs = buildseqcode( seq_dict , seq_label  ) 
        
      

    def __getitem__(self, index):
        img, label = self.imgs[index]
        return img,label
       

    def __len__(self):
        return len(self.imgs)
    
 
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
        
 
def readseqtodict(  seqfile,labelnum):
    fo=open( seqfile ,"r")
    ls=fo.readlines()
    seq_dict =dict()
    seq_label =dict()
    key=""
    for s in ls:            
        
        if s[0]==">":
            key=s
            continue        
        seq_dict[key]= s
        seq_label[key]= int(labelnum)
    return seq_dict , seq_label 


def readfragdatatodic():
    lseq_dict_train=dict()
    lseq_label_train=dict()
    lseq_dict_test=dict()
    lseq_label_test=dict() 
    
    #read mm mouse    
    #train_pos   
    lseq_dict_train_tmp , lseq_label_train_tmp =  readseqtodict(deepm6adatapath+"/mm/"+"train_pos.fa",1)
    lseq_dict_train.update(lseq_dict_train_tmp)
    lseq_label_train.update(lseq_label_train_tmp )
    #train_neg
    lseq_dict_train_tmp , lseq_label_train_tmp =  readseqtodict(deepm6adatapath+"/mm/"+"train_neg.fa",0)
    lseq_dict_train.update(lseq_dict_train_tmp)
    lseq_label_train.update(lseq_label_train_tmp )     
    
    #hs human
    #train_pos
    lseq_dict_train_tmp , lseq_label_train_tmp =  readseqtodict(deepm6adatapath+"/hs/"+"train_pos.fa",1)
    lseq_dict_train.update(lseq_dict_train_tmp)
    lseq_label_train.update(lseq_label_train_tmp )
    #train_neg
    lseq_dict_train_tmp , lseq_label_train_tmp =  readseqtodict(deepm6adatapath+"/hs/"+"train_neg.fa",0)
    lseq_dict_train.update(lseq_dict_train_tmp)
    lseq_label_train.update(lseq_label_train_tmp )   
  
    #test_pos    
    lseq_dict_test_tmp , lseq_label_test_tmp =  readseqtodict(deepm6adatapath+"/hs/hepg2_brain/"+"test_pos.fa",1)
    lseq_dict_test.update(lseq_dict_test_tmp)
    lseq_label_test.update(lseq_label_test_tmp )
    #test_neg
    lseq_dict_test_tmp , lseq_label_test_tmp =  readseqtodict(deepm6adatapath+"/hs/hepg2_brain/"+"test_neg.fa",0)
    lseq_dict_test.update(lseq_dict_test_tmp)
    lseq_label_test.update(lseq_label_test_tmp )   
    

    
    return lseq_dict_train , lseq_label_train,lseq_dict_test , lseq_label_test
torch.cuda.set_device(0)

if __name__ == "__main__":
   

    ttt=[75,80,85,95,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000,1100,1200,1400,1500,1600,1800,2000]
    for tt in ttt:
        batchnum=tt
        threekmercode=buildthreekmercode()
        auclist=dict()
        seq_dict_train , seq_label_train,seq_dict_test , seq_label_test  =  readfragdatatodic()
        len_test=len(seq_label_test)
        len_train=len(seq_label_train)
        SEED = 0
        torch.manual_seed(SEED)
        if cuda:
            torch.cuda.manual_seed(SEED)#为当前GPU设置随机种子
            torch.cuda.manual_seed_all(SEED)#为所有GPU设置随机种子
            torch.backends.cudnn.deterministic = True 
        np.random.seed(SEED)  
        img_train=SeqDataset(seq_dict_train , seq_label_train )
        img_test=SeqDataset(seq_dict_test , seq_label_test)
        
        train_loader = DataLoader(dataset=img_train, batch_size=batchnum, shuffle=True)
        test_loader= DataLoader(dataset= img_test, batch_size=1000, shuffle=False)
    
        model =SICNet() 
        
        fun_loss = torch.nn.CrossEntropyLoss()
        if cuda:
            model.cuda()
            fun_loss.cuda()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001  ) 
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2")
        model.train()
    
        for epoch in range( epochnum):
            now_time = datetime.datetime.now() 
            now_time=datetime.datetime.strftime(now_time,'%Y-%m-%d %H:%M:%S')
            print('epoch {}'.format(epoch ),now_time)
            #training----------------------------
            train_loss = 0.
            train_acc = 0.
            
            for ttt,(batch_x, batch_y) in  enumerate(train_loader):   
                batch_x, batch_y = Variable( batch_x.cuda() ),   Variable(batch_y.cuda())
                optimizer.zero_grad()
                out    = model(batch_x ) 
                loss  = fun_loss( out   , batch_y  )
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                optimizer.step()
    
                train_loss += loss.item()
                pred = torch.max(out, 1)[1]
                train_correct = (pred == batch_y).sum()
                train_acc += train_correct.item()
                
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
                
                for batch_x, batch_y in test_loader:
                    batch_x, batch_y = Variable(batch_x.cuda() ),   Variable(batch_y.cuda())
                    out   = model(batch_x )
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
                    
                v_auc=ROC(y_test_com,y_predicted_com,"" )
                auclist[epoch]= v_auc
                np_data_full=np.hstack((y_test_com ,y_predicted_com ))
                vaucint=int( v_auc*10000)
                torch.save(model.state_dict() , savepath+"_model_ind_hs_mm_"+Basestr+"_"+ str(batchnum)+"_"+str(vaucint)+"_"+str(epoch)+"_"+str(findseqlen)+ ".pkl")
                saveprobdata(savepath+"_ind_hs_mm_"+Basestr+"_"+ str(batchnum)+"_"+str(vaucint)+"_"+str(epoch)+"_"+str(findseqlen)+"_prob.txt", np_data_full)
                model.train()
                print(   auclist )    
