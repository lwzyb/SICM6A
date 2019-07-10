"""
Created on Wed Oct 31 12:14:44 2018

@author: liuwenzhong
"""
import torch.nn as nn 
import numpy as np 
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import pylab as pl 

 
torch.nn.Module.dump_patches = True    
  
class SICM6APredict(object): 
    def __init__(self, modelpklfile, threshold,fullseqlen,findseqlen,Basestr):
        self.seq_dict_test=dict()
        self.seq_label_test=dict()
        self.testlist_dict=dict()
 
        self.threekmercode=dict()
        self.modelpklfile=modelpklfile
        self.threshold=threshold
        self.fullseqlen=fullseqlen
        self.findseqlen=findseqlen
        self.Basestr=Basestr
      
     
     
    def ROC( self,y_test,y_predicted):
        y = y_test#np.array(y_test)
        pred =y_predicted# np.array(y_predicted)
        fpr, tpr, thresholds = roc_curve(y, pred, pos_label=1)
        #print(fpr)
        #print(tpr)
        #print(thresholds)
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
    
    

    
    
    def buildthreekmercode( self):
        encode=dict()
        len1=len(self.Basestr )
        for i in range(len1):
            s1=self.Basestr[i:i+1]
            for j in range(len1):  
                s2=self.Basestr[j:j+1]
                for k in range(len1):
                    s3=self.Basestr[k:k+1]
                    ss=s1+s2+s3
                    kk=int(str(i)+str(j)+str(k))
                    encode[ss]=kk
         
        return encode 
    
     
    
     



    def predict(self,seq_dict_test , seq_label_test  ):
        #print(seq_dict_test)
 
        self.threekmercode=self.buildthreekmercode()
         
        img_test=SeqDataset(seq_dict_test , seq_label_test,  self.fullseqlen, self.findseqlen, self.threekmercode )
     
        test_loader= DataLoader(dataset= img_test, batch_size=10, shuffle=False)
        
        model =SICNet(self.findseqlen)   
        model.load_state_dict(torch.load(self.modelpklfile,map_location='cpu'))
        #model=torch.load(self.modelpklfile,map_location='cpu')
        # evaluation--------------------------------
        model .eval()
     
     
        y_predicted_com =[]
     
        for batch_x, batch_y in test_loader:
     
     
            out  = model (batch_x )
            
            y_predicted= out.detach().numpy() [:,1]
            len1=len( batch_y) 
            y_predicted=y_predicted.reshape((len1,1))
            if len(y_predicted_com)<=0:
                y_predicted_com=y_predicted
            else:
                y_predicted_com=np.vstack((y_predicted_com,y_predicted)  ) 
        
        #print(y_predicted_com)
        score_predict=np.asarray(y_predicted_com)
        #print( score_predict)
        label_predict=np.int64(score_predict>=self.threshold)  
        
        #build prediction string
        predictionstring="name_position\tshort_sequence\tscore\tpredicted_label\n"
        kk=0
        for key in seq_dict_test:
            seq=seq_dict_test[key]
            predictionstring +=key+"\t"+seq+"\t"+str(score_predict[kk][0])+"\t"+str(label_predict[kk][0])+"\n"
            kk+=1
            
        return predictionstring
        
      
       

    
class SeqDataset(Dataset):
 
    def __init__(self,  seq_dict , seq_label , fullseqlen, findseqlen,threekmercode1): 
       
        self.threekmercode=threekmercode1
        self.fullseqlen=fullseqlen
        self.findseqlen=findseqlen        
        self.imgs = self.buildseqcode( seq_dict , seq_label  ) 

        
       
    def buildseqcode(self, seq_dict , seq_label  ):     
      
        imgs = []
        for key in  seq_dict.keys():
            seq= seq_dict[key]
           
            mat= self.buildseqqtbin( seq)
             
            label=seq_label[key]
            imgs.append((mat,label))
     
        return imgs 
     
    def buildseqqtbin( self,seq):
        seq=seq.strip()
        lefthalf=int((self.fullseqlen-self.findseqlen)/2)
        righthalf= self.fullseqlen-lefthalf
        seq=seq[lefthalf:righthalf]
        len1=len(seq)     
 
        mat    = np.zeros((self.findseqlen ), dtype=np.longlong)       
        i=0  
        
        while i <len1 :#3-mer  
           
            basestr=""
            if i<len1-2:
                basestr=seq[i:i+3 ]
            elif i==len1-2:
                basestr=seq[i:i+2 ]+"-"
            else:
                basestr=seq[i:i+1 ]+"--"
            bin_k=self.threekmercode[basestr]
            
        
            mat [i]=bin_k 
          
            i +=1
            
   
     
        return mat 
    
    def __getitem__(self, index):
        img, label = self.imgs[index]
        return img,label
       

    def __len__(self):
        return len(self.imgs)
    
       
class SICNet(nn.Module):

   def __init__(self,findseqlen ):   
  
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

    
