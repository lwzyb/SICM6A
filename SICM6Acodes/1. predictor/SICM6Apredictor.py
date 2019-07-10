import wx
import wx.xrc
import os 
from SICM6A_Predict import SICM6APredict,SICNet
from SICM6AResult import SICResultFrame 
import torch.nn as nn

import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import pylab as pl

cerevisiaemodelpklfile=""
thalianamodelpklfile=""
Mature_mRNAmodelpklfile=""
Full_transcriptmodelpklfile=""
hepg2_brainmodelpklfile="" 
#95% 90% 85%

hepg2_brain_threshold_high=-0.09196107
hepg2_brain_threshold_midum=-0.14587191
hepg2_brain_threshold_low=-0.19978276

cerevisiaethreshold_high=-0.27075472
cerevisiaethreshold_midum=-0.52439857
cerevisiaethreshold_low=-0.66015166

thalianathreshold_high=-0.1773178
thalianathreshold_midum=-0.59249824
thalianathreshold_low=-0.9193424

Mature_mRNAthreshold_high=-1.1005155
Mature_mRNAthreshold_midum=-1.4586651
Mature_mRNAthreshold_low=-1.7554178

Full_transcriptthreshold_high=-1.052967
Full_transcriptthreshold_midum=-1.6455091
Full_transcriptthreshold_low=-2.186074


        
def readconfig():
    global cerevisiaemodelpklfile 
    global thalianamodelpklfile 
    global Mature_mRNAmodelpklfile 
    global Full_transcriptmodelpklfile 
    global hepg2_brainmodelpklfile
    
    global hepg2_brain_threshold_high 
    global hepg2_brain_threshold_midum
    global hepg2_brain_threshold_low
    
    global cerevisiaethreshold_high
    global cerevisiaethreshold_midum
    global cerevisiaethreshold_low

    global thalianathreshold_high
    global thalianathreshold_midum
    global thalianathreshold_low

    global Mature_mRNAthreshold_high
    global Mature_mRNAthreshold_midum
    global Mature_mRNAthreshold_low

    global Full_transcriptthreshold_high
    global Full_transcriptthreshold_midum
    global Full_transcriptthreshold_low
    
    
    
    fh=open(os.getcwd()+"/config.txt")
     
    for line in fh:
        if line.startswith('cerevisiaemodelpklfile='):
            cerevisiaemodelpklfile=os.getcwd()+"/"+line.replace('cerevisiaemodelpklfile=',"").replace('\n',"")
        elif line.startswith('thalianamodelpklfile='):
            thalianamodelpklfile=os.getcwd()+"/"+line.replace('thalianamodelpklfile=',"").replace('\n',"")
        elif line.startswith('Mature_mRNAmodelpklfile='):
            Mature_mRNAmodelpklfile=os.getcwd()+"/"+line.replace('Mature_mRNAmodelpklfile=',"").replace('\n',"")
        elif line.startswith('hepg2_brainmodelpklfile='):
           hepg2_brainmodelpklfile=os.getcwd()+"/"+line.replace('hepg2_brainmodelpklfile=',"").replace('\n',"")       
        elif line.startswith('Full_transcriptmodelpklfile='):
           Full_transcriptmodelpklfile=os.getcwd()+"/"+line.replace('Full_transcriptmodelpklfile=',"").replace('\n',"")
        elif line.startswith('cerevisiaethreshold_high='):
           cerevisiaethreshold_high=float(line.replace('cerevisiaethreshold_high=',"").replace('\n',"") )
        elif line.startswith('thalianathreshold_high='):
           thalianathreshold_high=float(line.replace('thalianathreshold_high=',"").replace('\n',"") )
        elif line.startswith('Mature_mRNAthreshold_high='):
           Mature_mRNAthreshold_high=float(line.replace('Mature_mRNAthreshold_high=',"").replace('\n',"") )    
        elif line.startswith('Full_transcriptthreshold_high='):
           Full_transcriptthreshold_high=float(line.replace('Full_transcriptthreshold_high=',"").replace('\n',"") )
        elif line.startswith('hepg2_brain_threshold_high='):
           hepg2_brain_threshold_high=float(line.replace('hepg2_brain_threshold_high=',"").replace('\n',"") )          
        elif line.startswith('cerevisiaethreshold_midum='):
           cerevisiaethreshold_midum=float(line.replace('cerevisiaethreshold_midum=',"").replace('\n',"") )
        elif line.startswith('thalianathreshold_midum='):
           thalianathreshold_midum=float(line.replace('thalianathreshold_midum=',"").replace('\n',"") )
        elif line.startswith('Mature_mRNAthreshold_midum='):
           Mature_mRNAthreshold_midum=float(line.replace('Mature_mRNAthreshold_midum=',"").replace('\n',"") )    
        elif line.startswith('Full_transcriptthreshold_midum='):
           Full_transcriptthreshold_midum=float(line.replace('Full_transcriptthreshold_midum=',"").replace('\n',"") )  
        elif line.startswith('hepg2_brain_threshold_midum='):
           hepg2_brain_threshold_midum=float(line.replace('hepg2_brain_threshold_midum=',"").replace('\n',"") )             
        elif line.startswith('cerevisiaethreshold_low='):
           cerevisiaethreshold_low=float(line.replace('cerevisiaethreshold_low=',"").replace('\n',"") )
        elif line.startswith('thalianathreshold_low='):
           thalianathreshold_low=float(line.replace('thalianathreshold_low=',"").replace('\n',"") )
        elif line.startswith('Mature_mRNAthreshold_low='):
           Mature_mRNAthreshold_low=float(line.replace('Mature_mRNAthreshold_low=',"").replace('\n',"") )    
        elif line.startswith('Full_transcriptthreshold_low='):
           Full_transcriptthreshold_low=float(line.replace('Full_transcriptthreshold_low=',"").replace('\n',"") )   
        elif line.startswith('hepg2_brain_threshold_low='):
           hepg2_brain_threshold_low=float(line.replace('hepg2_brain_threshold_low=',"").replace('\n',"") )              
    fh.close()
    

class SICFrame ( wx.Frame ):
    def __init__( self, parent ):
        wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = u"SICM6A for Predicting M6A sites", pos = wx.DefaultPosition, size = wx.Size( 817,821 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )
        ######init##############
        self.threshold=0
        self.fullseqlen=101
        self.findseqlen=101
        self.Basestr="-AGCU"
        self.modefile=""
        ##############
        
        self.SetSizeHintsSz( wx.DefaultSize, wx.DefaultSize )
        fgSizer1 = wx.FlexGridSizer( 5, 1, 0, 0 )
        fgSizer1.SetFlexibleDirection( wx.BOTH )
        fgSizer1.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )
        
        fgSizer1.SetMinSize( wx.Size( 800,600 ) ) 
        self.m_panel16 = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
        fgSizer12 = wx.FlexGridSizer( 2, 1, 0, 0 )
        fgSizer12.SetFlexibleDirection( wx.BOTH )
        fgSizer12.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )
        
        self.m_panel1 = wx.Panel( self.m_panel16, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.NO_BORDER|wx.TRANSPARENT_WINDOW )
        self.m_panel1.SetMinSize( wx.Size( 800,50 ) )
        
        gSizer1 = wx.GridSizer( 1, 1, 0, 0 )
        
        self.m_staticText2 = wx.StaticText( self.m_panel1, wx.ID_ANY, u"SICM6A for Predicting M6A sites", wx.DefaultPosition, wx.Size( 800,45 ), wx.ALIGN_CENTRE )
        self.m_staticText2.Wrap( -1 )
        self.m_staticText2.SetFont( wx.Font( 16, 70, 90, 92, False, "Times New Roman" ) )
        
        gSizer1.Add( self.m_staticText2, 0, wx.ALL, 5 )
        
        self.m_panel1.SetSizer( gSizer1 )
        self.m_panel1.Layout()
        gSizer1.Fit( self.m_panel1 )
        fgSizer12.Add( self.m_panel1, 1, wx.EXPAND, 5 )
        
        self.m_panel2 = wx.Panel( self.m_panel16, wx.ID_ANY, wx.DefaultPosition, wx.Size( 800,30 ), wx.NO_BORDER|wx.TRANSPARENT_WINDOW )
        fgSizer2 = wx.FlexGridSizer( 0, 2, 0, 0 )
        fgSizer2.SetFlexibleDirection( wx.BOTH )
        fgSizer2.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )
        
        fgSizer2.SetMinSize( wx.Size( 800,20 ) ) 
        self.m_staticText4 = wx.StaticText( self.m_panel2, wx.ID_ANY, u"Input your genomic/mRNA sequences with FASTA format.", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_staticText4.Wrap( -1 )
        self.m_staticText4.SetFont( wx.Font( 12, 70, 90, 92, False, "Times New Roman" ) )
        
        fgSizer2.Add( self.m_staticText4, 0, wx.ALL, 5 )
        
        
        self.m_panel2.SetSizer( fgSizer2 )
        self.m_panel2.Layout()
        fgSizer12.Add( self.m_panel2, 1, wx.EXPAND, 5 )
        
        
        self.m_panel16.SetSizer( fgSizer12 )
        self.m_panel16.Layout()
        fgSizer12.Fit( self.m_panel16 )
        fgSizer1.Add( self.m_panel16, 1, wx.EXPAND, 5 )
        
        self.m_panel3 = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.Size( 800,400 ), wx.NO_BORDER|wx.TRANSPARENT_WINDOW )
        self.m_panel3.SetFont( wx.Font( 9, 70, 94, 92, False, "Times New Roman" ) )
        self.m_panel3.SetMinSize( wx.Size( 800,400 ) )
        
        fgSizer4 = wx.FlexGridSizer( 0, 3, 0, 0 )
        fgSizer4.SetFlexibleDirection( wx.BOTH )
        fgSizer4.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )
        
        self.m_txtseq = wx.TextCtrl( self.m_panel3, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 800,450 ), wx.TE_LEFT|wx.TE_MULTILINE )
        fgSizer4.Add( self.m_txtseq, 0, wx.ALL, 5 )
        
        
        self.m_panel3.SetSizer( fgSizer4 )
        self.m_panel3.Layout()
        fgSizer1.Add( self.m_panel3, 1, wx.EXPAND, 5 )
        
        self.m_panel4 = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.NO_BORDER|wx.TRANSPARENT_WINDOW )
        self.m_panel4.SetMinSize( wx.Size( 800,50 ) )
        
        fgSizer7 = wx.FlexGridSizer( 0, 2, 0, 0 )
        fgSizer7.SetFlexibleDirection( wx.BOTH )
        fgSizer7.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )
        
        self.m_staticText21 = wx.StaticText( self.m_panel4, wx.ID_ANY, u"Or upload a file(<=1M):", wx.DefaultPosition, wx.Size( 160,30 ), wx.ALIGN_LEFT )
        self.m_staticText21.Wrap( -1 )
        self.m_staticText21.SetFont( wx.Font( 12, 70, 90, 92, False, "Times New Roman" ) )
        
        fgSizer7.Add( self.m_staticText21, 0, wx.ALL, 5 )
        
        self.m_loadfile = wx.FilePickerCtrl( self.m_panel4, wx.ID_ANY, wx.EmptyString, u"Select a file", u"*.*", wx.DefaultPosition, wx.Size( 550,-1 ), wx.FLP_DEFAULT_STYLE|wx.FLP_FILE_MUST_EXIST|wx.FLP_OPEN )
        self.m_loadfile.SetFont( wx.Font( 10, 70, 90, 92, False, "Times New Roman" ) )
        
        fgSizer7.Add( self.m_loadfile, 0, wx.ALL, 5 )
        
        
        self.m_panel4.SetSizer( fgSizer7 )
        self.m_panel4.Layout()
        fgSizer7.Fit( self.m_panel4 )
        fgSizer1.Add( self.m_panel4, 1, wx.EXPAND, 5 )
        
        self.m_panel5 = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.Size( 800,160 ), wx.NO_BORDER|wx.TRANSPARENT_WINDOW )
        self.m_panel5.SetFont( wx.Font( 14, 70, 94, 92, False, "Times New Roman" ) )
        
        fgSizer41 = wx.FlexGridSizer( 0, 2, 0, 0 )
        fgSizer41.SetFlexibleDirection( wx.BOTH )
        fgSizer41.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )
        
        m_rb_modeChoices = [ u"Full transcript", u"Mature mRNA", u"Arabidopsis thaliana", u"Saccharomyces cerevisiae", u"hepg2_brain" ]
        self.m_rb_mode = wx.RadioBox( self.m_panel5, wx.ID_ANY, u"Mode", wx.DefaultPosition, wx.Size( 400,80 ), m_rb_modeChoices, 1, wx.RA_SPECIFY_COLS )
        self.m_rb_mode.SetSelection( 3 )
        self.m_rb_mode.SetFont( wx.Font( 10, 70, 94, 92, False, "Times New Roman" ) )
        
        fgSizer41.Add( self.m_rb_mode, 0, wx.ALL|wx.EXPAND, 5 )
        
        m_rb_mode1Choices = [ u"High(95%)", u"Midum(90%)", u"Low(85%)" ]
        self.m_rb_threshhold = wx.RadioBox( self.m_panel5, wx.ID_ANY, u"Threshold", wx.DefaultPosition, wx.Size( 400,160 ), m_rb_mode1Choices, 1, wx.RA_SPECIFY_COLS )
        self.m_rb_threshhold.SetSelection( 0 )
        self.m_rb_threshhold.SetFont( wx.Font( 10, 70, 94, 92, False, "Times New Roman" ) )
        
        fgSizer41.Add( self.m_rb_threshhold, 0, wx.ALL|wx.EXPAND, 5 )
        
        
        self.m_panel5.SetSizer( fgSizer41 )
        self.m_panel5.Layout()
        fgSizer1.Add( self.m_panel5, 1, wx.EXPAND, 5 )
        
        self.m_panel111 = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.NO_BORDER|wx.TRANSPARENT_WINDOW )
        self.m_panel111.SetMinSize( wx.Size( 800,60 ) )
        
        fgSizer71 = wx.FlexGridSizer( 1, 5, 0, 0 )
        fgSizer71.SetFlexibleDirection( wx.BOTH )
        fgSizer71.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )
        
        self.m_staticText5 = wx.StaticText( self.m_panel111, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 150,-1 ), 0 )
        self.m_staticText5.Wrap( -1 )
        fgSizer71.Add( self.m_staticText5, 0, wx.ALL, 5 )
        
        self.m_bt_submit = wx.Button( self.m_panel111, wx.ID_ANY, u"Submit", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_bt_submit.SetFont( wx.Font( 12, 70, 90, 92, False, "Times New Roman" ) )
        
        fgSizer71.Add( self.m_bt_submit, 0, wx.ALL, 5 )
        
        self.m_bt_reset = wx.Button( self.m_panel111, wx.ID_ANY, u"Reset", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_bt_reset.SetFont( wx.Font( 12, 70, 90, 92, False, "Times New Roman" ) )
        
        fgSizer71.Add( self.m_bt_reset, 0, wx.ALL, 5 )
        
        self.m_bt_exit = wx.Button( self.m_panel111, wx.ID_ANY, u"Exit", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_bt_exit.SetFont( wx.Font( 12, 70, 90, 92, False, "Times New Roman" ) )
        
        fgSizer71.Add( self.m_bt_exit, 0, wx.ALL, 5 )
        
        self.m_example = wx.Button( self.m_panel111, wx.ID_ANY, u"Example", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_example.SetFont( wx.Font( 12, 70, 90, 92, False, "Times New Roman" ) )
        
        fgSizer71.Add( self.m_example, 0, wx.ALL, 5 )
        
        
        self.m_panel111.SetSizer( fgSizer71 )
        self.m_panel111.Layout()
        fgSizer71.Fit( self.m_panel111 )
        fgSizer1.Add( self.m_panel111, 1, wx.EXPAND, 5 )
        
        
        self.SetSizer( fgSizer1 )
        self.Layout()
        
        self.Centre( wx.BOTH )
        
        # Connect Events
        self.m_example.Bind( wx.EVT_BUTTON, self.onbtexample )
  
        self.m_rb_threshhold.Bind( wx.EVT_KEY_DOWN, self.onrbthreshholdclicked )
        self.m_rb_mode.Bind( wx.EVT_KEY_DOWN, self.onrbmodeclicked )
        self.m_bt_submit.Bind( wx.EVT_BUTTON, self.onbtsubmit )
        self.m_bt_reset.Bind( wx.EVT_BUTTON, self.onbtreset )
        self.m_bt_exit.Bind( wx.EVT_BUTTON, self.onbtexit )
        
        icon = wx.Icon("puzzle.ico", wx.BITMAP_TYPE_ICO)
        self.SetIcon(icon)

        
        readconfig()
        #print(cerevisiaemodelpklfile)
    
    def __del__( self ):
        pass
    
    # Virtual event handlers, overide them in your derived class
    def onbtexample( self, event ):
        """
        dlg = wx.MessageDialog(None, u"消息对话框测试", u"标题信息", wx.YES_NO | wx.ICON_QUESTION) 
        if dlg.ShowModal() == wx.ID_YES: 
            self.Close(True) 
            dlg.Destroy() 
        """
        strexample=">test\nAUUGUCAAAAUGUCUGGAACAGAAUACUCAAAUUGACUAGUUCUGGCCUUUUCCCUUAAAUGGGCACGAGUAGGAGCAACAGACUACAUCAUCACUAUCUCUAGAGAAAUAGAUCUUGCGAGAGAAAAAAACGUUGGUUGGUCUGCUUUUGGCUCUUUUGUCAAUUAAAUCCCCGGAUGUACCUCAAAAAGACUGUAAAAGACUGGCUGGUGGACUAACGAUGGCUUUCCUCAGCAGAAAGGAGGGAGAAAAAAAAUUCAACUGGAACAUCCAAAAGCGUUGAAAUUCUUUGUGGGCAAUAUACAUGAGAUGGCUCCUAAGGAAUAUAAGGAGGUUGAAAAGAAGUUUCAUCUGUACCAGUGUUUCUGCAAUCCAGUUGGAGAGAACAGAUUAUGACUUAUCUAGUGGUGAGAUCGCACAGCCCAACUAUGCGACCUCCGAAGCACAUUUUCCCAGAAUCUUUCCCUCAAGCACCUGGAACCUGGAACCCCCGAAGAAGAAUGCUUGUCACGAUGCGGCAAGGGAACACCUUUGGAAGGAGUAGAUCUAUUUUUUUAUUUUUGAAUUUUUGGGACUGUUGACCUUGCCUGCCUGAGAGCAAAAGAGAGACAACGACUGAGCAAGCACUACCACCAGACACUGUUACUGGCGAAUUAGAAGACCUGAUGUUUCGUUGGUCCUAGACCCUCAGUGCAAACCAUCGAGGAUGACUCCAUAUCCCACAACCGUGAACUUAUGUCUCUGUGCCUCUCUGAAUUUGCCGUGUAGUGGCUUCAGCCUGGACACCUUGGCUAGACUGCAUACCCUGUCCCUUGAGUGCUAUUCCUAUCAGACACUAUCAGAACAGUGCAUUCCUCCAGAAUUGGGAUAGUAGCCAGGACAGAACAAUUUCCUAGUCGCGACCCUACACGGCUGAACACGGACUUGACCCGUUUACUAAGCAAACAACCGAGAUGGGCUGCACUUACUGGUACAUUGAAGCUACAUCGACUGUUGUCGAAAUGUUCUUGACCACUCCGCAAUUGACACUACCUGAAUAGAUAUCAGCAACCGCGUGUGAGGUUUGUGGAAGGACGUCUGGUGCUCCUCAUUGAGGGCCAAUAUUAGUAGCUAAGCGUUCGCGUAGCCAUCUGAUCGAGCCUCUUGUUAUCAGUUCCAAUGGUAACUCUUACUUCGUGUUUGCAAGAGACUAACUAAUGAAAGACCCAAAAUGUCUGCCAAAUACUGGGCCAAGCAUUGUUCCUGAUAAGGGGACUCGAGCAUAUCUAUCAUUGCGGUCUUCAUUUUCUCACAUACACUUACGACAUCGAACUAGGGUAUGCAGCUACAACACGCCCAGUUAGCAGUGUAUAUAUUGCGACGACCGUUGCGUUAUUUACCUUUGCAGCCUUUAACUUACAUGUAUUGCGCAAAGAUAAAUGCAGCUAAUAAGUCGUGUCUUAAAAAAGAAAAAAAAAAAAAAAGUGCUGGGAAUCUCUUCUUCUUCGUUCCGAUACCACCCUUGGCUAUCAUCUGCGCUAUUACUGCCCAGAUAUGUAUUAUGAGCUAAUUCAUUUAGCUCAGCUUUCUUAAUCAGUUUGAAACUAUCCCCGAUAGCGACAAUCUUUGAACACCCCUCCUUCAUCAUUGUGCAGUUUAAAAAUGUAUAACCAUUUGGAAAAAGUUUUGGUCUUGAGCUCAGCCAGUAACUAGAUGUUUUUUUCUCUUUGAAUUUGCUCUGCCCCUUGUUGGCCGCAGAGUUAUGUUCUAUUUUAAACGAUGAAUCUUU\n"
        self.m_txtseq.SetValue(strexample)

    
    def onrbfromatclicked( self, event ):
        event.Skip()
    
    def onrbformatbox( self, event ):
        event.Skip()
    
    def onrbthreshholdclicked( self, event ):
        event.Skip()
    
    def onrbmodeclicked( self, event ):
        event.Skip()
    
    def onbtsubmit( self, event ):
      
        #print(111,seqdictfull)

        #print(hepg2_brainmodelpklfile)
        
        if self.m_rb_mode.GetSelection() ==4:
            self.fullseqlen=101
            self.findseqlen=101
            self.Basestr="-AGCU"        
            self.modefile= hepg2_brainmodelpklfile


            
        elif  self.m_rb_mode.GetSelection() ==3:           
            self.fullseqlen=51
            self.findseqlen=51
            self.Basestr="-AGCU"  
            self.modefile=  cerevisiaemodelpklfile

    
            
        elif    self.m_rb_mode.GetSelection() ==2:
          
            self.fullseqlen=101
            self.findseqlen=101
            self.Basestr="-AGCU"
            self.modefile=   thalianamodelpklfile      

      
            
        elif  self.m_rb_mode.GetSelection()==1:
            
            self.fullseqlen=251
            self.findseqlen=251
            self.Basestr="-AGCU"
            self.modefile=   Mature_mRNAmodelpklfile        
    
            
        elif self.m_rb_mode.GetSelection() ==0:
          
            self.fullseqlen=501
            self.findseqlen=501
            self.Basestr="-AGCU"
            self.modefile=  Full_transcriptmodelpklfile
 
        
        
        
        
        
        if self.m_rb_threshhold.GetSelection()==0 and  self.m_rb_mode.GetSelection() ==4:
            self.threshold=hepg2_brain_threshold_high
   

        elif self.m_rb_threshhold.GetSelection()==1 and  self.m_rb_mode.GetSelection() ==4:
            self.threshold=hepg2_brain_threshold_midum
     
            
        elif self.m_rb_threshhold.GetSelection()==2 and  self.m_rb_mode.GetSelection() ==4:
            self.threshold=hepg2_brain_threshold_low
   
            
        if self.m_rb_threshhold.GetSelection()==0 and  self.m_rb_mode.GetSelection() ==3:
            self.threshold=cerevisiaethreshold_high
   

        elif self.m_rb_threshhold.GetSelection()==1 and  self.m_rb_mode.GetSelection() ==3:
            self.threshold=cerevisiaethreshold_midum
   
            
        elif self.m_rb_threshhold.GetSelection()==2 and  self.m_rb_mode.GetSelection() ==3:
            self.threshold=cerevisiaethreshold_low
   
            
        if self.m_rb_threshhold.GetSelection()==0 and  self.m_rb_mode.GetSelection() ==2:
            self.threshold=thalianathreshold_high
 

        elif self.m_rb_threshhold.GetSelection()==1 and  self.m_rb_mode.GetSelection() ==2:
            self.threshold=thalianathreshold_midum
   
            
        elif self.m_rb_threshhold.GetSelection()==2 and  self.m_rb_mode.GetSelection() ==2:
            self.threshold=thalianathreshold_low    
      
            
        if self.m_rb_threshhold.GetSelection()==0 and  self.m_rb_mode.GetSelection()==1:
            self.threshold=Mature_mRNAthreshold_high
         
        
        elif self.m_rb_threshhold.GetSelection()==1 and  self.m_rb_mode.GetSelection() ==1:
            self.threshold=Mature_mRNAthreshold_midum
        
            
        elif self.m_rb_threshhold.GetSelection()==2 and  self.m_rb_mode.GetSelection() ==1:
            self.threshold=Mature_mRNAthreshold_low 
        
            
        if self.m_rb_threshhold.GetSelection()==0 and  self.m_rb_mode.GetSelection() ==0:
            self.threshold=Full_transcriptthreshold_high
         
 
        elif self.m_rb_threshhold.GetSelection()==1 and  self.m_rb_mode.GetSelection() ==0:
            self.threshold=Full_transcriptthreshold_midum
    
             
        elif self.m_rb_threshhold.GetSelection()==2 and  self.m_rb_mode.GetSelection() ==0:
            self.threshold=Full_transcriptthreshold_low    
                       
   
        
        
        
        seqdictfull,labeldictfull=self.buildshortseq()   
        SICPredict=SICM6APredict(self.modefile,self.threshold,  self.fullseqlen,self.findseqlen,self.Basestr)
        ss=SICPredict.predict(seqdictfull,labeldictfull)
        #print(222,ss)
        frame = SICResultFrame(None,ss)
        frame.Show(True)

        
    def onbtreset( self, event ):
        self.resettxt()
    
    def onbtexit( self, event ):
        self.Close(True)
        #dlg = wx.MessageDialog(None, u"消息对话框测试", u"标题信息", wx.YES_NO | wx.ICON_QUESTION) 
        #if dlg.ShowModal() == wx.ID_YES:
        #    self.Close(True) 
        #    dlg.Destroy() 
        #self.Close(True) 
        
        
        #self.Destroy()
        #wx.Exit()
    
    def resettxt(self):
        self.m_txtseq.Clear()
        #self.m_loadfile.
        
    def buildshortseq(self):
        
        v1=self.m_txtseq.GetValue() 
        v2=self.m_loadfile.GetPath()
        if len(v1)==0 and len(v2)==0:
            dlg2 = wx.MessageDialog(None, u"Need to input sequence with fasta format!",u"Information",  wx.YES_NO  )
            if dlg2.ShowModal() == wx.ID_YES:
                #self.Close(True)
                pass
            dlg2.Destroy()
            return
        #print(v2)
        if len(v1)>0: 
            seqdict=self.readfastatxt(v1)
        elif len(v2)>0: 
            seqdict=self.readfastafile(v2)
        #print(len(seqdict))
        seqdictfull=dict()
        labeldictfull=dict()
        flag=""
        #print(self.m_rb_mode.Selection )
        if self.m_rb_mode.Selection ==3:
            flag="GAC"
        else:
            flag="DRACH"
         
        for key in seqdict :
            seq=seqdict[key]
            if seq=="":
                break
            #print(key,seq)
            seqdictfull,labeldictfull=self.findm6ashortseq(seqdictfull,labeldictfull, key,seq, flag, self.fullseqlen)
        #print(0000,seqdictfull)
        return seqdictfull,labeldictfull 
            
    def checkfastafilesize(self,fastafile):
        size = os.path.getsize(fastafile)
        size_in_Mb     = size/(1024*1024)
        if size_in_Mb >1:
            print(size_in_Mb,"Error!File size cannot exceed 2M!")
            dlg = wx.MessageDialog(None, u"File size cannot exceed 1M!", u"Error", wx.YES_NO | wx.ICON_QUESTION)
            if dlg.ShowModal() == wx.ID_YES: 
                dlg.Destroy()
                return -1
        return 1
        
                
    def checkfastatxtize(self,fastatxt):
        len1 =len(fastatxt)
        
        if len1 >100000:
            print(len1,"Error! The number of characters cannot exceed 100,000!")
            dlg = wx.MessageDialog(None, u"The number of characters cannot exceed 100,000!", u"Error", wx.YES_NO | wx.ICON_QUESTION)
            if dlg.ShowModal() == wx.ID_YES: 
                dlg.Destroy()
                return -1
        return 1
                    
            
    def readfastafile(self,fastafile):
        if self.checkfastafilesize( fastafile)==-1:
            return dict()
            
        fh=open(fastafile)
        seq=dict()
        for line in fh:
            if line.startswith('>'):
                name=line.repalce('>','').split()[0]
                seq[name]=''
            else:
                seq[name]+=line.replace('\n')
        fh.close()
        return seq
    
    def readfastatxt(self,fastatxt):
        if self.checkfastatxtize( fastatxt)        ==-1:
            return dict()
        seqdict=dict()
        ss=fastatxt.split("\n")
        
        
        
        for i in range(len(ss)):
            line=ss[i]
            #print(line)
            if line=="":
                break
            if line.startswith('>'):
                name=line [1:len(line)]
                #print(name)
                seqdict[name]=''
            else:
                seqdict[name]+=line 
         
        return seqdict
       
        
        
        
    
    #DRACH (where D = A, G or U; R = A or G; and H = A, C or U)  motifs or GAC 
    #drachset=set(["",])
    def findm6ashortseq(self,seqdictfull,labeldictfull,key,seq,flag,cutlen):
        #print(seq)
        leftcutlen=int((cutlen -1)/2)
        rightcutlen=int((cutlen -1)/2)
        len1=len(seq)
        seqdict=seqdictfull
        labeldict=labeldictfull
        for i in range(len1):
            kk=i + 1
            s_i=seq [i:kk]
            leftseq=""
            rightseq=""
            if s_i != "A":
                continue
            if flag=="DRACH":
                if i<=1:
                    continue
                s_i_l1=seq[i - 1:i]
                if s_i_l1 != "G" and s_i_l1 != "A":
                    continue
                s_i_l2=seq[i - 2:i - 1]
                if s_i_l2 != "G" and s_i_l2 != "A"  and s_i_l2 != "U":
                    continue
                s_i_r1=seq[i + 1 :i + 2]
                if s_i_r1 != "C"  :
                    continue
                s_i_l2=seq[i + 2:i +3  ]
                if s_i_l2 != "C" and s_i_l2 != "A"  and s_i_l2 != "U":
                    continue   
                
                
                #OK
            elif flag=="GAC":
                if i<=0:
                    continue
                s_i_l1=seq[i - 1:i]
                if s_i_l1 != "G":
                    continue
                s_i_r1=seq[i+1:i + 2]
                if s_i_r1 != "C":
                    continue
                #ok
            if leftcutlen > i   :# the len is short  
                leftseq=seq[0:i]
                leftseq=leftseq.rjust(leftcutlen,"-")
            else:
                #print(len1,leftcutlen,i,i-leftcutlen)
                leftseq=seq[i   - leftcutlen :i]
            if len1 - rightcutlen < i +1:# the len is short 6
                rightseq=seq[i + 1: len1]
                rightseq=rightseq.ljust(rightcutlen,"-")
            else:
                rightseq=seq[i +1    :i + 1+ rightcutlen]
                
            shortseq=leftseq+s_i+rightseq
            seqkey=key+"_"+str(i)
            seqdict[seqkey]=shortseq 
            labeldict[seqkey]=1
            
        return seqdict,labeldict
                
                
    
    
    
        
if __name__ == u'__main__': 
    app = None
    app = wx.App()
    frame = SICFrame(None)
    frame.Show(True)
    app.MainLoop()
    #

