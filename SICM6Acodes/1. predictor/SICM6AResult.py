import wx
import wx.xrc
import os 
 
    

class SICResultFrame ( wx.Dialog):
    def __init__( self, parent,result ):
        wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = u"Result of M6A sites prediction", pos = wx.DefaultPosition, size = wx.Size( 817,641 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )
		
        self.SetSizeHintsSz( wx.DefaultSize, wx.DefaultSize )
		
        fgSizer1 = wx.FlexGridSizer( 5, 1, 0, 0 )
        fgSizer1.SetFlexibleDirection( wx.BOTH )
        fgSizer1.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )
		
        fgSizer1.SetMinSize( wx.Size( 800,600 ) ) 
        self.m_panel16 = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
        self.m_panel16.SetMinSize( wx.Size( -1,50 ) )
		
        fgSizer12 = wx.FlexGridSizer( 2, 1, 0, 0 )
        fgSizer12.SetFlexibleDirection( wx.BOTH )
        fgSizer12.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )
		
        self.m_panel1 = wx.Panel( self.m_panel16, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.NO_BORDER|wx.TRANSPARENT_WINDOW )
        self.m_panel1.SetMinSize( wx.Size( 800,50 ) )
		
        gSizer1 = wx.GridSizer( 1, 1, 0, 0 )
		
        self.m_staticText2 = wx.StaticText( self.m_panel1, wx.ID_ANY, u"Result of M6A sites prediction", wx.DefaultPosition, wx.Size( 800,45 ), wx.ALIGN_CENTRE )
        self.m_staticText2.Wrap( -1 )
        self.m_staticText2.SetFont( wx.Font( 16, 70, 90, 92, False, "Times New Roman" ) )
		
        gSizer1.Add( self.m_staticText2, 0, wx.ALL, 5 )
		
		
        self.m_panel1.SetSizer( gSizer1 )
        self.m_panel1.Layout()
        gSizer1.Fit( self.m_panel1 )
        fgSizer12.Add( self.m_panel1, 1, wx.EXPAND, 5 )
		
		
        self.m_panel16.SetSizer( fgSizer12 )
        self.m_panel16.Layout()
        fgSizer12.Fit( self.m_panel16 )
        fgSizer1.Add( self.m_panel16, 1, wx.EXPAND, 5 )
		
        self.m_panel3 = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.Size( 800,500 ), wx.NO_BORDER|wx.TRANSPARENT_WINDOW )
        self.m_panel3.SetFont( wx.Font( 14, 70, 94, 92, False, "Times New Roman" ) )
        self.m_panel3.SetMinSize( wx.Size( 800,500 ) )
		
        fgSizer4 = wx.FlexGridSizer( 0, 3, 0, 0 )
        fgSizer4.SetFlexibleDirection( wx.BOTH )
        fgSizer4.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )
		
        self.m_txtseq = wx.TextCtrl( self.m_panel3, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 800,500 ), wx.TE_LEFT|wx.TE_MULTILINE )
        self.m_txtseq.SetFont( wx.Font( 8, 74, 90, 90, False, "Arial"  ) )
		
        fgSizer4.Add( self.m_txtseq, 0, wx.ALL, 5 )
		
		
        self.m_panel3.SetSizer( fgSizer4 )
        self.m_panel3.Layout()
        fgSizer1.Add( self.m_panel3, 1, wx.EXPAND, 5 )
		
        self.m_panel111 = wx.Panel( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.NO_BORDER|wx.TRANSPARENT_WINDOW )
        self.m_panel111.SetMinSize( wx.Size( 800,50 ) )
		
        fgSizer71 = wx.FlexGridSizer( 1, 5, 0, 0 )
        fgSizer71.SetFlexibleDirection( wx.BOTH )
        fgSizer71.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )
		
        self.m_staticText5 = wx.StaticText( self.m_panel111, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 300,-1 ), 0 )
        self.m_staticText5.Wrap( -1 )
        fgSizer71.Add( self.m_staticText5, 0, wx.ALL, 5 )
		
        self.m_bt_export = wx.Button( self.m_panel111, wx.ID_ANY, u"Export", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_bt_export.SetFont( wx.Font( 12, 70, 90, 92, False, "Times New Roman" ) )
		
        fgSizer71.Add( self.m_bt_export, 0, wx.ALL, 5 )
		
        self.m_bt_exit = wx.Button( self.m_panel111, wx.ID_ANY, u"Exit", wx.DefaultPosition, wx.DefaultSize, 0 )
        self.m_bt_exit.SetFont( wx.Font( 12, 70, 90, 92, False, "Times New Roman" ) )
		
        fgSizer71.Add( self.m_bt_exit, 0, wx.ALL, 5 )
		
		
        self.m_panel111.SetSizer( fgSizer71 )
        self.m_panel111.Layout()
        fgSizer71.Fit( self.m_panel111 )
        fgSizer1.Add( self.m_panel111, 1, wx.EXPAND, 5 )
		
		
        self.SetSizer( fgSizer1 )
        self.Layout()
		
        self.Centre( wx.BOTH )
 
		
        # Connect Events
        self.m_bt_export.Bind( wx.EVT_BUTTON, self.onbtsubmit )
        self.m_bt_exit.Bind( wx.EVT_BUTTON, self.onbtexit )
        
        self.result=result
        self.m_txtseq.SetValue(self.result)
        
    def __del__( self ):
        pass
	
	
    # Virtual event handlers, overide them in your derived class
    def onbtsubmit( self, event ):
        filesFilter = "TXT file (*.txt)|*.txt|" "All files (*.*)|*.*"
        fileDialog = wx.FileDialog(self, message ="保存文件", wildcard = filesFilter, style = wx.FD_SAVE)
        dialogResult = fileDialog.ShowModal()
        if dialogResult !=  wx.ID_OK:
            return


 
        fpath = fileDialog.GetPath()
        if  os.path.exists(fpath):
            dlg1 = wx.MessageDialog(None, u"The file already exists. Do you want to cover the documents?",u"Notification",  wx.YES_NO | wx.ICON_QUESTION)
            if dlg1.ShowModal() != wx.ID_YES:
                #self.Close(True)
                dlg1.Destroy()
                return
            dlg1.Destroy()
        
        
        target = open(fpath, 'a')
        s1=self.result
        target.write(s1)
        target.close()
        
        dlg2 = wx.MessageDialog(None, u"Successfully export information!",u"Information",  wx.YES_NO  )
        if dlg2.ShowModal() == wx.ID_YES:
            #self.Close(True)
            pass
        dlg2.Destroy()
         
 
	
    def onbtexit( self, event ):
        self.Close(True)
                
                
    
    
    
        
if __name__ == u'__main__': 
    app = None
    app = wx.App()
    frame = SICResultFrame(None,"")
    frame.Show(True)
    app.MainLoop()
    #

