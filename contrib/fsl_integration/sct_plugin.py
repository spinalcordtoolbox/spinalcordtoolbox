# coding=utf-8

#########################################################################################
# This code will provide SCT integration into fsleyes for the following tools:
#
#    - sct_propseg
#    - sct_deepseg_gm
#    - sct_deepseg_sc
#    - sct_label_vertebrae
#    - sct_register_to_template
#    - sct_process_segmentation
#    - sct_dmri_moco
#    - sct_dmri_compute_dti
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2018 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Christian S. Perone
#          Thiago JR Rezende
# Created: 2 Mar 2018
#
#########################################################################################

import os
import subprocess
from threading import Thread

import wx
import wx.lib.agw.aui as aui
import wx.html as html

aui_manager = frame.getAuiManager()


class SCTCallThread(Thread):
    def __init__(self, command):
        Thread.__init__(self)
        self.command = command

    @staticmethod
    def sct_call(command):
        env = os.environ.copy()
        del env["PYTHONHOME"]
        del env["PYTHONPATH"]
        p = subprocess.Popen([command], stdout=subprocess.PIPE,
                             shell=True, env=env)
        stdout, stderr = p.communicate()
        print(stdout)
        print(stderr)
        return stdout, stderr

    def run(self):
        self.sct_call(self.command)


class ProgressDialog(wx.Dialog):
    def __init__(self, parent):
        wx.Dialog.__init__(self, parent, title="SCT / Processing")
        self.SetSize((350, 100))
        self.btns = self.CreateSeparatedButtonSizer(wx.CANCEL)

        save_ico = wx.ArtProvider.GetBitmap(wx.ART_INFORMATION,
                                            wx.ART_TOOLBAR,
                                            (16, 16))
        img_info = wx.StaticBitmap(self, -1, save_ico, wx.DefaultPosition,
                                   (save_ico.GetWidth(), save_ico.GetHeight()))

        sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.lbldesc = wx.StaticText(self, id=wx.ID_ANY,
                                     label="Please wait while the algorithm is running...")
        sizer.Add(img_info, 0, wx.ALL, 15)
        sizer.Add(self.lbldesc, 0, wx.ALL, 15)
        sizer_c = wx.BoxSizer(wx.VERTICAL)
        sizer_c.Add(sizer)
        sizer_c.Add(self.btns, 0, wx.ALL, 5)

        self.SetSizer(sizer_c)

        self.Centre()
        self.CenterOnParent()

#Creates the standard panel for each tool
class SCTPanel(wx.Panel):
    DESCRIPTION_SCT = """
    <b>General citation (please always cite)</b>:<br>
    De Leener B, Levy S, Dupont SM, Fonov VS, Stikov N, Louis Collins D, Callot V,
    Cohen-Adad J. <i>SCT: Spinal Cord Toolbox, an open-source software for processing
    spinal cord MRI data</i>. Neuroimage. 2017 Jan 15;145(Pt A):24-43.
    """

    SCT_DIR_ENV = 'SCT_DIR'
    SCT_LOGO_REL_PATH = 'documentation/imgs/logo_sct.png'
    SCT_TUTORIAL_PATH = 'documentation/Manual_v1_SCT.pdf'

    def __init__(self, parent, id_):
        super(SCTPanel, self).__init__(parent=parent,
                                       id=id_)
        self.img_logo = self.get_logo()
        self.html_desc = self.get_description()

        button_help = wx.Button(self, id=id_, label="Help")
        button_help.Bind(wx.EVT_BUTTON, self.tutorial)

        self.sizer_logo_sct = wx.BoxSizer(wx.VERTICAL)
        self.sizer_logo_sct.Add(self.img_logo, 0, wx.ALL, 5)

        txt_sct_citation = wx.VSCROLL | \
                           wx.HSCROLL | wx.TE_READONLY | \
                           wx.BORDER_SIMPLE
        html_sct_citation = html.HtmlWindow(self, wx.ID_ANY,
                                            size=(280, 115),
                                            style=txt_sct_citation)
        html_sct_citation.SetPage(self.DESCRIPTION_SCT)

        self.sizer_logo_sct.Add(html_sct_citation, 0, wx.ALL, 5)
        self.sizer_logo_sct.Add(button_help, 0, wx.ALL, 5)

        self.sizer_logo_text = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer_logo_text.Add(self.sizer_logo_sct, 0, wx.ALL, 5)
        self.sizer_logo_text.Add(self.html_desc, 0, wx.ALL, 5)

        self.sizer_h = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer_h.Add(self.sizer_logo_text)


    def tutorial(self,event):
        pdfpath = os.path.join(os.environ[self.SCT_DIR_ENV],self.SCT_TUTORIAL_PATH)
        print('PDF path:', pdfpath)
        cmd_line = "open {}".format(pdfpath)
        print('Command line:', cmd_line)
        self.call_sct_command(cmd_line)


    def get_logo(self):
        logo_file = os.path.join(os.environ[self.SCT_DIR_ENV],
                                 self.SCT_LOGO_REL_PATH)
        png = wx.Image(logo_file,
                       wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        png.SetSize((png.GetWidth() // 6, png.GetHeight() // 6))
        img_logo = wx.StaticBitmap(self, -1, png, wx.DefaultPosition,
                                   (png.GetWidth(), png.GetHeight()))
        return img_logo

    def get_description(self):
        txt_style = wx.VSCROLL | \
                    wx.HSCROLL | wx.TE_READONLY | \
                    wx.BORDER_SIMPLE
        htmlw = html.HtmlWindow(self, wx.ID_ANY,
                                size=(280, 208),
                                style=txt_style)
        htmlw.SetPage(self.DESCRIPTION)
        return htmlw

    def call_sct_command(self, command):
        disable_window = wx.WindowDisabler()
        binfo = ProgressDialog(frame)
        binfo.Show()

        thr = SCTCallThread(command)
        thr.start()

        # No access to app.pending() from here
        while True:
            thr.join(0.1)
            wx.Yield()
            if not thr.isAlive():
                break
        thr.join()

        binfo.Destroy()

# Run Spinal Cord segmentation using propseg
class TabPanelPropSeg(SCTPanel):
    DESCRIPTION = """This segmentation tool automatically segment the spinal cord with
    robustness, accuracy and speed.<br><br>
    <b>Usage</b>:<br>
    To launch the script, the upload the raw image (t1, t2, t2s and dwi) into FSLeyes and always keeping it
    as the first in the Overlay list field from the bottom to the top. If you uploaded more then one image, it is not necessary
    uploading the images in such order, with the arrows is possible to sort them and only the first imaging will be used. 
    For more information, please refer to the article below.<br><br>
    <b>Specific citation</b>:<br>
    De Leener et al.
    <i>Robust, accurate and fast automatic segmentation of the spinal cord.
    (2014)</i>. Neuroimage. 

    """

    def __init__(self, parent):
        super(TabPanelPropSeg, self).__init__(parent=parent,
                                            id_=wx.ID_ANY)
        button_gm = wx.Button(self, id=wx.ID_ANY, label="Spinal Cord Segmentation")
        button_gm.Bind(wx.EVT_BUTTON, self.onButtonSC)

        lbl_contrasts = ['t1', 't2', 't2s', 'dwi']
        self.rbox_contrast = wx.RadioBox(self, label='Select contrast:',
                                         choices=lbl_contrasts,
                                         majorDimension=1,
                                         style=wx.RA_SPECIFY_ROWS)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.rbox_contrast, 0, wx.ALL, 5)
        sizer.Add(button_gm, 0, wx.ALL, 5)
        self.sizer_h.Add(sizer)
        self.SetSizerAndFit(self.sizer_h)

    def onButtonSC(self, event):

        overlayORD = displayCtx.overlayOrder

        img1 = overlayORD[0]
        rawimg = overlayList[img1].dataSource
        print('Raw Image:', rawimg)
        contrast = self.rbox_contrast.GetStringSelection()

        base_name = os.path.basename(rawimg)
        fname, fext = base_name.split(os.extsep, 1)
        out_name = "{}_seg.{}".format(fname, fext)

        cmd_line = "sct_propseg -i {} -c {}".format(rawimg, contrast)
        print('Command line:', cmd_line)
        self.call_sct_command(cmd_line)

        outfilename = os.path.join(os.getcwd(), out_name)
        image = Image(outfilename)
        overlayList.append(image)

        opts = displayCtx.getOpts(image)
        opts.cmap = 'red'

# Run Spinal Cord segmentation using deep-learning
class TabPanelSCSeg(SCTPanel):
    DESCRIPTION = """This segmentation tool is based on Deep Learning and
    a 3D U-Net. For more information, please refer to the
    article below.<br><br>
    <b>Usage</b>:<br>
    To launch the script, the upload the raw image (t1, t2, t2s and dwi) into FSLeyes and always keeping it
    as the first in the Overlay list field from the bottom to the top. If you uploaded more then one image, it is not necessary
    uploading the images in such order, with the arrows is possible to sort them and only the first imaging will be used. 
    For more information, please refer to the article below.<br><br>
    <b>Specific citation</b>:<br>
    Gros et al.
    <i>Automatic segmentation of the spinal cord and intramedullary multiple sclerosis lesions with convolutional neural networks.
    (2019)</i>. Neuroimage. 1;184:901-915.

    """

    def __init__(self, parent):
        super(TabPanelSCSeg, self).__init__(parent=parent,
                                            id_=wx.ID_ANY)
        button_gm = wx.Button(self, id=wx.ID_ANY, label="Spinal Cord Segmentation")
        button_gm.Bind(wx.EVT_BUTTON, self.onButtonSC)

        lbl_contrasts = ['t1', 't2', 't2s', 'dwi']
        self.rbox_contrast = wx.RadioBox(self, label='Select contrast:',
                                         choices=lbl_contrasts,
                                         majorDimension=1,
                                         style=wx.RA_SPECIFY_ROWS)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.rbox_contrast, 0, wx.ALL, 5)
        sizer.Add(button_gm, 0, wx.ALL, 5)
        self.sizer_h.Add(sizer)
        self.SetSizerAndFit(self.sizer_h)

    def onButtonSC(self, event):

        overlayORD = displayCtx.overlayOrder

        img1 = overlayORD[0]
        rawimg = overlayList[img1].dataSource
        print('Raw Image:', rawimg)

        contrast = self.rbox_contrast.GetStringSelection()

        base_name = os.path.basename(rawimg)
        fname, fext = base_name.split(os.extsep, 1)
        out_name = "{}_seg.{}".format(fname, fext)

        cmd_line = "sct_deepseg_sc -i {} -c {}".format(rawimg, contrast)
        print('Command line:', cmd_line)
        self.call_sct_command(cmd_line)

        outfilename = os.path.join(os.getcwd(), out_name)
        image = Image(outfilename)
        overlayList.append(image)
        opts = displayCtx.getOpts(image)
        opts.cmap = 'red'


#Run Gray Matter Segmentation of Spinal Cord
class TabPanelGMSeg(SCTPanel):
    DESCRIPTION = """This segmentation tool is based on Deep Learning and
    dilated convolutions. For more information, please refer to the
    article below.<br><br>
    <b>Usage</b>:<br>
    To launch the script, upload the T2s imaging into FSLeyes and always keeping it
    as the first in the Overlay list field from the bottom to the top. If you uploaded more then one image
    with the arrows is possible to sort them and only the first imaging will be used.
    If you choose a customized output file name, include the imaging format. 
    Ex: filename.nii.gz or filename.nii
    For more information, please refer to the article below.<br><br>
    <b>Specific citation</b>:<br>
    Perone et al. 
    <i>Spinal cord gray matter segmentation using deep dilated convolutions
    (2018)</i>. Sci Rep. 13;8(1):5966.
    """

    def __init__(self, parent):
        super(TabPanelGMSeg, self).__init__(parent=parent,
                                            id_=wx.ID_ANY)
        button_gm = wx.Button(self, id=wx.ID_ANY,
                              label="Gray Matter Segmentation")
        button_gm.Bind(wx.EVT_BUTTON, self.onButtonGM)

        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        l1 = wx.StaticText(self, wx.ID_ANY, "Output File Name:")
        hbox1.Add(l1, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)
        self.t1 = wx.TextCtrl(self)
        self.t1.Bind(wx.EVT_TEXT, self.OnKeyTyped)
        hbox1.Add(self.t1, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(hbox1)
        sizer.Add(button_gm, 0, wx.ALL, 5)
        self.sizer_h.Add(sizer)
        self.SetSizerAndFit(self.sizer_h)

    def OnKeyTyped(self, event):
        txt = event.GetString()

    def onButtonGM(self, event):

        overlayORD = displayCtx.overlayOrder

        img1 = overlayORD[0]
        rawimg = overlayList[img1].dataSource
        gmimg = self.t1.GetValue()

        print('Raw Image:', rawimg)

        if gmimg == '':
            print('Output file name not defined, standard name will be used.')
            base_name = os.path.basename(rawimg)
            fname, fext = base_name.split(os.extsep, 1)
            gmname = "{}_gmseg.{}".format(fname, fext)
            print('Output File Name:', gmname)

            #Standard output file name
            cmd_line = "sct_deepseg_gm -i {}".format(rawimg)
            print('Command Line:', cmd_line)

        else:
            gmname = "{}.nii.gz".format(gmimg)
            print('Output File Name:', gmname)

            #Personalized output file name
            cmd_line = "sct_deepseg_gm -i {} -o {}".format(rawimg, gmname)
            print('Command Line:', cmd_line)

        self.call_sct_command(cmd_line)

        outfilename = os.path.join(os.getcwd(), gmname)
        image = Image(outfilename)
        overlayList.append(image)
        opts = displayCtx.getOpts(image)
        opts.cmap = 'yellow'

#Computes Cross-Sectional Area for GM, WM and GM+WM (Total)
class TabPanelCSA(SCTPanel):
    DESCRIPTION = """This tool computes the spinal cord cross-sectional area. 
    For more information, please refer to the article below.<br><br>
    <b>Specific citation</b>:<br>
    <b>Usage</b>:<br>
    To launch the script, upload the segmentation imaging.
    If you uploaded more then one image, just keep the binary segmentation in the bottom of the Overlay List field. 
    With the arrows is possible to sort them and only the first imaging will be used.
    The table is written in CSV format (.csv). 
    For more information, please refer to the article below.<br><br>
    Martin et al.
    <i>Can Microstructural MRI detect subclinical tissue injury in subjects with asymptomatic
    cervical spinal cord compression? A prospective cohort study
    (2018)</i>. BMJ Open. 13;8(4)e01980
    """


    def __init__(self, parent):
        super(TabPanelCSA, self).__init__(parent=parent,
                                          id_=wx.ID_ANY)
        button_gm = wx.Button(self, id=wx.ID_ANY, label="Process Segmentation")
        button_gm.Bind(wx.EVT_BUTTON, self.onButtonCSA, id = button_gm.GetId())

        lbl_vertfile = ['Off', 'On']
        self.rbox_vertfile = wx.RadioBox(self, label='Vertfile:',
                                         choices=lbl_vertfile,
                                         majorDimension=1,
                                         style=wx.RA_SPECIFY_ROWS)

        lbl_perlevel = ['No', 'Yes']
        self.rbox_perlevel = wx.RadioBox(self, label='Perlevel:',
                                         choices=lbl_perlevel,
                                         majorDimension=1,
                                         style=wx.RA_SPECIFY_ROWS)

        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        l1 = wx.StaticText(self, wx.ID_ANY, "Vertebral Levels/Slice Range")
        hbox1.Add(l1, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)
        self.t1 = wx.TextCtrl(self)
        self.t1.Bind(wx.EVT_TEXT, self.OnKeyTyped)
        hbox1.Add(self.t1, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)

        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        l2 = wx.StaticText(self, wx.ID_ANY, "Output Table Name")
        hbox2.Add(l2, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)
        self.t2 = wx.TextCtrl(self)
        self.t2.Bind(wx.EVT_TEXT, self.OnKeyTyped)
        hbox2.Add(self.t2, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.rbox_perlevel, 0, wx.ALL, 5)
        sizer.Add(hbox1)
        sizer.Add(hbox2)
        sizer.Add(self.rbox_vertfile, 0, wx.ALL, 5)
        sizer.Add(button_gm, 0, wx.ALL, 5)
        self.sizer_h.Add(sizer)
        self.SetSizerAndFit(self.sizer_h)

    def OnKeyTyped(self, event):
        txt = event.GetString()


    def onButtonCSA(self, event):

        img = overlayORD[0]
        segimg = overlayList[img].dataSource

        print('Segmentation:', segimg)

        vertfile = self.rbox_vertfile.GetStringSelection()
        perlevel = self.rbox_perlevel.GetStringSelection()
        vert = self.t1.GetValue()
        tabname = self.t2.GetValue()

        if vert == '':
            print('Vertebral labeling or slice range not defined.')
        else:
            print('Verts:', vert)

        if tabname == '':
            print('Output table name not defined, standard name, CSA.csv, will be used.')
        else:
            print('Output Table Name:', tabname)

        print('Perlevel:',perlevel)

        #Composition of sct_process_segmentation taken in account flags -z, -vert and -vertfile
        if perlevel == 'No':

            if tabname == '':

                outfilename = os.path.join(os.getcwd(), 'CSA.csv')

                #Slice Range
                if vert == '':

                    cmd_line = "sct_process_segmentation -i {} -o {}".format(segimg, outfilename)
                    print('Command line:', cmd_line)
                    self.call_sct_command(cmd_line)

                else:

                    cmd_line = "sct_process_segmentation -i {} -z {} -o {}".format(segimg, vert, outfilename)
                    print('Command line:', cmd_line)
                    self.call_sct_command(cmd_line)

            else:

                outfilename = os.path.join(os.getcwd(), '{}.csv'.format(tabname))

                if vert == '':

                    cmd_line = "sct_process_segmentation -i {} -o {}".format(segimg, outfilename)
                    print('Command line:', cmd_line)
                    self.call_sct_command(cmd_line)

                else:

                    cmd_line = "sct_process_segmentation -i {} -z {} -o {}".format(segimg, vert, outfilename)
                    print('Command line:', cmd_line)
                    self.call_sct_command(cmd_line)

        else:

            if vertfile == 'Off':

                if tabname == '':

                    outfilename = os.path.join(os.getcwd(), 'CSA.csv')

                    #Vertebral Levels
                    if vert == '':

                        print('Vertebral levels was not defined, standard value will be used (C2 to C6).')

                        cmd_line = "sct_process_segmentation -i {} -vert 3:7 -perlevel 1 -o {}".format(segimg, vert,
                                                                                                      outfilename)
                        print('Command line:', cmd_line)
                        self.call_sct_command(cmd_line)

                    else:

                        cmd_line = "sct_process_segmentation -i {} -vert {} -perlevel 1 -o {}".format(segimg, vert, outfilename)
                        print('Command line:', cmd_line)
                        self.call_sct_command(cmd_line)

                else:

                    outfilename = os.path.join(os.getcwd(), '{}.csv'.format(tabname))

                    if vert == '':

                        print('Vertebral levels was not defined, standard value will be used (C2 to C6).')
                        cmd_line = "sct_process_segmentation -i {} -vert 3:7 -perlevel 1 -o {}".format(segimg, vert, outfilename)
                        print(cmd_line)
                        self.call_sct_command(cmd_line)

                    else:

                        cmd_line = "sct_process_segmentation -i {} -vert {} -perlevel 1 -o {}".format(segimg, vert, outfilename)
                        print('Command line:', cmd_line)
                        self.call_sct_command(cmd_line)

            else:

                try:

                    img2 = overlayORD[1]
                    vertfileimg = overlayList[img2].dataSource
                    print('Vert file:', vertfileimg)

                except:

                    print('Error: Upload PAM50_levels.nii.gz')

                try:

                    if tabname == '':

                        outfilename = os.path.join(os.getcwd(), 'CSA.csv')

                        #Non-standard PAM50_levels.nii.gz
                        if vert == '':

                            print('Vertebral levels was not defined, standard value will be used (C2 to C6).')
                            cmd_line = "sct_process_segmentation -i {} -vert 3:7 -perlevel 1 -vertfile {} -o {}".format(
                                segimg, vert, vertfileimg, outfilename)
                            print('Command line:', cmd_line)
                            self.call_sct_command(cmd_line)

                        else:

                            cmd_line = "sct_process_segmentation -i {} -vert {} -perlevel 1 -vertfile {} -o {}".format(
                                segimg, vert, vertfileimg, outfilename)
                            print('Command line:', cmd_line)
                            self.call_sct_command(cmd_line)

                    else:

                        outfilename = os.path.join(os.getcwd(), '{}.csv'.format(tabname))

                        if vert == '':

                            print('Vertebral levels was not defined, standard value will be used (C2 to C6).')
                            cmd_line = "sct_process_segmentation -i {} -vert 3:7 -perlevel 1 -vertfile {} -o {}".format(
                                segimg, vert, vertfileimg, outfilename)
                            print('Command line:', cmd_line)
                            self.call_sct_command(cmd_line)

                        else:

                            cmd_line = "sct_process_segmentation -i {} -vert {} -perlevel 1 -vertfile {} -o {}".format(
                                segimg, vert, vertfileimg, outfilename)
                            print('Command line:', cmd_line)
                            self.call_sct_command(cmd_line)

                except:

                    print('Error: Upload PAM50_levels.nii.gz')

#Automatically identifies the vertebral levels
class TabPanelVertLB(SCTPanel):
    DESCRIPTION = """This tool automatically identifies the 
    vertebral levels. For more information, please refer to the
    article below.<br><br>
    <b>Usage</b>:<br>
    To launch the script, the uploaded images into FSLeyes must respect a sequence.
    In the Overlay list field, the images order is, from the bottom to the top, raw imaging
    (t1 and t2) and the segmentation imaging (output propseg or deepseg_sc). It is not necessary
    uploading the images in such order, with the arrows is possible to sort them.
    For more information, please refer to the article below.<br><br>
    <b>Specific citation</b>:<br>
    Ullmann et al.
    <i>Automatic labeling of vertebral levels using a robust template-based approach.
    (2014)</i>. Int J Biomed Imaging. 2014:719520.

    """

    def __init__(self, parent):
        super(TabPanelVertLB, self).__init__(parent=parent,
                                             id_=wx.ID_ANY)
        button_gm = wx.Button(self, id=wx.ID_ANY, label="Vertebral Labeling")
        button_gm.Bind(wx.EVT_BUTTON, self.onButtonVL)

        lbl_contrasts = ['t1', 't2']
        self.rbox_contrast = wx.RadioBox(self, label='Select contrast:',
                                         choices=lbl_contrasts,
                                         majorDimension=1,
                                         style=wx.RA_SPECIFY_ROWS)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.rbox_contrast, 0, wx.ALL, 5)
        sizer.Add(button_gm, 0, wx.ALL, 5)
        self.sizer_h.Add(sizer)
        self.SetSizerAndFit(self.sizer_h)

    def onButtonVL(self, event):

        overlayORD = displayCtx.overlayOrder

        print('Overlay Order:', overlayORD)

        img1 = overlayORD[0]
        img2 = overlayORD[1]
        rawimg = overlayList[img1].dataSource
        segimg = overlayList[img2].dataSource

        print('Raw Image:', rawimg,)
        print('Segmentation:', segimg)

        contrast = self.rbox_contrast.GetStringSelection()
        base_name = os.path.basename(segimg)

        fname, fext = base_name.split(os.extsep, 1)

        out_name = "{}_labeled.{}".format(fname, fext)

        cmd_line = "sct_label_vertebrae -i {} -s {} -c {}".format(rawimg, segimg, contrast)
        print('Command:', cmd_line)
        self.call_sct_command(cmd_line)

        outfilename = os.path.join(os.getcwd(), out_name)
        image = Image(outfilename)
        overlayList.append(image)

        opts = displayCtx.getOpts(image)
        opts.cmap = 'subcortical'

#Motion correction for dwi
class TabPanelMOCO(SCTPanel):
    DESCRIPTION = """This tool automatically performs
    motion correction in the diffusion-weighted imaging. For more information, please refer to the
    article below.<br><br>
    <b>Usage</b>:<br>
    To launch the script, the uploaded images into FSLeyes must respect a sequence.
    In the Overlay list field, the images order is, from the bottom to the top, the diffusion imaging
    (dwi) and, if you want, a mask imaging (binary mask to limit voxels considered by the registration metric). 
    It is not necessary uploading the images in such order, with the arrows is possible to sort them.
    Bvec and bval must be named as subjname_bvecs.txt and subjname_bvals.txt.
    For more information, please refer to the article below.<br><br>
    <b>Specific citation</b>:<br>
    Xu et al.
    <i>Evaluation of slice accelerations using multiband echo planar imaging at 3 T.
    (2013)</i>. Neuroimage. 83:991-1001.

    """

    def __init__(self, parent):
        super(TabPanelMOCO, self).__init__(parent=parent,
                                            id_=wx.ID_ANY)
        button_gm = wx.Button(self, id=wx.ID_ANY,
                              label="dMRI Moco")
        button_gm.Bind(wx.EVT_BUTTON, self.onButtonMOCO)

        lbl_mask = ['No', 'Yes']
        self.rbox_mask = wx.RadioBox(self, label='Use Mask:',
                                         choices=lbl_mask,
                                         majorDimension=1,
                                         style=wx.RA_SPECIFY_ROWS)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.rbox_mask, 0, wx.ALL, 5)
        sizer.Add(button_gm, 0, wx.ALL, 5)
        self.sizer_h.Add(sizer)
        self.SetSizerAndFit(self.sizer_h)

    def onButtonMOCO(self, event):

        overlayORD = displayCtx.overlayOrder

        print('Overlay Order:', overlayORD)

        img1 = overlayORD[0]
        rawimg = overlayList[img1].dataSource
        print('Raw Image:', rawimg)

        path_name = os.path.dirname(rawimg)
        base_name = os.path.basename(rawimg)

        fname, fext = base_name.split(os.extsep, 1)

        bvec_name = "{}_bvecs.txt".format(fname)
        bvecname_path = os.path.join(path_name, bvec_name)

        print('Bvec file:', bvecname_path)

        out_name = "{}_moco.{}".format(fname, fext)
        outname_path = os.path.join(path_name, out_name)

        mask = self.rbox_mask.GetStringSelection()

        if mask == 'Yes':

            print('Selected Mask: Yes')
            img2 = overlayORD[1]
            maskimg = overlayList[img2].dataSource
            print('Mask:', maskimg)

            cmd_line = "sct_dmri_moco -i {} -bvec {} -m {}".format(rawimg, bvecname_path, maskimg)
            print('Command line:', cmd_line)
            self.call_sct_command(cmd_line)

        else:

            print('Selected Mask: No')
            cmd_line = "sct_dmri_moco -i {} -bvec {}".format(rawimg, bvecname_path)
            print('Command line:', cmd_line)
            self.call_sct_command(cmd_line)

        image = Image(outname_path)
        overlayList.append(image)
        opts = displayCtx.getOpts(image)
        opts.cmap = 'gray'

#Compution of the diffusion maps
class TabPanelCompDTI(SCTPanel):
    DESCRIPTION = """This tool automatically compute the diffusion maps of the spinal cord. 
    For more information, please refer to the article below.<br><br>
    <b>Usage</b>:<br>
    To launch the script, the uploaded the image that you want to compute the diffusion parameters into FSLeyes.
    If you have more than one imaging uploaded, the first image from the bottom to the top, in the Overlay list field, 
    will be used to launch this script.
    It is not necessary uploading the images in such order, with the arrows is possible to sort them.
    Bvec and bval must be named as subjname_bvecs.txt and subjname_bvals.txt.
    For more information, please refer to the article below.<br><br>
    <b>Specific citation</b>:<br>
    Garyfallidis et al.
    <i>Dipy, a library for the analysis of diffusion MRI data.
    (2014)</i>. Front Neuroinform. 21;8:8.

    """

    def __init__(self, parent):
        super(TabPanelCompDTI, self).__init__(parent=parent,
                                            id_=wx.ID_ANY)
        button_gm = wx.Button(self, id=wx.ID_ANY,
                              label="Compute DTI")
        button_gm.Bind(wx.EVT_BUTTON, self.onButtonCDTI)

        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        l1 = wx.StaticText(self, wx.ID_ANY, "Output File Name:")
        hbox1.Add(l1, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)
        self.t1 = wx.TextCtrl(self)
        self.t1.Bind(wx.EVT_TEXT, self.OnKeyTyped)
        hbox1.Add(self.t1, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(hbox1)
        sizer.Add(button_gm, 0, wx.ALL, 5)
        self.sizer_h.Add(sizer)
        self.SetSizerAndFit(self.sizer_h)

    def OnKeyTyped(self, event):
        txt = event.GetString()

    def onButtonCDTI(self, event):

        overlayORD = displayCtx.overlayOrder

        img1 = overlayORD[0]
        rawimg = overlayList[img1].dataSource
        print('Input Image:', rawimg)

        outname = self.t1.GetValue()

        path_name = os.path.dirname(rawimg)
        base_name = os.path.basename(rawimg)
        fnamecrop, fext = base_name.split(os.extsep, 1)
        fname = fnamecrop.split('_')[0]

        bvec_name = "{}_bvecs.txt".format(fname)
        bvecname_path = os.path.join(path_name, bvec_name)

        print('Bvec file:', bvecname_path)

        bval_name = "{}_bvals.txt".format(fname)
        bvalname_path = os.path.join(path_name, bval_name)

        print('Bval file:', bvalname_path)

        if outname == '':

            print('Output file name not defined, standard name will be used.')
            fa_name = "dti_FA.nii.gz"
            faname_path = os.path.join(os.getcwd(), fa_name)

            ad_name = "dti_AD.nii.gz"
            adname_path = os.path.join(os.getcwd(), ad_name)

            md_name = "dti_MD.nii.gz"
            mdname_path = os.path.join(os.getcwd(), md_name)

            rd_name = "dti_RD.nii.gz"
            rdname_path = os.path.join(os.getcwd(), rd_name)

            cmd_line = "sct_dmri_compute_dti -i {} -bval {} -bvec {}".format(rawimg, bvalname_path,
                                                                             bvecname_path)
            print('Command line:', cmd_line)

        else:

            print('Output file name:', outname)
            fa_name = "{}FA.nii.gz".format(outname)
            faname_path = os.path.join(os.getcwd(), fa_name)

            ad_name = "{}AD.nii.gz".format(outname)
            adname_path = os.path.join(os.getcwd(), ad_name)

            md_name = "{}MD.nii.gz".format(outname)
            mdname_path = os.path.join(os.getcwd(), md_name)

            rd_name = "{}RD.nii.gz".format(outname)
            rdname_path = os.path.join(os.getcwd(), rd_name)

            cmd_line = "sct_dmri_compute_dti -i {} -bval {} -bvec {} -o {}".format(rawimg, bvalname_path, bvecname_path, outname)
            print('Command line:', cmd_line)

        self.call_sct_command(cmd_line)

        image = Image(faname_path)
        overlayList.append(image)
        del overlayList[0]
        opts = displayCtx.getOpts(image)
        opts.cmap = 'gray'

        image = Image(adname_path)
        overlayList.append(image)
        opts = displayCtx.getOpts(image)
        opts.cmap = 'gray'

        image = Image(mdname_path)
        overlayList.append(image)
        opts = displayCtx.getOpts(image)
        opts.cmap = 'gray'

        image = Image(rdname_path)
        overlayList.append(image)
        opts = displayCtx.getOpts(image)
        opts.cmap = 'gray'

#Registraction between subject imaging (T1w or T2w) with PAM50 template
class TabPanelREG(SCTPanel):
    DESCRIPTION = """This tool automatically performs
    the registration between your subject and the PAM50 template. 
    For more information, please refer to the article below.<br><br>
    <b>Usage</b>:<br>
    To launch the script, the uploaded images into FSLeyes must respect a sequence.
    In the Overlay list field, the images order is, from the bottom to the top, raw imaging
    (t1 and t2) segmentation imaging (output propseg or deepseg_sc) and disc labeling imaging (output label_utils). 
    It is not necessary uploading the images in such order, with the arrows is possible to sort them. 
    For more information, please refer to the article below.<br><br>
    <b>Specific citation</b>:<br>
    De Lenner et al.
    <i>PAM50: Unbiased multimodal template of the brainstem and spinal cord aligned with the ICBM152 space
    (2018)</i>. Neuroimage. 15;165:170-179.

    """

    def __init__(self, parent):
        super(TabPanelREG, self).__init__(parent=parent,
                                             id_=wx.ID_ANY)
        button_gm = wx.Button(self, id=wx.ID_ANY, label="Registration")
        button_gm.Bind(wx.EVT_BUTTON, self.onButtonREG)

        lbl_contrasts = ['t1', 't2', 't2s']
        self.rbox_contrast = wx.RadioBox(self, label='Select contrast:',
                                         choices=lbl_contrasts,
                                         majorDimension=1,
                                         style=wx.RA_SPECIFY_ROWS)

        lbl_label = ['Automatic', 'Manual']
        self.rbox_label = wx.RadioBox(self, label='Disc Labeling:',
                                      choices=lbl_label,
                                      majorDimension=1,
                                      style=wx.RA_SPECIFY_ROWS)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.rbox_contrast, 0, wx.ALL, 5)
        sizer.Add(self.rbox_label, 0, wx.ALL, 5)
        sizer.Add(button_gm, 0, wx.ALL, 5)
        self.sizer_h.Add(sizer)
        self.SetSizerAndFit(self.sizer_h)


    def onButtonREG(self, event):

        overlayORD = displayCtx.overlayOrder

        print('Overlay Order:', overlayORD)

        img1 = overlayORD[0]
        img2 = overlayORD[1]
        img3 = overlayORD[2]
        rawimg = overlayList[img1].dataSource
        segimg = overlayList[img2].dataSource
        labimg = overlayList[img3].dataSource

        print('Raw Image:', rawimg)
        print('Segmentation:', segimg)
        print('Vertenbral Labeling:', labimg)

        contrast = self.rbox_contrast.GetStringSelection()
        label = self.rbox_label.GetStringSelection()

        out_name = 'template2anat.nii.gz'
        outfilename = os.path.join(os.getcwd(),out_name)

        print('Contrast:', contrast)

        print('Label:', label)

        if label == 'Automatic':

            cmd_line = "sct_register_to_template -i {} -s {} -l {} -c {}".format(rawimg, segimg, labimg, contrast)
            print('Command:', cmd_line)
            self.call_sct_command(cmd_line)

            image = Image(outfilename)
            overlayList.append(image)
            opts = displayCtx.getOpts(image)
            opts.cmap = 'gray'

        else:

            cmd_line = "sct_register_to_template -i {} -s {} -ldisc {} -c {}".format(rawimg, segimg,
                                                                                  labimg, contrast)
            print('Command:', cmd_line)
            self.call_sct_command(cmd_line)

            image = Image(outfilename)
            overlayList.append(image)
            opts = displayCtx.getOpts(image)
            opts.cmap = 'gray'



def run_main():
    window = aui_manager.GetManagedWindow()

    if 'SCT_DIR' not in os.environ:
        dlg = wx.MessageDialog(window, 'Spinal Cord Toolbox (SCT) was not '
                                       'found in your system. Make sure you open fsleyes '
                                       'from the Terminal (not by clicking on the App). '
                                       'If you are indeed running from the Terminal, please '
                                       'check the installation procedure at: '
                                       'https://github.com/neuropoly/spinalcordtoolbox',
                               'SCT not found!', wx.OK | wx.ICON_INFORMATION)
        dlg.ShowModal()
        dlg.Destroy()
        return
    # Adding panels
    notebook = aui.AuiNotebook(window)
    panel_propseg = TabPanelPropSeg(notebook)
    panel_sc = TabPanelSCSeg(notebook)
    panel_gm = TabPanelGMSeg(notebook)
    panel_vlb = TabPanelVertLB(notebook)
    panel_reg = TabPanelREG(notebook)
    panel_csa = TabPanelCSA(notebook)
    panel_moco = TabPanelMOCO(notebook)
    panel_cdti = TabPanelCompDTI(notebook)

    notebook.AddPage(panel_propseg, "sct_propseg", True)
    notebook.AddPage(panel_sc, "sct_deepseg_sc", False)
    notebook.AddPage(panel_gm, "sct_deepseg_gm", False)
    notebook.AddPage(panel_vlb, "sct_label_vertebrae", False)
    notebook.AddPage(panel_reg, "sct_register_to_template", False)
    notebook.AddPage(panel_csa, "sct_process_segmentation", False)
    notebook.AddPage(panel_moco, "sct_dmri_moco", False)
    notebook.AddPage(panel_cdti, "sct_dmri_compute_dti", False)

    aui_manager.AddPane(notebook,
                        aui.AuiPaneInfo().Name("notebook_content").
                        CenterPane().PaneBorder(False))
    aui_manager.Update()


run_main()
