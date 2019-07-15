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
# Uptaded: 11 Jul 2019
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
        self.SetSize((350, 80))

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

        self.SetSizer(sizer)

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

    def __init__(self, parent, id_):
        super(SCTPanel, self).__init__(parent=parent,
                                       id=id_)
        self.img_logo = self.get_logo()
        self.html_desc = self.get_description()

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

        self.sizer_logo_text = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer_logo_text.Add(self.sizer_logo_sct, 0, wx.ALL, 5)
        self.sizer_logo_text.Add(self.html_desc, 0, wx.ALL, 5)

        self.sizer_h = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer_h.Add(self.sizer_logo_text)

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
    robustness, accuracy and speed. For more information, please refer to the
    article below.<br><br>
    <b>Specific citation</b>:<br>
    B De Leener, S Kadoury, J Cohen-Adad
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
        selected_overlay = displayCtx.getSelectedOverlay()
        filename_path = selected_overlay.dataSource
        contrast = self.rbox_contrast.GetStringSelection()

        base_name = os.path.basename(filename_path)
        fname, fext = base_name.split(os.extsep, 1)
        out_name = "{}_seg.{}".format(fname, fext)

        cmd_line = "sct_propseg -i {} -c {}".format(filename_path, contrast)
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
    <b>Specific citation</b>:<br>
    C Gros, B De Leener et al
    <i>Automatic segmentation of the spinal cord and intramedullary multiple sclerosis lesions with convolutional neural networks.
    (2019)</i>. arXiv:1805.06349.

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
        selected_overlay = displayCtx.getSelectedOverlay()
        filename_path = selected_overlay.dataSource
        contrast = self.rbox_contrast.GetStringSelection()

        base_name = os.path.basename(filename_path)
        fname, fext = base_name.split(os.extsep, 1)
        out_name = "{}_seg.{}".format(fname, fext)

        cmd_line = "sct_deepseg_sc -i {} -c {}".format(filename_path, contrast)
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
    <b>Specific citation</b>:<br>
    CS Perone, E Calabrese, J Cohen-Adad.
    <i>Spinal cord gray matter segmentation using deep dilated convolutions
    (2017)</i>. ArXiv: arxiv.org/abs/1710.01269
    """

    def __init__(self, parent):
        super(TabPanelGMSeg, self).__init__(parent=parent,
                                            id_=wx.ID_ANY)
        button_gm = wx.Button(self, id=wx.ID_ANY,
                              label="Gray Matter Segmentation")
        button_gm.Bind(wx.EVT_BUTTON, self.onButtonGM)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(button_gm, 0, wx.ALL, 5)
        self.sizer_h.Add(sizer)
        self.SetSizerAndFit(self.sizer_h)

    def onButtonGM(self, event):
        selected_overlay = displayCtx.getSelectedOverlay()
        filename_path = selected_overlay.dataSource
        cmd_line = "sct_deepseg_gm -i {} -o seg.nii.gz".format(filename_path)

        self.call_sct_command(cmd_line)

        outfilename = os.path.join(os.getcwd(), 'seg.nii.gz')
        image = Image(outfilename)
        overlayList.append(image)

        opts = displayCtx.getOpts(image)
        opts.cmap = 'yellow'

#Computes Cross-Sectional Area for GM, WM and GM+WM (Total)
class TabPanelCSA(SCTPanel):
    DESCRIPTION = """This tool computes the spinal cord cross-sectional area. 
    For more information, please refer to the article below.<br><br>
    <b>Specific citation</b>:<br>
    AR Martin, B De Leener, J Cohen-Adad, et al.
    <i>Can Microstructural MRI detect subclinical tissue injury in subjects with asymptomatic
    cervical spinal cord compression? A prospective cohort study
    (2018)</i>. BMJ Open. 13;8(4)e01980
    """


    def __init__(self, parent):
        super(TabPanelCSA, self).__init__(parent=parent,
                                          id_=wx.ID_ANY)
        button_gm = wx.Button(self, id=wx.ID_ANY, label="Cross-Sectional Area")
        button_gm.Bind(wx.EVT_BUTTON, self.onButtonCSA, id = button_gm.GetId())

        lbl_contrasts = ['GM', 'WM', 'Total']
        self.rbox_contrast = wx.RadioBox(self, label='Select tissue:',
                                         choices=lbl_contrasts,
                                         majorDimension=1,
                                         style=wx.RA_SPECIFY_ROWS)

        lbl_perlevel = ['No', 'Yes']
        self.rbox_perlevel = wx.RadioBox(self, label='Perlevel:',
                                         choices=lbl_perlevel,
                                         majorDimension=1,
                                         style=wx.RA_SPECIFY_ROWS)

        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        l1 = wx.StaticText(self, wx.ID_ANY, "Vertebral Levels")
        hbox1.Add(l1, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)
        self.t1 = wx.TextCtrl(self)
        self.t1.Bind(wx.EVT_TEXT, self.OnKeyTyped)
        hbox1.Add(self.t1, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.rbox_contrast, 0, wx.ALL, 5)
        sizer.Add(self.rbox_perlevel, 0, wx.ALL, 5)
        sizer.Add(hbox1)
        sizer.Add(button_gm, 0, wx.ALL, 5)
        self.sizer_h.Add(sizer)
        self.SetSizerAndFit(self.sizer_h)


    def OnKeyTyped(self, event):
        vert = event.GetString()


    def onButtonCSA(self, event):
        selected_overlay = displayCtx.getSelectedOverlay()
        filename_path = selected_overlay.dataSource
        contrast = self.rbox_contrast.GetStringSelection()
        perlevel = self.rbox_perlevel.GetStringSelection()
        vert = self.t1.GetValue()

        print('Contrast:', contrast)

        print('Verts:', vert)

        print('Perlevel:',perlevel)

        path_name = os.path.dirname(filename_path)
        base_name = os.path.basename(filename_path)

        fname, fext = base_name.split(os.extsep, 1)
        seg_name = "{}_seg.{}".format(fname, fext)
        segname_path = os.path.join(path_name, seg_name)

        print('Tissue:', contrast)

        if perlevel == 'No':

            if contrast == 'GM':
                outfilename = os.path.join(os.getcwd(), 'gmCSA.csv')

            elif contrast == 'WM':
                outfilename = os.path.join(os.getcwd(), 'wmCSA.csv')

            else:
                outfilename = os.path.join(os.getcwd(), 'totalCSA.csv')

            cmd_line = "sct_process_segmentation -i {} -o {}".format(segname_path, outfilename)
            print('Command line:', cmd_line)
            self.call_sct_command(cmd_line)

        else:

            if contrast == 'GM':
                outfilename = os.path.join(os.getcwd(), 'gmCSA_perlevel.csv')

            elif contrast == 'WM':
                outfilename = os.path.join(os.getcwd(), 'wmCSA_perlevel.csv')

            else:
                outfilename = os.path.join(os.getcwd(), 'totalCSA_perlevel.csv')

            cmd_line1 = "sct_warp_template -d {} -w warp_template2anat.nii.gz".format(filename_path)
            print('Command line:', cmd_line1)
            self.call_sct_command(cmd_line1)

            cmd_line2 = "sct_process_segmentation -i {} -vert {} -perlevel 1 -o {}".format(segname_path, vert, outfilename)
            print('Command line:', cmd_line2)
            self.call_sct_command(cmd_line2)

#Automatically identifies the vertebral levels
class TabPanelVertLB(SCTPanel):
    DESCRIPTION = """This tool automatically identifies the 
    vertebral levels. For more information, please refer to the
    article below.<br><br>
    <b>Specific citation</b>:<br>
    E Ullmann, JF Pelletier Paquette, WE Thong, J Cohen-Adad
    <i>Automatic labeling of vertebral levels using a robust template-based approach.
    (2014)</i>. Int J Biomed Imaging.

    """

    def __init__(self, parent):
        super(TabPanelVertLB, self).__init__(parent=parent,
                                             id_=wx.ID_ANY)
        button_gm = wx.Button(self, id=wx.ID_ANY, label="Vertebral Labeling")
        button_gm.Bind(wx.EVT_BUTTON, self.onButtonVL)

        lbl_contrasts = ['t1', 't2', 't2s']
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
        selected_overlay = displayCtx.getSelectedOverlay()
        filename_path = selected_overlay.dataSource
        contrast = self.rbox_contrast.GetStringSelection()

        path_name = os.path.dirname(filename_path)
        base_name = os.path.basename(filename_path)

        fname, fext = base_name.split(os.extsep, 1)
        seg_name = "{}_seg.{}".format(fname, fext)
        segname_path = os.path.join(path_name, seg_name)

        out_name = "{}_seg_labeled.{}".format(fname, fext)

        cmd_line = "sct_label_vertebrae -i {} -s {} -c {}".format(filename_path, segname_path, contrast)
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
    <b>Specific citation</b>:<br>
    J Xu, S Moeller, EJ Auerbach, et al.
    <i>Evaluation of slice accelerations using multiband echo planar imaging at 3 T.
    (2013)</i>. Neuroimage.

    """

    def __init__(self, parent):
        super(TabPanelMOCO, self).__init__(parent=parent,
                                            id_=wx.ID_ANY)
        button_gm = wx.Button(self, id=wx.ID_ANY,
                              label="dMRI Moco")
        button_gm.Bind(wx.EVT_BUTTON, self.onButtonMOCO)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(button_gm, 0, wx.ALL, 5)
        self.sizer_h.Add(sizer)
        self.SetSizerAndFit(self.sizer_h)

    def onButtonMOCO(self, event):
        selected_overlay = displayCtx.getSelectedOverlay()
        filename_path = selected_overlay.dataSource

        path_name = os.path.dirname(filename_path)
        base_name = os.path.basename(filename_path)

        fname, fext = base_name.split(os.extsep, 1)

        mean_name = "{}_mean.{}".format(fname,fext)
        meanname_path = os.path.join(path_name, mean_name)

        seg_name = "{}_mean_seg.{}".format(fname, fext)
        segname_path = os.path.join(path_name, seg_name)

        mask_name = "mask_{}_mean.{}".format(fname,fext)
        maskname_path = os.path.join(path_name, mask_name)

        crop_name = "{}_crop.{}".format(fname,fext)
        cropname_path = os.path.join(path_name, crop_name)

        bvec_name = "{}_bvecs.txt".format(fname)
        bvecname_path = os.path.join(path_name, bvec_name)

        out_name = "{}_crop_moco.{}".format(fname, fext)
        outname_path = os.path.join(path_name, out_name)

        cmd_line1 = "sct_maths -i {} -mean t -o {}".format(filename_path, meanname_path)
        print('Command:', cmd_line1)
        self.call_sct_command(cmd_line1)

        image = Image(meanname_path)
        overlayList.append(image)
        del overlayList[0]

        opts = displayCtx.getOpts(image)
        opts.cmap = 'gray'

        cmd_line2 = "sct_deepseg_sc -i {} -c dwi".format(meanname_path)
        print('Command:', cmd_line2)
        self.call_sct_command(cmd_line2)

        image = Image(segname_path)
        overlayList.append(image)

        opts = displayCtx.getOpts(image)
        opts.cmap = 'red'

        cmd_line3 = "sct_create_mask -i {} -p centerline,{} -size 35mm".format(meanname_path, segname_path)
        print('Command:', cmd_line3)
        self.call_sct_command(cmd_line3)

        del overlayList[1]

        image = Image(maskname_path)
        overlayList.append(image)

        opts = displayCtx.getOpts(image)
        opts.cmap = 'red'

        cmd_line4 = "sct_crop_image -i {} -m {} -o {}".format(filename_path, maskname_path, cropname_path)
        print('Command:', cmd_line4)
        self.call_sct_command(cmd_line4)

        cmd_line5 = "sct_dmri_moco -i {} -bvec {}".format(cropname_path, bvecname_path)
        print('Command:', cmd_line5)
        self.call_sct_command(cmd_line5)

        del overlayList[0]
        image = Image(outname_path)
        overlayList.append(image)

        del overlayList[0]

        opts = displayCtx.getOpts(image)
        opts.cmap = 'gray'

#Compution of the diffusion maps
class TabPanelCompDTI(SCTPanel):
    DESCRIPTION = """This tool automatically compute the diffusion maps of the spinal cord. 
    For more information, please refer to the article below.<br><br>
    <b>Specific citation</b>:<br>
    E Garyfallidis, M Brett, B Amirbekian, et al.
    <i>Dipy, a library for the analysis of diffusion MRI data.
    (2014)</i>. Front Neuroinform.

    """

    def __init__(self, parent):
        super(TabPanelCompDTI, self).__init__(parent=parent,
                                            id_=wx.ID_ANY)
        button_gm = wx.Button(self, id=wx.ID_ANY,
                              label="Compute DTI")
        button_gm.Bind(wx.EVT_BUTTON, self.onButtonCDTI)
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(button_gm, 0, wx.ALL, 5)
        self.sizer_h.Add(sizer)
        self.SetSizerAndFit(self.sizer_h)

    def onButtonCDTI(self, event):
        selected_overlay = displayCtx.getSelectedOverlay()
        filename_path = selected_overlay.dataSource

        path_name = os.path.dirname(filename_path)
        base_name = os.path.basename(filename_path)

        fnamecrop, fext = base_name.split(os.extsep, 1)

        fname = fnamecrop.split('_')[0]

        bvec_name = "{}_bvecs.txt".format(fname)
        bvecname_path = os.path.join(path_name, bvec_name)

        bval_name = "{}_bvals.txt".format(fname)
        bvalname_path = os.path.join(path_name, bval_name)

        fa_name = "dti_FA.nii.gz"
        faname_path = os.path.join(os.getcwd(), fa_name)

        ad_name = "dti_AD.nii.gz"
        adname_path = os.path.join(os.getcwd(), ad_name)

        md_name = "dti_MD.nii.gz"
        mdname_path = os.path.join(os.getcwd(), md_name)

        rd_name = "dti_RD.nii.gz"
        rdname_path = os.path.join(os.getcwd(), rd_name)

        cmd_line = "sct_dmri_compute_dti -i {} -bval {} -bvec {}".format(filename_path, bvalname_path, bvecname_path)
        print('Command:', cmd_line)
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
    <b>Specific citation</b>:<br>
    B De Lenner, VS Fonov, D Louis Collins, et al.
    <i>PAM50: Unbiased multimodal template of the brainstem and spinal cord aligned with the ICBM152 space
    (2017)</i>. Neuroimage.

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

        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        l1 = wx.StaticText(self, wx.ID_ANY, "Disc Levels")
        hbox1.Add(l1, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)
        self.t1 = wx.TextCtrl(self)
        self.t1.Bind(wx.EVT_TEXT, self.OnKeyTyped)
        hbox1.Add(self.t1, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.rbox_contrast, 0, wx.ALL, 5)
        sizer.Add(self.rbox_label, 0, wx.ALL, 5)
        sizer.Add(hbox1)
        sizer.Add(button_gm, 0, wx.ALL, 5)
        self.sizer_h.Add(sizer)
        self.SetSizerAndFit(self.sizer_h)

    def OnKeyTyped(self, event):
        vert = event.GetString()


    def onButtonREG(self, event):
        selected_overlay = displayCtx.getSelectedOverlay()
        filename_path = selected_overlay.dataSource
        contrast = self.rbox_contrast.GetStringSelection()
        label = self.rbox_label.GetStringSelection()
        vert = self.t1.GetValue()

        print('Contrast:', contrast)

        print('Label:', label)

        print('Vertebral Levels:', vert)

        path_name = os.path.dirname(filename_path)
        base_name = os.path.basename(filename_path)

        fname, fext = base_name.split(os.extsep, 1)

        seg_name = "{}_seg.{}".format(fname, fext)
        segname_path = os.path.join(path_name, seg_name)

        out_name = "template2anat.nii.gz"
        outfilename = os.path.join(os.getcwd(), out_name)

        if label == 'Automatic':

            vl_name = "{}_seg_labeled.{}".format(fname, fext)
            vlname_path = os.path.join(path_name, vl_name)

            vert_name = "{}_labels_vert.{}".format(fname, fext)
            vertname_path = os.path.join(path_name, vert_name)

            cmd_line1 = "sct_label_utils -i {} -vert-body {} -o {}".format(vlname_path, vert, vertname_path)
            print('Command:', cmd_line1)
            self.call_sct_command(cmd_line1)

            cmd_line2 = "sct_register_to_template -i {} -s {} -l {} -c {}".format(filename_path, segname_path, vertname_path, contrast)
            print('Command:', cmd_line2)
            self.call_sct_command(cmd_line2)

            image = Image(outfilename)
            overlayList.append(image)
            opts = displayCtx.getOpts(image)
            opts.cmap = 'gray'

        else:

            disc_name = "{}_labels_disc.{}".format(fname, fext)
            discname_path = os.path.join(path_name, disc_name)

            cmd_line1 = "sct_label_utils -i {} -create-viewer {} -o {}".format(filename_path, vert, discname_path)
            print('Command:', cmd_line1)
            self.call_sct_command(cmd_line1)

            cmd_line2 = "sct_register_to_template -i {} -s {} -ldisc {} -c {}".format(filename_path, segname_path,
                                                                                  discname_path, contrast)
            print('Command:', cmd_line2)
            self.call_sct_command(cmd_line2)

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
