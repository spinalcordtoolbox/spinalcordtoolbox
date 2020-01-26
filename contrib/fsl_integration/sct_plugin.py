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

# TODO: display window if process fails
# TODO: add shortcuts to Run
# TODO: add help when user leaves cursor on button

import os
import sys
import subprocess
from threading import Thread
import logging

import wx
import wx.lib.agw.aui as aui
import wx.html as html

logger = logging.getLogger(__name__)
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
        p = subprocess.Popen([command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, env=env)
        stdout, stderr = p.communicate()
        # TODO: Fix: tqdm progress bar causes the printing of stdout to stop
        print("\n\033[94m{}\033[0m\n".format(stdout.decode('utf-8')))
        if stderr:
            print("\n\033[91mERROR: {}\033[0m\n".format(stderr.decode('utf-8')))
        return stdout, stderr

    def run(self):
        """
        overrides Thread.run() function
        :return:
        """
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


class TextBox:
    """
    Create a horizontal box composed of a button (left) and a text box (right). When the button is
    pressed, the file name highlighted in the list of overlay is fetched and passed into the text box.
    This file name can be accessed by: TextBox.textctrl.GetValue()
    """
    def __init__(self, sctpanel, label=""):
        """
        :param sctpanel: SCTPanel Class
        :param label: Label to display on the button
        """
        self.textctrl = wx.TextCtrl(sctpanel, -1, "", wx.DefaultPosition, wx.Size(1000, 10))
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        button_fetch_file = wx.Button(sctpanel, -1, label=label)
        button_fetch_file.Bind(wx.EVT_BUTTON, self.get_highlighted_file_name)
        hbox.Add(button_fetch_file, proportion=0, flag=wx.ALIGN_LEFT | wx.ALL, border=5)
        hbox.Add(self.textctrl, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)
        self.hbox = hbox

    def get_highlighted_file_name(self, event):
        """
        Fetch path to file highlighted in the Overlay list.
        """
        selected_overlay = displayCtx.getSelectedOverlay()  # displayCtx is a class from FSLeyes
        filename_path = selected_overlay.dataSource
        print("Fetched file name: {}".format(filename_path))
        # display file name in text box
        self.textctrl.SetValue(filename_path)


# Creates the standard panel for each tool
class SCTPanel(wx.Panel):
    """
    Creates the standard panel for each tool
    :param sizer_h: Main wx.BoxSizer object that encloses SCT information, for each panel
    """

    DESCRIPTION_SCT = """
    <b>General citation (please always cite)</b>:<br>
    De Leener B, Levy S, Dupont SM, Fonov VS, Stikov N, Louis Collins D, Callot V,
    Cohen-Adad J. <i>SCT: Spinal Cord Toolbox, an open-source software for processing
    spinal cord MRI data</i>. Neuroimage. 2017 Jan 15;145(Pt A):24-43.
    """

    SCT_DIR_ENV = 'SCT_DIR'
    SCT_LOGO_REL_PATH = 'documentation/imgs/logo_sct.png'
    SCT_TUTORIAL_PATH = 'documentation/Manual_v1_SCT.pdf'  # TODO: fix this path

    def __init__(self, parent, id_):
        super(SCTPanel, self).__init__(parent=parent, id=id_)

        # Logo
        self.img_logo = self.get_logo()
        self.sizer_logo_sct = wx.BoxSizer(wx.VERTICAL)
        self.sizer_logo_sct.Add(self.img_logo, 0, wx.ALL, 5)

        # Citation
        txt_sct_citation = wx.VSCROLL | \
                           wx.HSCROLL | wx.TE_READONLY | \
                           wx.BORDER_SIMPLE
        html_sct_citation = html.HtmlWindow(self, wx.ID_ANY,
                                            size=(280, 115),
                                            style=txt_sct_citation)
        html_sct_citation.SetPage(self.DESCRIPTION_SCT)
        self.sizer_logo_sct.Add(html_sct_citation, 0, wx.ALL, 5)

        # Help button
        button_help = wx.Button(self, id=id_, label="Help")
        button_help.Bind(wx.EVT_BUTTON, self.tutorial)
        self.sizer_logo_sct.Add(button_help, 0, wx.ALL, 5)

        # Get function-specific description
        self.html_desc = self.get_description()

        # Organize boxes
        self.sizer_logo_text = wx.BoxSizer(wx.HORIZONTAL)  # create main box
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
        print("Running: {}".format(command))
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


class TabPanelPropSeg(SCTPanel):
    """
    sct_propseg
    """

    DESCRIPTION = """
    Segment the spinal cord using a deformable 3D mesh. This method is fast and robust, but could be prone to "leaking"
    if the contrast between the cord and the CSF is not high enough. 
    <br><br>
    <b>Usage</b>:
    <br>
    Select an image from the overlay list, then click the "Input file" button to fetch the file name. Then, select the
    appropriate contrast and click "Run". For more options, please use the Terminal version of this function.
    <br><br>
    <b>Specific citation</b>:
    <br>
    De Leener et al. <i>Robust, accurate and fast automatic segmentation of the spinal cord.</i> Neuroimage 2014
    """

    def __init__(self, parent):
        super(TabPanelPropSeg, self).__init__(parent=parent, id_=wx.ID_ANY)

        # Fetch input file
        self.hbox_filein = TextBox(self, label="Input file")

        # Select contrast
        lbl_contrasts = ['t1', 't2', 't2s', 'dwi']
        self.rbox_contrast = wx.RadioBox(self, label='Select contrast:',
                                         choices=lbl_contrasts,
                                         majorDimension=1,
                                         style=wx.RA_SPECIFY_ROWS)

        # Display all options
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.hbox_filein.hbox, 0, wx.ALL, 5)
        sizer.Add(self.rbox_contrast, 0, wx.ALL, 5)

        # Run button
        button_run = wx.Button(self, id=wx.ID_ANY, label="Run")
        button_run.Bind(wx.EVT_BUTTON, self.on_button_run)
        sizer.Add(button_run, 0, wx.ALL, 5)

        # Add to main sizer
        self.sizer_h.Add(sizer)
        self.SetSizerAndFit(self.sizer_h)

    def on_button_run(self, event):

        # Build and run SCT command
        fname_input = self.hbox_filein.textctrl.GetValue()
        contrast = self.rbox_contrast.GetStringSelection()
        base_name = os.path.basename(fname_input)
        fname, fext = base_name.split(os.extsep, 1)
        fname_out = "{}_seg.{}".format(fname, fext)
        cmd_line = "sct_propseg -i {} -c {}".format(fname_input, contrast)
        self.call_sct_command(cmd_line)

        # Add output to the list of overlay
        image = Image(fname_out)  # <class 'fsl.data.image.Image'>
        overlayList.append(image)
        opts = displayCtx.getOpts(image)
        opts.cmap = 'red'


class TabPanelSCSeg(SCTPanel):
    """
    sct_deepseg_sc
    """

    DESCRIPTION = """
    Segment the spinal cord using deep learning. The convolutional neural network was trained on ~1,500 subjects 
    from multiple centers, and including various pathologies (compression, MS, ALS, etc.). 
    <br><br>
    <b>Usage</b>:
    <br>
    Select an image from the overlay list, then click the "Input file" button to fetch the file name. Then, select the
    appropriate contrast and click "Run". For more options, please use the Terminal version of this function.
    <br><br>
    <b>Specific citation</b>:
    <br>
    Gros et al. <i>Automatic segmentation of the spinal cord and intramedullary multiple sclerosis lesions with 
    convolutional neural networks.</i> Neuroimage 2019
    """

    def __init__(self, parent):
        super(TabPanelSCSeg, self).__init__(parent=parent, id_=wx.ID_ANY)

        # Fetch input file
        self.hbox_filein = TextBox(self, label="Input file")

        # Select contrast
        lbl_contrasts = ['t1', 't2', 't2s', 'dwi']
        self.rbox_contrast = wx.RadioBox(self, label='Select contrast:',
                                         choices=lbl_contrasts,
                                         majorDimension=1,
                                         style=wx.RA_SPECIFY_ROWS)

        # Display all options
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.hbox_filein.hbox, 0, wx.ALL, 5)
        sizer.Add(self.rbox_contrast, 0, wx.ALL, 5)

        # Run button
        button_run = wx.Button(self, id=wx.ID_ANY, label="Run")
        button_run.Bind(wx.EVT_BUTTON, self.on_button_run)
        sizer.Add(button_run, 0, wx.ALL, 5)

        # Add to main sizer
        self.sizer_h.Add(sizer)
        self.SetSizerAndFit(self.sizer_h)

    def on_button_run(self, event):

        # Build and run SCT command
        fname_input = self.hbox_filein.textctrl.GetValue()
        contrast = self.rbox_contrast.GetStringSelection()
        base_name = os.path.basename(fname_input)
        fname, fext = base_name.split(os.extsep, 1)
        fname_out = "{}_seg.{}".format(fname, fext)
        cmd_line = "sct_deepseg_sc -i {} -c {}".format(fname_input, contrast)
        self.call_sct_command(cmd_line)

        # Add output to the list of overlay
        image = Image(fname_out)  # <class 'fsl.data.image.Image'>
        overlayList.append(image)
        opts = displayCtx.getOpts(image)
        opts.cmap = 'red'


class TabPanelGMSeg(SCTPanel):
    """
    sct_deepseg_gm
    """

    DESCRIPTION = """
    Segment the spinal cord gray matter using deep learning. The convolutional neural network features dilated 
    convolutions and was trained on 232 subjects (3963 axial slices) from multiple centers, and including various 
    pathologies (compression, MS, ALS, etc.). 
    <br><br>
    <b>Usage</b>:
    <br>
    Select an image from the overlay list that has a good white and gray matter contrast (e.g., T2*-weighted image), 
    then click "Run". For more options, please use the Terminal version of this function.
    <br><br>
    <b>Specific citation</b>:
    <br>
    Perone et al. <i>Spinal cord gray matter segmentation using deep dilated convolutions.</i> Sci Rep. 2018
    """

    def __init__(self, parent):
        super(TabPanelGMSeg, self).__init__(parent=parent, id_=wx.ID_ANY)

        # Fetch input file
        self.hbox_filein = TextBox(self, label="Input file")

        # Display all options
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.hbox_filein.hbox, 0, wx.ALL, 5)

        # Run button
        button_run = wx.Button(self, id=wx.ID_ANY, label="Run")
        button_run.Bind(wx.EVT_BUTTON, self.on_button_run)
        sizer.Add(button_run, 0, wx.ALL, 5)

        # Add to main sizer
        self.sizer_h.Add(sizer)
        self.SetSizerAndFit(self.sizer_h)

    def on_button_run(self, event):

        # Build and run SCT command
        fname_input = self.hbox_filein.textctrl.GetValue()
        base_name = os.path.basename(fname_input)
        fname, fext = base_name.split(os.extsep, 1)
        fname_out = "{}_gmseg.{}".format(fname, fext)
        cmd_line = "sct_deepseg_gm -i {} -o {}".format(fname_input, fname_out)
        self.call_sct_command(cmd_line)

        # Add output to the list of overlay
        image = Image(fname_out)  # <class 'fsl.data.image.Image'>
        overlayList.append(image)
        opts = displayCtx.getOpts(image)
        opts.cmap = 'yellow'


class TabPanelVertLB(SCTPanel):
    """
    sct_label_vertebrae
    """

    DESCRIPTION = """
    Automatically find intervertebral discs and label an input segmentation with vertebral levels. The values on the 
    output labeled segmentation corresponds to the level, e.g., 2 corresponds to C2, 8 corresponds to T1, etc. 
    <br><br>
    <b>Usage</b>:
    <br>
    Select an image from the overlay list where discs are clearly visible (e.g., T1w or T2w scans are usually good for
    this task). Then, select a segmentation associated with the image, select the appropriate contrast and click "Run". 
    For more options, please use the Terminal version of this function.
    <br><br>
    <b>Specific citation</b>:
    <br>
    Ullmann et al. <i>Automatic labeling of vertebral levels using a robust template-based approach.</i> Int J Biomed 
    Imaging 2014
    """

    def __init__(self, parent):
        super(TabPanelVertLB, self).__init__(parent=parent, id_=wx.ID_ANY)

        # Fetch input files
        self.hbox_im = TextBox(self, label="Input image")
        self.hbox_seg = TextBox(self, label="Input segmentation")

        # Select contrast
        lbl_contrasts = ['t1', 't2']
        self.rbox_contrast = wx.RadioBox(self, label='Select contrast:',
                                         choices=lbl_contrasts,
                                         majorDimension=1,
                                         style=wx.RA_SPECIFY_ROWS)

        # Display all options
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.hbox_im.hbox, 0, wx.ALL, 5)
        sizer.Add(self.hbox_seg.hbox, 0, wx.ALL, 5)
        sizer.Add(self.rbox_contrast, 0, wx.ALL, 5)

        # Run button
        button_run = wx.Button(self, id=wx.ID_ANY, label="Run")
        button_run.Bind(wx.EVT_BUTTON, self.on_button_run)
        sizer.Add(button_run, 0, wx.ALL, 5)

        # Add to main sizer
        self.sizer_h.Add(sizer)
        self.SetSizerAndFit(self.sizer_h)

    def on_button_run(self, event):

        # Build and run SCT command
        fname_im = self.hbox_im.textctrl.GetValue()
        fname_seg = self.hbox_seg.textctrl.GetValue()
        contrast = self.rbox_contrast.GetStringSelection()

        base_name = os.path.basename(fname_seg)
        fname, fext = base_name.split(os.extsep, 1)
        fname_out = "{}_labeled.{}".format(fname, fext)
        cmd_line = "sct_label_vertebrae -i {} -s {} -c {}".format(fname_im, fname_seg, contrast)
        self.call_sct_command(cmd_line)

        # Add output to the list of overlay
        image = Image(fname_out)  # <class 'fsl.data.image.Image'>
        overlayList.append(image)
        opts = displayCtx.getOpts(image)
        opts.cmap = 'subcortical'


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
        self.t1.Bind(wx.EVT_TEXT, self.get_box_text)
        hbox1.Add(self.t1, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)

        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        l2 = wx.StaticText(self, wx.ID_ANY, "Output Table Name")
        hbox2.Add(l2, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)
        self.t2 = wx.TextCtrl(self)
        self.t2.Bind(wx.EVT_TEXT, self.get_box_text)
        hbox2.Add(self.t2, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.rbox_perlevel, 0, wx.ALL, 5)
        sizer.Add(hbox1)
        sizer.Add(hbox2)
        sizer.Add(self.rbox_vertfile, 0, wx.ALL, 5)
        sizer.Add(button_gm, 0, wx.ALL, 5)
        self.sizer_h.Add(sizer)
        self.SetSizerAndFit(self.sizer_h)

    def get_box_text(self, event):
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
        self.t1.Bind(wx.EVT_TEXT, self.get_box_text)
        hbox1.Add(self.t1, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(hbox1)
        sizer.Add(button_gm, 0, wx.ALL, 5)
        self.sizer_h.Add(sizer)
        self.SetSizerAndFit(self.sizer_h)

    def get_box_text(self, event):
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

#
# def get_highlighted_file_name(self, event):
#     """
#     Fetch path to file highlighted in the Overlay list. displayCtx is a hidden class from FSLeyes.
#     :return: filename_path
#     """
#     selected_overlay = displayCtx.getSelectedOverlay()
#     filename_path = selected_overlay.dataSource
#     print("Fetched file name: {}".format(filename_path))
#     self.t1.SetValue(filename_path)
#
#
# def add_fetch_file_button(hbox, label=""):
#     """
#     Add a button and a text box where user can fetch any highlighted file name from the overlay list.
#     :param label: Text on the button
#     :return: BoxSizer object: hbox:
#     """
#     hbox = wx.BoxSizer(wx.HORIZONTAL)
#     button_fetch_file = wx.Button(self, -1, label=label)
#     button_fetch_file.Bind(wx.EVT_BUTTON, self.get_highlighted_file_name)
#     hbox.Add(button_fetch_file, proportion=0, flag=wx.ALIGN_LEFT | wx.ALL, border=5)
#     t1 = wx.TextCtrl(self, -1, "", wx.DefaultPosition, wx.Size(1000, 10))
#     hbox.Add(t1, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL, 5)
#     return hbox


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
