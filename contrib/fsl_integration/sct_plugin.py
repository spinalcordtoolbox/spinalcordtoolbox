# coding=utf-8
#########################################################################################
# This code provides SCT integration into FSLeyes for the following tools:
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
# Authors: Christian S. Perone, Thiago JR Rezende, Julien Cohen-Adad
##########################################################################################

# TODO: display error if number of input arguments is not correct
# TODO: add shortcuts to Run
# TODO: add help when user leaves cursor on button

import os
import subprocess
from threading import Thread
import logging

import wx
import wx.lib.agw.aui as aui
import wx.html as html

logger = logging.getLogger(__name__)
aui_manager = frame.getAuiManager()


class ErrorDialog(wx.Dialog):
    """
    Panel to display if there is an error, instructing user what to do.
    """
    def __init__(self, parent):
        wx.Dialog.__init__(self, parent, title="SCT Error")
        self.SetSize((510, 170))

        vbox = wx.BoxSizer(wx.VERTICAL)
        lbldesc = wx.StaticText(self,
                                id=-1,
                                label="An error has occurred while running SCT. Please go to the Terminal, copy all "
                                      "the content and paste it as a new issue in SCT's forum: \n"
                                      "http://forum.spinalcordmri.org/",
                                size=wx.Size(470, 60),
                                style=wx.ALIGN_LEFT)
        vbox.Add(lbldesc, 0, wx.ALIGN_LEFT | wx.ALL, 10)

        btns = self.CreateSeparatedButtonSizer(wx.OK)
        vbox.Add(btns, 0, wx.ALIGN_LEFT | wx.ALL, 5)

        hbox = wx.BoxSizer(wx.HORIZONTAL)

        save_ico = wx.ArtProvider.GetBitmap(wx.ART_ERROR, wx.ART_TOOLBAR, (50, 50))
        img_info = wx.StaticBitmap(self, -1, save_ico, wx.DefaultPosition, (save_ico.GetWidth(), save_ico.GetHeight()))

        hbox.Add(img_info, 0, wx.ALL, 10)
        hbox.Add(vbox, 0, wx.ALL, 0)

        self.SetSizer(hbox)
        self.Centre()
        self.CenterOnParent()


class ProgressDialog(wx.Dialog):
    """
    Panel to display while running SCT command.
    """
    def __init__(self, parent):
        # TODO: try to use MessageBox instead, as they already include buttons, icons, etc.
        wx.Dialog.__init__(self, parent, title="SCT Processing")
        self.SetSize((300, 120))

        vbox = wx.BoxSizer(wx.VERTICAL)
        lbldesc = wx.StaticText(self, id=wx.ID_ANY, label="Processing, please wait...")
        vbox.Add(lbldesc, 0, wx.ALIGN_LEFT | wx.ALL, 10)

        btns = self.CreateSeparatedButtonSizer(wx.CANCEL)
        vbox.Add(btns, 0, wx.ALIGN_LEFT | wx.ALL, 5)

        hbox = wx.BoxSizer(wx.HORIZONTAL)

        # TODO: use a nicer image, showing two gears (similar to ID_EXECUTE)
        save_ico = wx.ArtProvider.GetBitmap(wx.ART_INFORMATION, wx.ART_TOOLBAR, (50, 50))
        img_info = wx.StaticBitmap(self, -1, save_ico, wx.DefaultPosition, (save_ico.GetWidth(), save_ico.GetHeight()))

        hbox.Add(img_info, 0, wx.ALL, 10)
        hbox.Add(vbox, 0, wx.ALL, 0)

        self.SetSizer(hbox)
        self.Centre()
        self.CenterOnParent()
        # TODO: retrieve action from the cancel button


class SCTCallThread(Thread):
    def __init__(self, command):
        Thread.__init__(self)
        self.command = command
        self.status = None
        self.stdout = ""
        self.stderr = ""

    @staticmethod
    def sct_call(command):
        # command="boo"  # for debug
        env = os.environ.copy()
        if 'PYTHONHOME' in env:
            del env["PYTHONHOME"]
        if 'PYTHONPATH' in env:
            del env["PYTHONPATH"]
        p = subprocess.Popen([command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, env=env)
        # TODO: printout process in stdout in real time (instead of dumping the output once done).
        stdout, stderr = [i.decode('utf-8') for i in p.communicate()]
        # TODO: Fix: tqdm progress bar causes the printing of stdout to stop
        print("\n\033[94m{}\033[0m\n".format(stdout))
        if p.returncode is not 0:
            print("\n\033[91mERROR: {}\033[0m\n".format(stderr))
        return p.returncode, stdout, stderr

    def run(self):
        """
        overrides Thread.run() function
        :return:
        """
        self.status, self.stdout, self.stderr = self.sct_call(self.command)


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
        # TODO: instead of this hard-coded 1000 value, extended the text box towards the most right part of the panel
        #  (include a margin)
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
        # TODO: increase the width of the description box
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
                       wx.BITMAP_TYPE_ANY)
        png = png.Scale(png.GetWidth() // 6, png.GetHeight() // 6,
                        wx.IMAGE_QUALITY_HIGH).ConvertToBitmap()
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
        # Open error dialog if stderr is not null
        if thr.status:
            binfo = ErrorDialog(frame)
            binfo.Show()


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


class TabPanelRegisterToTemplate(SCTPanel):
    """
    sct_register_to_template
    """

    DESCRIPTION = """
    Register an image with the default PAM50 spinal cord MRI template. 
    <br><br>
    <b>Usage</b>:
    <br>
    Select an image, its segmentation and a label file. The label file contains single-pixel labels located at the 
    posterior edge of the intervertebral discs. The value of the label corresponds to the lower vertebrae, e.g., label 3
    corresponds to the C2-C3 disc. This label file can be created within FSLeyes by clicking on Tools > Edit mode, then
    Edit > Create mask. Select the "pen", adjust the size to one pixel width and select the proper label value, then 
    click on the image and save the label(s): Overlay > save. Then, select the appropriate contrast and click "Run". 
    For more options, please use the Terminal version of this function.
    <br><br>
    <b>Specific citation</b>:
    <br>
    De Leener et al. <i>PAM50: Unbiased multimodal template of the brainstem and spinal cord aligned with the ICBM152 
    space.</i> Neuroimage 2017
    """

    def __init__(self, parent):
        super(TabPanelRegisterToTemplate, self).__init__(parent=parent, id_=wx.ID_ANY)

        # Fetch input files
        self.hbox_im = TextBox(self, label="Input image")
        self.hbox_seg = TextBox(self, label="Input segmentation")
        self.hbox_label = TextBox(self, label="Input labels")

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
        sizer.Add(self.hbox_label.hbox, 0, wx.ALL, 5)
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
        fname_label = self.hbox_label.textctrl.GetValue()
        contrast = self.rbox_contrast.GetStringSelection()
        cmd_line = \
            "sct_register_to_template -i {} -s {} -ldisc {} -c {}".format(fname_im, fname_seg, fname_label, contrast)
        self.call_sct_command(cmd_line)

        # Add output to the list of overlay
        base_name = os.path.basename(fname_im)
        fname, fext = base_name.split(os.extsep, 1)
        # TODO: at some point we will modify SCT's function to output the file name below
        # fname_out = "PAM50_{}_reg.{}".format(contrast, fext)
        fname_out = 'template2anat.nii.gz'
        image = Image(fname_out)  # <class 'fsl.data.image.Image'>
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
    panel_reg = TabPanelRegisterToTemplate(notebook)

    notebook.AddPage(panel_propseg, "sct_propseg", True)
    notebook.AddPage(panel_sc, "sct_deepseg_sc", False)
    notebook.AddPage(panel_gm, "sct_deepseg_gm", False)
    notebook.AddPage(panel_vlb, "sct_label_vertebrae", False)
    notebook.AddPage(panel_reg, "sct_register_to_template", False)

    aui_manager.AddPane(notebook, aui.AuiPaneInfo().Name("notebook_content").CenterPane().PaneBorder(False))
    aui_manager.Update()


run_main()
