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

# TODO: add keyboard shortcuts to Run (ctrl+R)
# TODO: add help when user leaves cursor on button

import os
import select
import subprocess
import signal
from threading import Thread
import logging
import webbrowser

import wx
import wx.lib.agw.aui as aui
import wx.html as html

logger = logging.getLogger(__name__)
aui_manager = frame.getAuiManager() # from FSLeyes context

class ErrorDialog(wx.Dialog):
    """
    Panel to display if there is an error, instructing user what to do.
    """
    def __init__(self, parent, msg=None):
        wx.Dialog.__init__(self, parent, title="An Error Occurred")

        self.SetSize((600, 275))

        if msg is None:
            msg = "An error has occurred while running SCT. Please go to the Terminal, copy all the content and paste it as a new issue in SCT's forum: \
            http://forum.spinalcordmri.org/"

        vbox = wx.BoxSizer(wx.VERTICAL)

        error_msg_box = wx.TextCtrl(self, wx.ID_ANY, size=(500,150),
                          style = wx.TE_MULTILINE|wx.TE_READONLY|wx.HSCROLL)

        error_msg_box.AppendText(msg)
        vbox.Add(error_msg_box, 0, wx.TOP|wx.EXPAND, 20)

        btns = self.CreateSeparatedButtonSizer(wx.OK)
        vbox.Add(btns, 0, wx.CENTER|wx.ALL, 10)

        hbox = wx.BoxSizer(wx.HORIZONTAL)

        save_ico = wx.ArtProvider.GetBitmap(wx.ART_ERROR, wx.ART_TOOLBAR, (50, 50))
        img_info = wx.StaticBitmap(self, -1, save_ico, wx.DefaultPosition, (save_ico.GetWidth(), save_ico.GetHeight()))

        hbox.Add(img_info, 0, wx.ALL, 20)
        hbox.Add(vbox, 0, wx.ALL, 0)

        self.SetSizer(hbox)
        self.Centre()
        self.CenterOnParent()


class ProgressDialog(wx.Dialog):
    """
    Panel to display while running SCT command.
    """
    def __init__(self, parent):
        self.stop_run = False
        wx.Dialog.__init__(self, parent, title="SCT Processing")
        self.SetSize((300, 120))

        vbox = wx.BoxSizer(wx.VERTICAL)
        lbldesc = wx.StaticText(self, id=wx.ID_ANY, label="Processing, please wait...")
        vbox.Add(lbldesc, 0, wx.ALIGN_CENTER|wx.ALL, 10)

        stop_button = wx.Button(self, wx.ID_CANCEL, 'Stop')
        vbox.Add(stop_button, 0, wx.CENTER|wx.ALL, 10)
        hbox = wx.BoxSizer(wx.HORIZONTAL)

        # TODO: use a nicer image, showing two gears (similar to ID_EXECUTE)
        save_ico = wx.ArtProvider.GetBitmap(wx.ART_INFORMATION, wx.ART_TOOLBAR, (50, 50))
        img_info = wx.StaticBitmap(self, -1, save_ico, wx.DefaultPosition, (save_ico.GetWidth(), save_ico.GetHeight()))

        hbox.Add(img_info, 0, wx.ALL, 10)
        hbox.Add(vbox, 0, wx.ALL, 0)

        self.SetSizer(hbox)
        self.Centre()
        self.CenterOnParent()

        stop_button.Bind(wx.EVT_BUTTON, self.OnStop)

    def OnStop(self, event):
        print(f"Stop was pressed. event={event}")
        self.stop_run = True
        self.Destroy()


class SCTCallThread(Thread):
    def __init__(self, command, text_window_ctrl):
        Thread.__init__(self)
        self.command = [command]
        self.status = None
        self.stdout = ""
        self.stderr = ""
        self.text_window = text_window_ctrl

    def sct_call(self, command):
        env = os.environ.copy()
        if 'PYTHONHOME' in env:
            del env["PYTHONHOME"]
        if 'PYTHONPATH' in env:
            del env["PYTHONPATH"]

        proc = subprocess.Popen(self.command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, env=env)
        self.p = proc

        stdout_fd = proc.stdout.fileno()
        stderr_fd = proc.stderr.fileno()
        os.set_blocking(stdout_fd, False)
        os.set_blocking(stderr_fd, False)

        while proc.poll() is None:
            timeout = 1
            rs = [ proc.stdout, proc.stderr ]
            ws = []
            xs = []

            rs, ws, xs = select.select(rs, ws, xs, timeout)
            for r in rs:
                msg = None
                if r is proc.stdout:
                    msg = os.read(stdout_fd, 1024)
                    if msg:
                        self.stdout += msg.decode('utf-8')

                elif r is proc.stderr:
                    msg = os.read(stderr_fd, 1024)
                    if msg:
                        self.stderr += msg.decode('utf-8')

                if msg:
                    wx.CallAfter(self.text_window.WriteText, msg)

        return proc.returncode, self.stdout, self.stderr

    def sct_interrupt(self):
        if self.p:
            self.p.send_signal(signal.SIGINT)
        else:
            print("No process running?")

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
        self.textctrl = wx.TextCtrl(sctpanel)
        self.hbox = wx.BoxSizer(wx.HORIZONTAL)

        button_fetch_file = wx.Button(sctpanel, -1, label=label)
        button_fetch_file.Bind(wx.EVT_BUTTON, self.get_highlighted_file_name)

        self.hbox.Add(button_fetch_file, 0, wx.ALIGN_LEFT| wx.RIGHT, 10)
        self.hbox.Add(self.textctrl, 1, wx.ALIGN_LEFT|wx.LEFT, 10)

    def get_highlighted_file_name(self, event):
        """
        Fetch path to file highlighted in the Overlay list.
        """
        selected_overlay = displayCtx.getSelectedOverlay()  # displayCtx is a class from FSLeyes
        filename_path = selected_overlay.dataSource
        print("Fetched file name: {}".format(filename_path))
        self.textctrl.SetValue(filename_path)

    def get_file_name(self):
        return self.textctrl.GetValue()


# Creates the standard panel for each tool
class SCTPanel(wx.Panel):
    """
    Creates the standard panel for each tool
    :param sizer_h: Main wx.BoxSizer object that encloses SCT information, for each panel
    """

    DESCRIPTION_SCT = """
    <br><br><b>General citation (please always cite)</b>:<br>
    De Leener B, Levy S, Dupont SM, Fonov VS, Stikov N, Louis Collins D, Callot V,
    Cohen-Adad J. <i>SCT: Spinal Cord Toolbox, an open-source software for processing
    spinal cord MRI data</i>. Neuroimage. 2017 Jan 15;145(Pt A):24-43.
    """

    SCT_DIR_ENV = 'SCT_DIR'
    SCT_LOGO_REL_PATH = 'documentation/imgs/logo_sct_small.png'
    SCT_TUTORIAL_PATH = 'documentation/Manual_v1_SCT.pdf'  # TODO: fix this path

    def __init__(self, parent, id_):
        super(SCTPanel, self).__init__(parent=parent, id=id_)

        # main layout consists of one row with 3 main columns
        self.main_row = wx.BoxSizer(wx.HORIZONTAL)

        self.column_left = wx.BoxSizer(wx.VERTICAL)
        self.column_center = wx.BoxSizer(wx.VERTICAL)
        self.column_right = wx.BoxSizer(wx.VERTICAL)

        sct_logo = self.get_logo()
        logo_help_hbox = wx.BoxSizer(wx.HORIZONTAL)
        logo_help_hbox.Add(sct_logo, 1, wx.HORIZONTAL, 5)

        button_help = wx.Button(self, id=id_, label="Help")
        button_help.Bind(wx.EVT_BUTTON, self.help_url)
        logo_help_hbox.Add(button_help, 0, wx.ALIGN_BOTTOM|wx.LEFT, 90)

        self.column_left.Add(logo_help_hbox, proportion=0, flag=wx.ALL, border=5)

        html_desc_window = self.get_description()
        self.column_left.Add(html_desc_window, 0, wx.ALL, 5)

        self.log_window = wx.TextCtrl(self, wx.ID_ANY, size=(100, 300),
                          style = wx.TE_MULTILINE|wx.TE_READONLY|wx.HSCROLL)

        self.column_right.Add(self.log_window, 1, wx.EXPAND|wx.ALL, 5)

        self.main_row.Add(self.column_left, 0, wx.ALL, 10)
        self.main_row.Add(self.column_center, 1, wx.ALL, 10)
        self.main_row.Add(self.column_right, 1, wx.ALL, 10)

        self.SetSizerAndFit(self.main_row)

    def log_to_window(self, msg, level=None):
        if level is None:
            self.log_window.AppendText("{}\n".format(msg))
        else:
            self.log_window.AppendText("{}: {}\n".format(level, msg))

    def tutorial(self,event):
        pdfpath = os.path.join(os.environ[self.SCT_DIR_ENV],self.SCT_TUTORIAL_PATH)
        print('PDF path:', pdfpath)
        cmd_line = "open {}".format(pdfpath)
        print('Command line:', cmd_line)
        self.call_sct_command(cmd_line)

    def help_url(self, event):
        url = "http://forum.spinalcordmri.org/c/sct"
        webbrowser.open(url)

    def get_logo(self):
        logo_file = os.path.join(os.environ[self.SCT_DIR_ENV],
                                 self.SCT_LOGO_REL_PATH)
        png = wx.Image(logo_file,
                       wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        img_logo = wx.StaticBitmap(self, -1, png, wx.DefaultPosition,
                                   (png.GetWidth(), png.GetHeight()))
        return img_logo

    def get_description(self):
        txt_style = wx.VSCROLL | \
                    wx.HSCROLL | wx.TE_READONLY | \
                    wx.BORDER_SIMPLE
        htmlw = html.HtmlWindow(self, wx.ID_ANY,
                                size=(400, 220),
                                style=txt_style)
        htmlw.SetPage(self.DESCRIPTION + self.DESCRIPTION_SCT)
        htmlw.SetStandardFonts(size=10, normal_face="Noto Sans")

        return htmlw

    def call_sct_command(self, command):
        self.log_to_window("Running: {}".format(command), level="INFO")
        progress_dialog = ProgressDialog(frame)
        progress_dialog.Show()

        thr = SCTCallThread(command, self.log_window)
        thr.start()

        # No access to app.pending() from here
        while True:
            thr.join(0.1)
            wx.Yield()
            if not thr.isAlive():
                break
            if progress_dialog.stop_run:
                thr.sct_interrupt()

        thr.join()

        self.log_to_window("Command completed.", level="INFO")

        if progress_dialog:
            progress_dialog.Destroy()

        # show stderr output if an error occurred
        if thr.status:
            self.log_to_window("An error occurred", level="ERROR")
            error_dialog = ErrorDialog(frame, msg=thr.stderr)
            error_dialog.Show()


class TabPanelPropSeg(SCTPanel):
    """
    sct_propseg
    """

    DESCRIPTION = """
    <b>Function description</b>:<br>
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

        self.hbox_filein = TextBox(self, label="Input file")
        lbl_contrasts = ['t1', 't2', 't2s', 'dwi']

        self.rbox_contrast = wx.RadioBox(self, label='Select contrast:',
                                         choices=lbl_contrasts,
                                         majorDimension=1,
                                         style=wx.RA_SPECIFY_ROWS)

        button_run = wx.Button(self, id=wx.ID_ANY, label="Run")
        button_run.Bind(wx.EVT_BUTTON, self.on_button_run)

        self.column_center.Add(self.hbox_filein.hbox, 0, wx.EXPAND|wx.ALL, 5)
        self.column_center.Add(self.rbox_contrast, 0, wx.ALL, 5)
        self.column_center.Add(button_run, 0, wx.ALL, 5)

    def on_button_run(self, event):

        # Build and run SCT command
        fname_input = self.hbox_filein.get_file_name()
        if not fname_input:
            msg = "No input file selected! Select a file from the overlay list and then press Input file."
            self.log_to_window(msg, level="ERROR")
            error_dialog = ErrorDialog(frame, msg=msg)
            error_dialog.Show()
            return

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
    <b>Function description</b>:<br>
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

        self.hbox_filein = TextBox(self, label="Input file")
        lbl_contrasts = ['t1', 't2', 't2s', 'dwi']
        self.rbox_contrast = wx.RadioBox(self, label='Select contrast:',
                                         choices=lbl_contrasts,
                                         majorDimension=1,
                                         style=wx.RA_SPECIFY_ROWS)

        button_run = wx.Button(self, id=wx.ID_ANY, label="Run")
        button_run.Bind(wx.EVT_BUTTON, self.on_button_run)

        self.column_center.Add(self.hbox_filein.hbox, 0, wx.EXPAND|wx.ALL, 5)
        self.column_center.Add(self.rbox_contrast, 0, wx.ALL, 5)
        self.column_center.Add(button_run, 0, wx.ALL, 5)

    def on_button_run(self, event):

        # Build and run SCT command
        fname_input = self.hbox_filein.get_file_name()
        if not fname_input:
            msg = "No input file selected! Select a file from the overlay list and then press Input file."
            self.log_to_window(msg, level="ERROR")
            error_dialog = ErrorDialog(frame, msg=msg)
            error_dialog.Show()
            return

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
    <b>Function description</b>:<br>
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

        self.hbox_filein = TextBox(self, label="Input file")

        button_run = wx.Button(self, id=wx.ID_ANY, label="Run")
        button_run.Bind(wx.EVT_BUTTON, self.on_button_run)

        self.column_center.Add(self.hbox_filein.hbox, 0, wx.EXPAND|wx.ALL, 5)
        self.column_center.Add(button_run, 0, wx.ALL, 5)

    def on_button_run(self, event):

        # Build and run SCT command
        fname_input = self.hbox_filein.get_file_name()
        if not fname_input:
            msg = "No input file selected! Select a file from the overlay list and then press Input file."
            self.log_to_window(msg, level="ERROR")
            error_dialog = ErrorDialog(frame, msg=msg)
            error_dialog.Show()
            return

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
    <b>Function description</b>:<br>
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

        self.hbox_im = TextBox(self, label="Input image")
        self.hbox_seg = TextBox(self, label="Input segmentation")

        lbl_contrasts = ['t1', 't2']
        self.rbox_contrast = wx.RadioBox(self, label='Select contrast:',
                                         choices=lbl_contrasts,
                                         majorDimension=1,
                                         style=wx.RA_SPECIFY_ROWS)

        # Run button
        button_run = wx.Button(self, id=wx.ID_ANY, label="Run")
        button_run.Bind(wx.EVT_BUTTON, self.on_button_run)

        self.column_center.Add(self.hbox_im.hbox, 0, wx.EXPAND|wx.ALL, 5)
        self.column_center.Add(self.hbox_seg.hbox, 0, wx.EXPAND|wx.ALL, 5)
        self.column_center.Add(self.rbox_contrast, 0, wx.ALL, 5)
        self.column_center.Add(button_run, 0, wx.ALL, 5)

    def on_button_run(self, event):

        # Build and run SCT command
        fname_im = self.hbox_im.textctrl.GetValue()
        fname_seg = self.hbox_seg.textctrl.GetValue()

        fname_im = self.hbox_im.get_file_name()
        if not fname_im:
            msg = "No input image selected! Select an image from the overlay list and then press Input image."
            self.log_to_window(msg, level="ERROR")
            error_dialog = ErrorDialog(frame, msg=msg)
            error_dialog.Show()
            return

        fname_seg = self.hbox_seg.get_file_name()
        if not fname_seg:
            msg = "No input segmentation selected! Select a segmentation file from the overlay list and then press Input segmentation."
            self.log_to_window(msg, level="ERROR")
            error_dialog = ErrorDialog(frame, msg=msg)
            error_dialog.Show()
            return

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
    <b>Function description</b>:<br>
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

        self.hbox_im = TextBox(self, label="Input image")
        self.hbox_seg = TextBox(self, label="Input segmentation")
        self.hbox_label = TextBox(self, label="Input labels")

        lbl_contrasts = ['t1', 't2']
        self.rbox_contrast = wx.RadioBox(self, label='Select contrast:',
                                         choices=lbl_contrasts,
                                         majorDimension=1,
                                         style=wx.RA_SPECIFY_ROWS)

        button_run = wx.Button(self, id=wx.ID_ANY, label="Run")
        button_run.Bind(wx.EVT_BUTTON, self.on_button_run)

        self.column_center.Add(self.hbox_im.hbox, 0, wx.EXPAND|wx.ALL, 5)
        self.column_center.Add(self.hbox_seg.hbox, 0, wx.EXPAND|wx.ALL, 5)
        self.column_center.Add(self.hbox_label.hbox, 0, wx.EXPAND|wx.ALL, 5)
        self.column_center.Add(self.rbox_contrast, 0, wx.ALL, 5)
        self.column_center.Add(button_run, 0, wx.ALL, 5)

    def on_button_run(self, event):

        # Build and run SCT command
        fname_im = self.hbox_im.textctrl.GetValue()
        fname_seg = self.hbox_seg.textctrl.GetValue()
        fname_label = self.hbox_label.textctrl.GetValue()

        fname_im = self.hbox_im.textctrl.GetValue()
        fname_seg = self.hbox_seg.textctrl.GetValue()

        fname_im = self.hbox_im.get_file_name()
        if not fname_im:
            msg = "No input image selected! Select an image from the overlay list and then press Input image."
            self.log_to_window(msg, level="ERROR")
            error_dialog = ErrorDialog(frame, msg=msg)
            error_dialog.Show()
            return

        fname_seg = self.hbox_seg.get_file_name()
        if not fname_seg:
            msg = "No input segmentation selected! Select a segmentation file from the overlay list and then press Input segmentation."
            self.log_to_window(msg, level="ERROR")
            error_dialog = ErrorDialog(frame, msg=msg)
            error_dialog.Show()
            return

        fname_label = self.hbox_label.get_file_name()
        if not fname_label:
            msg = "No input labels selected! Select input labels from the overlay list and then press Input labels."
            self.log_to_window(msg, level="ERROR")
            error_dialog = ErrorDialog(frame, msg=msg)
            error_dialog.Show()
            return

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
    notebook = aui.AuiNotebook(parent=window)
    panel_propseg = TabPanelPropSeg(parent=notebook)
    panel_sc = TabPanelSCSeg(parent=notebook)
    panel_gm = TabPanelGMSeg(parent=notebook)
    panel_vlb = TabPanelVertLB(parent=notebook)
    panel_reg = TabPanelRegisterToTemplate(parent=notebook)

    notebook.AddPage(page=panel_propseg, caption="sct_propseg", select=True)
    notebook.AddPage(page=panel_sc, caption="sct_deepseg_sc", select=False)
    notebook.AddPage(page=panel_gm, caption="sct_deepseg_gm", select=False)
    notebook.AddPage(page=panel_vlb, caption="sct_label_vertebrae", select=False)
    notebook.AddPage(page=panel_reg, caption="sct_register_to_template", select=False)

    aui_manager.AddPane(notebook, aui.AuiPaneInfo().Name("notebook_content").CenterPane().PaneBorder(False))
    aui_manager.Update()

run_main()
