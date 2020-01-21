# coding=utf-8

#########################################################################################
# This code will provide SCT integration into fsleyes for the following tools:
#
#    - sct_deepseg_gm
#    - sct_deepseg_sc
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2018 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Christian S. Perone
# Created: 2 Mar 2018
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
        if "PYTHONHOME" in os.environ:
            del env["PYTHONHOME"]
        if "PYTHONPATH" in os.environ:
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


class TabPanelSCSeg(SCTPanel):

    DESCRIPTION = """This segmentation tool is based on Deep Learning and
    a 3D U-Net. For more information, please refer to the
    article below.<br><br>
    <b>Specific citation</b>:<br>
    (TODO: add Charley paper)
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

    notebook = aui.AuiNotebook(window)
    panel_sc = TabPanelSCSeg(notebook)
    panel_gm = TabPanelGMSeg(notebook)

    notebook.AddPage(panel_sc, "sct_deepseg_sc", True)
    notebook.AddPage(panel_gm, "sct_deepseg_gm", False)

    aui_manager.AddPane(notebook, 
                        aui.AuiPaneInfo().Name("notebook_content").
                        CenterPane().PaneBorder(False)) 
    aui_manager.Update()
    

run_main()
