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
import inspect
import subprocess

import wx
import wx.lib.agw.aui as aui


class TabPanelGMSeg(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)

        self.SetInitialSize((100, 100))
        self.SetSize((100, 100))

        sizer = wx.BoxSizer(wx.VERTICAL)
        button_gm = wx.Button(self, id=wx.ID_ANY,
                              label="Gray Matter Segmentation")
        button_gm.Bind(wx.EVT_BUTTON, self.onButtonGM)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer_h = wx.BoxSizer(wx.HORIZONTAL)

        sizer.Add(button_gm, 0, wx.ALL, 5)

        logo_file = os.path.join(os.environ['SCT_DIR'],
                                 'documentation/imgs/logo_sct.png')
        png = wx.Image(logo_file,
                       wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        self.img_logo = wx.StaticBitmap(self, -1, png, (10, 5),
                                        (png.GetWidth(), png.GetHeight()))
        sizer_h.Add(self.img_logo, 0, wx.ALL, 5)
        sizer_h.Add(sizer)

        self.SetSizerAndFit(sizer_h)

    def onButtonGM(self, event):
        selected_overlay = displayCtx.getSelectedOverlay()
        filename_path = selected_overlay.dataSource
        cmd_line = ["sct_deepseg_gm -i {} -o seg.nii.gz".format(filename_path)]
        env = os.environ.copy()
        del env["PYTHONHOME"]
        del env["PYTHONPATH"]
        p = subprocess.Popen(cmd_line, stdout=subprocess.PIPE,
                             shell=True, env=env)
        out, err = p.communicate()

        outfilename = os.path.join(os.getcwd(), 'seg.nii.gz')
        image = Image(outfilename)
        overlayList.append(image)

        display = displayCtx.getDisplay(image)
        opts = displayCtx.getOpts(image)
        opts.cmap = 'red'


class TabPanelSCSeg(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent=parent, id=wx.ID_ANY)

        self.SetInitialSize((100, 100))
        self.SetSize((100, 100))

        sizer = wx.BoxSizer(wx.VERTICAL)
        button_gm = wx.Button(self, id=wx.ID_ANY, label="Spinal Cord Segmentation")
        button_gm.Bind(wx.EVT_BUTTON, self.onButtonSC)

        lbl_contrasts = ['t1', 't2', 't2s', 'dwi']
        self.rbox_contrast = wx.RadioBox(self, label='Select contrast:',
                                         choices=lbl_contrasts,
                                         majorDimension=1,
                                         style=wx.RA_SPECIFY_ROWS)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer_h = wx.BoxSizer(wx.HORIZONTAL)

        logo_file = os.path.join(os.environ['SCT_DIR'],
                                 'documentation/imgs/logo_sct.png')
        png = wx.Image(logo_file,
                       wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        self.img_logo = wx.StaticBitmap(self, -1, png, (10, 5),
                                        (png.GetWidth(), png.GetHeight()))

        sizer_h.Add(self.img_logo, 0, wx.ALL, 5)
        sizer.Add(self.rbox_contrast, 0, wx.ALL, 5)
        sizer.Add(button_gm, 0, wx.ALL, 5)
        sizer_h.Add(sizer)

        self.SetSizerAndFit(sizer_h)

    def onButtonSC(self, event):
        selected_overlay = displayCtx.getSelectedOverlay()
        filename_path = selected_overlay.dataSource
        contrast = self.rbox_contrast.GetStringSelection()

        base_name = os.path.basename(filename_path)
        fname, fext = base_name.split(os.extsep, 1)
        out_name = "{}_seg.{}".format(fname, fext)

        cmd_line = ["sct_deepseg_sc -i {} -c {}".format(filename_path, contrast)]
        env = os.environ.copy()
        del env["PYTHONHOME"]
        del env["PYTHONPATH"]
        p = subprocess.Popen(cmd_line, stdout=subprocess.PIPE, shell=True, env=env)
        out, err = p.communicate()

        outfilename = os.path.join(os.getcwd(), out_name)
        image = Image(outfilename)
        overlayList.append(image)

        display = displayCtx.getDisplay(image)
        opts = displayCtx.getOpts(image)
        opts.cmap = 'red'


def run_main():
    aui_manager = frame.getAuiManager()
    window = aui_manager.GetManagedWindow()

    if 'SCT_DIR' not in os.environ:
        dlg = wx.MessageDialog(window, 'Spinal Cord Toolbox (SCT) was not '
                               'found in your system. Please check the '
                               'installation procedure at https://github'
                               '.com/neuropoly/spinalcordtoolbox',
                               'SCT not found !', wx.OK | wx.ICON_INFORMATION)
        dlg.ShowModal()
        dlg.Destroy()
        return

    notebook = aui.AuiNotebook(window)
    panel_gm = TabPanelGMSeg(notebook)
    panel_sc = TabPanelSCSeg(notebook)

    notebook.AddPage(panel_gm, "sct_deepseg_gm", True)
    notebook.AddPage(panel_sc, "sct_deepseg_sc", False)

    aui_manager.AddPane(notebook, 
                        aui.AuiPaneInfo().Name("notebook_content").
                        CenterPane().PaneBorder(False)) 
    aui_manager.Update()


run_main()

