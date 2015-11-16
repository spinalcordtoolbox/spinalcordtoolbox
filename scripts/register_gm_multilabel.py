#!/usr/bin/env python
#
# multi-label registration of spinal cord internal structure
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2013 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Sara Dupont
# Modified: 2015-05-20
#
# About the license: see the file LICENSE.TXT
#########################################################################################
import sys, os, time
from msct_parser import Parser
from msct_image import Image
import sct_utils as sct

def multi_label_reg(fname_gm, fname_wm, path_template, fname_warp_template, fname_target):
    path_template = sct.slash_at_the_end(path_template, 1)
    im_gm = Image(fname_gm)
    im_wm = Image(fname_wm)

    im_template_gm = Image(path_template+'template/MNI-Poly-AMU_GM.nii.gz')
    im_template_wm = Image(path_template+'template/MNI-Poly-AMU_WM.nii.gz')

    # accentuate separation WM/GM
    im_gm.data[im_gm.data > 0.5] = 1
    im_gm.data[im_gm.data < 0.01] = 0
    im_wm.data[im_wm.data > 0.5] = 1
    im_wm.data[im_wm.data < 0.01] = 0
    im_template_gm.data[im_template_gm.data > 0.5] = 1
    im_template_gm.data[im_template_gm.data < 0.01] = 0
    im_template_wm.data[im_template_wm.data > 0.5] = 1
    im_template_wm.data[im_template_wm.data < 0.01] = 0

    im_automatic_ml = im_gm.copy()
    im_template_ml = im_template_gm.copy()

    im_automatic_ml.data = 200*im_gm.data + 100*im_wm.data
    im_template_ml.data = 200*im_template_gm.data + 100*im_template_wm.data

    fname_automatic_ml = 'multilabel_automatic_seg.nii.gz'
    fname_template_ml = 'multilabel_template_seg.nii.gz'

    path_automatic_ml, file_automatic_ml, ext_automatic_ml = sct.extract_fname(fname_automatic_ml)
    path_template_ml, file_template_ml, ext_template_ml = sct.extract_fname(fname_template_ml)
    path_target, file_target, ext_target = sct.extract_fname(fname_target)
    path_warp_template, file_warp_template, ext_warp_template = sct.extract_fname(fname_warp_template)

    im_automatic_ml.setFileName(fname_automatic_ml)
    im_template_ml.setFileName(fname_template_ml)

    tmp_dir = 'tmp_multilabel_reg'+'_'+time.strftime("%y%m%d%H%M%S")+'/'  # TODO: add tmp dir name
    sct.run('mkdir '+tmp_dir)

    sct.run('cp '+fname_target+' '+tmp_dir+file_target+ext_target)
    sct.run('cp '+fname_warp_template+' '+tmp_dir+file_warp_template+ext_warp_template)

    os.chdir(tmp_dir)
    # TODO assert RPI, if not, change orientation
    im_automatic_ml.save()
    im_template_ml.save()

    fname_automatic_ml_smooth = file_automatic_ml+'_smooth'+ext_automatic_ml
    sct.run('sct_maths -i '+fname_automatic_ml+' -smooth 0.8,0.8,0 -o '+fname_automatic_ml_smooth)
    fname_automatic_ml = fname_automatic_ml_smooth
    path_automatic_ml, file_automatic_ml, ext_automatic_ml = sct.extract_fname(fname_automatic_ml)

    sct.run('sct_register_multimodal -i '+fname_template_ml+' -d '+fname_automatic_ml+' -p step=1,algo=slicereg,metric=MeanSquares,step=2,algo=syn,metric=MeanSquares,iter=2:step=3,algo=bsplinesyn,metric=MeanSquares,iter=5,smooth=1') #TODO: complete params
    sct.run('sct_concat_transfo -w '+file_warp_template+ext_warp_template+',warp_'+file_template_ml+'2'+file_automatic_ml+'.nii.gz -d '+file_target+ext_target+' -o warp_template2'+file_target+'_wm_corrected_multilabel.nii.gz')
    sct.run('sct_warp_template -d '+fname_target+' -w warp_template2'+file_target+'_wm_corrected_multilabel.nii.gz')
    os.chdir('..')

    sct.run('cp -r '+tmp_dir+'label  ./label_wm_corrected_multilabel')
    sct.generate_output_file(tmp_dir+'warp_'+file_template_ml+'2'+file_automatic_ml+'.nii.gz', './warp_'+file_template_ml+'2'+file_automatic_ml+'.nii.gz')
    sct.generate_output_file(tmp_dir+'warp_'+file_automatic_ml+'2'+file_template_ml+'.nii.gz', './warp_'+file_automatic_ml+'2'+file_template_ml+'.nii.gz')

    sct.printv('fslview '+fname_target+' label_wm_corrected_multilabel/template/MNI-Poly-AMU_GM.nii.gz -l Red-Yellow -b 0.5,1 label_wm_corrected_multilabel/template/MNI-Poly-AMU_WM.nii.gz -l Blue-Lightblue -b 0.5,1 &', 1, 'info')


########################################################################################################################
def get_parser():
    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Multi-label registration\n')
    parser.add_option(name="-gm",
                      type_value="file",
                      description="Gray matter automatic segmentation",
                      mandatory=True,
                      example='t2star_gmseg.nii.gz')
    parser.add_option(name="-wm",
                      type_value="file",
                      description="White matter automatic segmentation",
                      mandatory=True,
                      example='t2star_wmseg.nii.gz')
    parser.add_option(name="-t",
                      type_value="folder",
                      description="Path to template (registered on target image)",
                      mandatory=True,
                      example='label/')
    parser.add_option(name="-d",
                      type_value="file",
                      description="Target image (WM/GM contrasted with WM, GM automatic segmentations",
                      mandatory=True,
                      example='t2star.nii.gz')
    parser.add_option(name="-w",
                      type_value="file",
                      description="Warping field template --> target image",
                      mandatory=True,
                      example='warp_template2t2star.nii.gz')

    parser.usage.addSection('MISC')
    parser.add_option(name='-qc',
                      type_value='multiple_choice',
                      description='Output images for quality control.',
                      mandatory=False,
                      example=['0', '1'],
                      default_value='1')
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description='Remove temporary files.',
                      mandatory=False,
                      default_value='1',
                      example=['0', '1'])
    parser.add_option(name="-v",
                      type_value='multiple_choice',
                      description="verbose: 0 = nothing, 1 = classic, 2 = expended",
                      mandatory=False,
                      example=['0', '1', '2'],
                      default_value='1')

    return parser


if __name__ == "__main__":
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])
    fname_gm = arguments['-gm']
    fname_wm = arguments['-wm']
    path_template = arguments['-t']
    fname_warp_template = arguments['-w']
    fname_target = arguments['-d']

    multi_label_reg(fname_gm, fname_wm, path_template, fname_warp_template, fname_target)