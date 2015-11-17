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
#TODO: add remove tmp

class Param:
    def __init__(self):
        self.thr = 0.5
        self.gap = (100, 200)
        self.smooth = 0.8

        self.param_reg = 'step=1,algo=slicereg,metric=MeanSquares,step=2,algo=syn,metric=MeanSquares,iter=2:step=3,algo=bsplinesyn,metric=MeanSquares,iter=5,smooth=1'

        self.output_folder = './'
        self.verbose = 1
        self.remove_tmp = 1
        self.qc = 1


class MultiLabelRegistration:
    def __init__(self, fname_gm, fname_wm, path_template, fname_warp_template, fname_target, param=None):
        if param is None:
            self.param = Param()
        else:
            self.param = param
        self.im_gm = Image(fname_gm)
        self.im_wm = Image(fname_wm)
        self.path_template = sct.slash_at_the_end(path_template, 1)
        self.im_template_gm = Image(self.path_template+'template/MNI-Poly-AMU_GM.nii.gz')
        self.im_template_wm = Image(self.path_template+'template/MNI-Poly-AMU_WM.nii.gz')

        self.fname_target = fname_target
        self.fname_warp_template2target = fname_warp_template

        self.path_new_template = 'label_wm_corrected_multilabel/'

    def register(self):
        # accentuate separation WM/GM
        self.im_gm = thr_im(self.im_gm, 0.01, self.param.thr)
        self.im_wm = thr_im(self.im_wm, 0.01, self.param.thr)
        self.im_template_gm = thr_im(self.im_template_gm, 0.01, self.param.thr)
        self.im_template_wm = thr_im(self.im_template_wm, 0.01, self.param.thr)

        # create multilabel images
        im_automatic_ml = self.im_gm.copy()
        im_template_ml = self.im_template_gm.copy()

        im_automatic_ml.data = self.param.gap[1]*self.im_gm.data + self.param.gap[0]*self.im_wm.data
        im_template_ml.data = self.param.gap[1]*self.im_template_gm.data + self.param.gap[0]*self.im_template_wm.data

        fname_automatic_ml = 'multilabel_automatic_seg.nii.gz'
        fname_template_ml = 'multilabel_template_seg.nii.gz'

        path_automatic_ml, file_automatic_ml, ext_automatic_ml = sct.extract_fname(fname_automatic_ml)
        path_template_ml, file_template_ml, ext_template_ml = sct.extract_fname(fname_template_ml)
        path_target, file_target, ext_target = sct.extract_fname(fname_target)
        path_warp_template, file_warp_template, ext_warp_template = sct.extract_fname(fname_warp_template)

        im_automatic_ml.setFileName(fname_automatic_ml)
        im_template_ml.setFileName(fname_template_ml)

        # Create temporary folder and put files in it
        tmp_dir = sct.tmp_create()

        sct.run('cp '+fname_target+' '+tmp_dir+file_target+ext_target)
        sct.run('cp '+fname_warp_template+' '+tmp_dir+file_warp_template+ext_warp_template)

        os.chdir(tmp_dir)
        # TODO assert RPI, if not, change orientation
        im_automatic_ml.save()
        im_template_ml.save()

        fname_automatic_ml_smooth = file_automatic_ml+'_smooth'+ext_automatic_ml
        sct.run('sct_maths -i '+fname_automatic_ml+' -smooth '+str(self.param.smooth)+','+str(self.param.smooth)+',0 -o '+fname_automatic_ml_smooth)
        fname_automatic_ml = fname_automatic_ml_smooth
        path_automatic_ml, file_automatic_ml, ext_automatic_ml = sct.extract_fname(fname_automatic_ml)

        # Register multilabel images together
        sct.run('sct_register_multimodal -i '+fname_template_ml+' -d '+fname_automatic_ml+' -p '+self.param.param_reg) #TODO: complete params
        sct.run('sct_concat_transfo -w '+file_warp_template+ext_warp_template+',warp_'+file_template_ml+'2'+file_automatic_ml+'.nii.gz -d '+file_target+ext_target+' -o warp_template2'+file_target+'_wm_corrected_multilabel.nii.gz')
        sct.run('sct_warp_template -d '+fname_target+' -w warp_template2'+file_target+'_wm_corrected_multilabel.nii.gz')
        os.chdir('..')

        sct.run('cp -r '+tmp_dir+'label  '+self.param.output_folder+self.path_new_template)

        sct.generate_output_file(tmp_dir+'warp_'+file_template_ml+'2'+file_automatic_ml+'.nii.gz', self.param.output_folder+'warp_'+file_template_ml+'2'+file_automatic_ml+'.nii.gz')
        sct.generate_output_file(tmp_dir+'warp_'+file_automatic_ml+'2'+file_template_ml+'.nii.gz', self.param.output_folder+'warp_'+file_automatic_ml+'2'+file_template_ml+'.nii.gz')

        sct.printv('fslview '+fname_target+' '+self.param.output_folder+self.path_new_template+'template/MNI-Poly-AMU_GM.nii.gz -l Red-Yellow -b 0.5,1 '+self.param.output_folder+self.path_new_template+'template/MNI-Poly-AMU_WM.nii.gz -l Blue-Lightblue -b 0.5,1 &', 1, 'info')

    def validation(self, fname_manual_gmseg, fname_sc_seg):
        path_manual_gmseg, file_manual_gmseg, ext_manual_gmseg = sct.extract_fname(fname_manual_gmseg)
        path_sc_seg, file_sc_seg, ext_sc_seg = sct.extract_fname(fname_sc_seg)

        im_new_template_gm = Image(self.param.output_folder+self.path_new_template+'template/MNI-Poly-AMU_GM.nii.gz')
        im_new_template_wm = Image(self.param.output_folder+self.path_new_template+'template/MNI-Poly-AMU_WM.nii.gz')

        im_new_template_gm = thr_im(im_new_template_gm, self.param.thr, self.param.thr)
        im_new_template_wm = thr_im(im_new_template_wm, self.param.thr, self.param.thr)

        self.im_template_gm = thr_im(self.im_template_gm, self.param.thr, self.param.thr)
        self.im_template_wm = thr_im(self.im_template_wm, self.param.thr, self.param.thr)

        # Create tmp folder and copy files in it
        tmp_dir = sct.tmp_create()
        sct.run('cp '+fname_manual_gmseg+' '+tmp_dir+file_manual_gmseg+ext_manual_gmseg)
        sct.run('cp '+fname_sc_seg+' '+tmp_dir+file_sc_seg+ext_sc_seg)
        os.chdir(tmp_dir)

        fname_new_template_gm = 'new_template_gm.nii.gz'
        im_new_template_gm.setFileName(fname_new_template_gm)
        im_new_template_gm.save()

        fname_new_template_wm = 'new_template_wm.nii.gz'
        im_new_template_wm.setFileName(fname_new_template_wm)
        im_new_template_wm.save()

        fname_old_template_wm = 'old_template_wm.nii.gz'
        self.im_template_wm.setFileName(fname_old_template_wm)
        self.im_template_wm.save()

        fname_old_template_gm = 'old_template_gm.nii.gz'
        self.im_template_gm.setFileName(fname_old_template_gm)
        self.im_template_gm.save()

        fname_manual_wmseg = 'target_manual_wmseg.nii.gz'
        sct.run('sct_maths -i '+file_sc_seg+ext_sc_seg+' -sub '+file_manual_gmseg+ext_manual_gmseg+' -o '+fname_manual_wmseg)

        # TODO: Hausdorff GM with old/new_template : output HD/MD new template and diff

        # Compute Hausdorff distance
        status, output_old_hd = sct.run('sct_compute_hausdorff_distance -i '+fname_old_template_gm+' -r '+file_manual_gmseg+ext_manual_gmseg+' -t 1  -v 1')
        status, output_new_hd = sct.run('sct_compute_hausdorff_distance -i '+fname_new_template_gm+' -r '+file_manual_gmseg+ext_manual_gmseg+' -t 1  -v 1')

        hd_name = 'hd_md_multilabel_reg.txt'
        hd_fic = open(hd_name, 'w')
        hd_fic.write('The "diff" columns are comparisons between regular template registration and corrected template registration according to SC internal structure\n'
                     'Diff = metric_regular_reg - metric_corrected_reg\n')
        hd_fic.write('#Slice, HD, HD diff, MD, MD diff\n')

        no_ref_slices = []

        init_hd = "Hausdorff's distance  -  First relative Hausdorff's distance median - Second relative Hausdorff's distance median(all in mm)\n"
        old_gm_hd = output_old_hd[output_old_hd.find(init_hd)+len(init_hd):].split('\n')
        new_gm_hd = output_new_hd[output_new_hd.find(init_hd)+len(init_hd):].split('\n')

        for i in range(len(old_gm_hd)-3):  # last two lines are informations
            i_old, val_old = old_gm_hd[i].split(':')
            i_new, val_new = new_gm_hd[i].split(':')
            i_old = int(i_old[-2:])
            i_new = int(i_new[-2:])

            assert i == i_old == i_new, 'ERROR: when comparing Hausdorff distances, slice numbers differs.'
            hd_old, med1_old, med2_old = val_old.split('-')
            hd_new, med1_new, med2_new = val_new.split('-')

            if float(hd_old) == 0.0:
                no_ref_slices.append(i)
                hd_fic.write(str(i)+', NO MANUAL SEGMENTATION\n')
            else:
                md_new = max(float(med1_new), float(med2_new))
                md_old = max(float(med1_old), float(med2_old))

                hd_fic.write(str(i)+', '+hd_new+', '+str(float(hd_old)-float(hd_new))+', '+str(md_new)+', '+str(md_old-md_new)+'\n')
        hd_fic.close()

        # Compute Dice coefficient
        # --- DC old template
        try:
            status_old_gm, output_old_gm = sct.run('sct_dice_coefficient '+file_manual_gmseg+ext_manual_gmseg+' '+fname_old_template_gm+' -2d-slices 2', error_exit='warning', raise_exception=True)
        except Exception:
            # put the result and the reference in the same space using a registration with ANTs with no iteration:
            corrected_manual_gmseg = file_manual_gmseg+'_in_old_template_space'+ext_manual_gmseg
            sct.run('isct_antsRegistration -d 3 -t Translation[0] -m MI['+fname_old_template_gm+','+file_manual_gmseg+ext_manual_gmseg+',1,16] -o [reg_ref_to_res,'+corrected_manual_gmseg+'] -n BSpline[3] -c 0 -f 1 -s 0')
            sct.run('sct_maths -i '+corrected_manual_gmseg+' -thr 0.1 -o '+corrected_manual_gmseg)
            sct.run('sct_maths -i '+corrected_manual_gmseg+' -bin -o '+corrected_manual_gmseg)
            status_old_gm, output_old_gm = sct.run('sct_dice_coefficient '+corrected_manual_gmseg+' '+fname_old_template_gm+'  -2d-slices 2', error_exit='warning')

        try:
            status_old_wm, output_old_wm = sct.run('sct_dice_coefficient '+fname_manual_wmseg+' '+fname_old_template_wm+' -2d-slices 2', error_exit='warning', raise_exception=True)
        except Exception:
            # put the result and the reference in the same space using a registration with ANTs with no iteration:
            path_manual_wmseg, file_manual_wmseg, ext_manual_wmseg = sct.extract_fname(fname_manual_wmseg)
            corrected_manual_wmseg = file_manual_wmseg+'_in_old_template_space'+ext_manual_wmseg
            sct.run('isct_antsRegistration -d 3 -t Translation[0] -m MI['+fname_old_template_wm+','+fname_manual_wmseg+',1,16] -o [reg_ref_to_res,'+corrected_manual_wmseg+'] -n BSpline[3] -c 0 -f 1 -s 0')
            sct.run('sct_maths -i '+corrected_manual_wmseg+' -thr 0.1 -o '+corrected_manual_wmseg)
            sct.run('sct_maths -i '+corrected_manual_wmseg+' -bin -o '+corrected_manual_wmseg)
            status_old_wm, output_old_wm = sct.run('sct_dice_coefficient '+corrected_manual_wmseg+' '+fname_old_template_wm+'  -2d-slices 2', error_exit='warning')

        # --- DC new template
        try:
            status_new_gm, output_new_gm = sct.run('sct_dice_coefficient '+file_manual_gmseg+ext_manual_gmseg+' '+fname_new_template_gm+' -2d-slices 2', error_exit='warning', raise_exception=True)
        except Exception:
            # put the result and the reference in the same space using a registration with ANTs with no iteration:
            corrected_manual_gmseg = file_manual_gmseg+'_in_new_template_space'+ext_manual_gmseg
            sct.run('isct_antsRegistration -d 3 -t Translation[0] -m MI['+fname_new_template_gm+','+file_manual_gmseg+ext_manual_gmseg+',1,16] -o [reg_ref_to_res,'+corrected_manual_gmseg+'] -n BSpline[3] -c 0 -f 1 -s 0')
            sct.run('sct_maths -i '+corrected_manual_gmseg+' -thr 0.1 -o '+corrected_manual_gmseg)
            sct.run('sct_maths -i '+corrected_manual_gmseg+' -bin -o '+corrected_manual_gmseg)
            status_new_gm, output_new_gm = sct.run('sct_dice_coefficient '+corrected_manual_gmseg+' '+fname_new_template_gm+'  -2d-slices 2', error_exit='warning')

        try:
            status_new_wm, output_new_wm = sct.run('sct_dice_coefficient '+fname_manual_wmseg+' '+fname_new_template_wm+' -2d-slices 2', error_exit='warning', raise_exception=True)
        except Exception:
            # put the result and the reference in the same space using a registration with ANTs with no iteration:
            path_manual_wmseg, file_manual_wmseg, ext_manual_wmseg = sct.extract_fname(fname_manual_wmseg)
            corrected_manual_wmseg = file_manual_wmseg+'_in_new_template_space'+ext_manual_wmseg
            sct.run('isct_antsRegistration -d 3 -t Translation[0] -m MI['+fname_new_template_wm+','+fname_manual_wmseg+',1,16] -o [reg_ref_to_res,'+corrected_manual_wmseg+'] -n BSpline[3] -c 0 -f 1 -s 0')
            sct.run('sct_maths -i '+corrected_manual_wmseg+' -thr 0.1 -o '+corrected_manual_wmseg)
            sct.run('sct_maths -i '+corrected_manual_wmseg+' -bin -o '+corrected_manual_wmseg)
            status_new_wm, output_new_wm = sct.run('sct_dice_coefficient '+corrected_manual_wmseg+' '+fname_new_template_wm+'  -2d-slices 2', error_exit='warning')

        dice_name = 'dice_multilabel_reg.txt'
        dice_fic = open(dice_name, 'w')
        dice_fic.write('The "diff" columns are comparisons between regular template registration and corrected template registration according to SC internal structure\n'
                     'Diff = metric_corrected_reg - metric_regular_reg\n')
        dice_fic.write('#Slice, WM DC, WM diff, GM DC, GM diff\n')

        init_dc = '2D Dice coefficient by slice:\n'

        old_gm_dc = output_old_gm[output_old_gm.find(init_dc)+len(init_dc):].split('\n')
        old_wm_dc = output_old_wm[output_old_wm.find(init_dc)+len(init_dc):].split('\n')
        new_gm_dc = output_new_gm[output_new_gm.find(init_dc)+len(init_dc):].split('\n')
        new_wm_dc = output_new_wm[output_new_wm.find(init_dc)+len(init_dc):].split('\n')

        for i in range(len(old_gm_dc)):
            if i not in no_ref_slices:
                i_new_gm, val_new_gm = new_gm_dc[i].split(' ')
                i_new_wm, val_new_wm = new_wm_dc[i].split(' ')
                i_old_gm, val_old_gm = old_gm_dc[i].split(' ')
                i_old_wm, val_old_wm = old_wm_dc[i].split(' ')

                assert i == int(i_new_gm) == int(i_new_wm) == int(i_old_gm) == int(i_old_wm), 'ERROR: when comparing Dice coefficients, slice numbers differs.'

                dice_fic.write(str(i)+', '+val_new_wm+', '+str(float(val_new_wm)-float(val_old_wm))+', '+val_new_gm+', '+str(float(val_new_gm)-float(val_old_gm))+'\n')
            else:
                dice_fic.write(str(i)+', NO MANUAL SEGMENTATION\n')
        dice_fic.close()
        os.chdir('..')

        sct.generate_output_file(tmp_dir+hd_name, self.param.output_folder+hd_name)
        sct.generate_output_file(tmp_dir+dice_name, self.param.output_folder+dice_name)


def thr_im(im, low_thr, high_thr):
    im.data[im.data > high_thr] = 1
    im.data[im.data <= low_thr] = 0
    return im



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

    parser.add_option(name="-p",
                      type_value="str",
                      description="Parameters for the multimodal registration between multilabel images",
                      mandatory=False,
                      example='step=1,algo=slicereg,metric=MeanSquares,step=2,algo=syn,metric=MeanSquares,iter=2:step=3,algo=bsplinesyn,metric=MeanSquares,iter=5,smooth=1')

    parser.usage.addSection('OUTPUT OTIONS')
    parser.add_option(name="-ofolder",
                      type_value="folder_creation",
                      description="Path to an output folder",
                      mandatory=False,
                      example='multilabel_registration/')

    parser.usage.addSection('VALIDATION: Use both option for validation.')
    parser.add_option(name="-manual-gm",
                      type_value='file',
                      description='Manual gray matter segmentation on the target image. ',
                      mandatory=False,
                      example='t2star_manual_gmseg.nii.gz')
    parser.add_option(name="-sc",
                      type_value='file',
                      description='Spinal cord segmentation on the target image. ',
                      mandatory=False,
                      example='t2star_seg.nii.gz')


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
    ml_param = Param()

    fname_gm = arguments['-gm']
    fname_wm = arguments['-wm']
    path_template = arguments['-t']
    fname_warp_template = arguments['-w']
    fname_target = arguments['-d']
    fname_manual_gmseg = None
    fname_sc_seg = None

    if '-p' in arguments:
        ml_param.param_reg = arguments['-p']
    if '-manual-gm' in arguments:
        fname_manual_gmseg = arguments['-manual-gm']
    if '-sc' in arguments:
        fname_sc_seg = arguments['-sc']
    if '-ofolder' in arguments:
        ml_param.output_folder = sct.slash_at_the_end(arguments['-ofolder'], 1)
    if '-qc' in arguments:
        ml_param.qc = int(arguments['-qc'])
    if '-r' in arguments:
        ml_param.remove_tmp = int(arguments['-r'])
    if '-v' in arguments:
        ml_param.verbose = int(arguments['-v'])

    if (fname_manual_gmseg is not None and fname_sc_seg is None) or (fname_manual_gmseg is None and fname_sc_seg is not None):
        sct.printv(parser.usage.generate(error='ERROR: you need to specify both arguments : -manual-gm and -sc.'))

    ml_reg = MultiLabelRegistration(fname_gm, fname_wm, path_template, fname_warp_template, fname_target, param=ml_param)
    ml_reg.register()
    if fname_manual_gmseg is not None:
        ml_reg.validation(fname_manual_gmseg, fname_sc_seg)