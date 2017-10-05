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
import os
import sys
from msct_parser import Parser
from msct_image import Image
from sct_convert import convert
import sct_utils as sct


class Param:
    def __init__(self):
        self.thr = 0.5
        self.gap = (100, 200)
        self.smooth = 0.8

        self.param_reg = 'step=1,type=seg,algo=centermassrot,metric=MeanSquares:step=2,type=im,algo=syn,metric=MeanSquares,iter=10,smooth=0,shrink=2:step=3,type=im,algo=bsplinesyn,metric=MeanSquares,iter=5,smooth=0'
        # Previous default param (less efficient): 'step=1,algo=slicereg,metric=MeanSquares:step=2,algo=bsplinesyn,metric=MeanSquares,iter=5,smooth=1'

        self.output_folder = './'
        self.verbose = 1
        self.remove_tmp = 1
        self.qc = 1


class MultiLabelRegistration:
    def __init__(self, fname_gm, fname_wm, path_template, fname_warp_template2target, param=None, fname_warp_target2template=None, apply_warp_template=0, fname_template_dest=None):
        if param is None:
            self.param = Param()
        else:
            self.param = param
        self.im_gm = Image(fname_gm)
        self.im_wm = Image(fname_wm)
        self.path_template = sct.slash_at_the_end(path_template, 1)

        # get GM and WM files from template:
        fname_template_gm, fname_template_wm = None, None
        for fname in os.listdir(self.path_template + 'template/'):
            if 'gm' in fname.lower():
                fname_template_gm = self.path_template + 'template/'+fname
            elif 'wm' in fname.lower():
                fname_template_wm = self.path_template + 'template/'+fname
        if fname_template_gm is not None and fname_template_wm is not None:
            self.im_template_gm = Image(fname_template_gm)
            self.im_template_wm = Image(fname_template_wm)
            if fname_template_gm.split('/')[-1] == 'MNI-Poly-AMU_GM.nii.gz':
                self.template = 'MNI-Poly-AMU'
            elif fname_template_gm.split('/')[-1] == 'PAM50_gm.nii.gz':
                self.template = 'PAM50'
            else:
                self.template = 'custom'

        # template file in its original space:
        self.fname_template_dest = fname_template_dest

        # Previous warping fields:
        self.fname_warp_template2target = fname_warp_template2target
        self.fname_warp_target2template = fname_warp_target2template

        # new warping fields:
        self.fname_warp_template2gm = ''
        self.fname_warp_gm2template = ''

        # temporary fix - related to issue #871
        self.apply_warp_template = apply_warp_template

    def register(self):
        # accentuate separation WM/GM
        self.im_gm = thr_im(self.im_gm, 0.01, self.param.thr)
        self.im_wm = thr_im(self.im_wm, 0.01, self.param.thr)
        self.im_template_gm = thr_im(self.im_template_gm, 0.01, self.param.thr)
        self.im_template_wm = thr_im(self.im_template_wm, 0.01, self.param.thr)

        # create multilabel images:
        # copy GM images to keep header information
        im_automatic_ml = self.im_gm.copy()
        im_template_ml = self.im_template_gm.copy()

        # create multi-label segmentation with GM*200 + WM*100 (100 and 200 encoded in self.param.gap)
        im_automatic_ml.data = self.param.gap[1] * self.im_gm.data + self.param.gap[0] * self.im_wm.data
        im_template_ml.data = self.param.gap[1] * self.im_template_gm.data + self.param.gap[0] * self.im_template_wm.data

        # set new names
        fname_automatic_ml = 'multilabel_automatic_seg.nii.gz'
        fname_template_ml = 'multilabel_template_seg.nii.gz'
        im_automatic_ml.setFileName(fname_automatic_ml)
        im_template_ml.setFileName(fname_template_ml)

        # Create temporary folder and put files in it
        tmp_dir = sct.tmp_create()

        path_gm, file_gm, ext_gm = sct.extract_fname(fname_gm)
        path_warp_template2target, file_warp_template2target, ext_warp_template2target = sct.extract_fname(self.fname_warp_template2target)

        convert(fname_gm, tmp_dir + file_gm + ext_gm)
        convert(fname_warp_template, tmp_dir + file_warp_template2target + ext_warp_template2target, squeeze_data=0)
        if self.fname_warp_target2template is not None:
            path_warp_target2template, file_warp_target2template, ext_warp_target2template = sct.extract_fname(self.fname_warp_target2template)
            convert(self.fname_warp_target2template, tmp_dir + file_warp_target2template + ext_warp_target2template, squeeze_data=0)

        os.chdir(tmp_dir)
        # save images
        im_automatic_ml.save()
        im_template_ml.save()

        # apply template2image warping field
        if self.apply_warp_template == 1:
            fname_template_ml_new = sct.add_suffix(fname_template_ml, '_r')
            sct.run('sct_apply_transfo -i ' + fname_template_ml + ' -d ' + fname_automatic_ml + ' -w ' + file_warp_template2target + ext_warp_template2target + ' -o ' + fname_template_ml_new)
            fname_template_ml = fname_template_ml_new

        nx, ny, nz, nt, px, py, pz, pt = im_automatic_ml.dim
        size_mask = int(22.5 / px)
        fname_mask = 'square_mask.nii.gz'
        sct.run('sct_create_mask -i ' + fname_automatic_ml + ' -p centerline,' + fname_automatic_ml + ' -f box -size ' + str(size_mask) + ' -o ' + fname_mask)

        fname_automatic_ml, xi, xf, yi, yf, zi, zf = crop_im(fname_automatic_ml, fname_mask)
        fname_template_ml, xi, xf, yi, yf, zi, zf = crop_im(fname_template_ml, fname_mask)

#        fname_automatic_ml_smooth = sct.add_suffix(fname_automatic_ml, '_smooth')
#        sct.run('sct_maths -i '+fname_automatic_ml+' -smooth '+str(self.param.smooth)+','+str(self.param.smooth)+',0 -o '+fname_automatic_ml_smooth)
#        fname_automatic_ml = fname_automatic_ml_smooth

        path_automatic_ml, file_automatic_ml, ext_automatic_ml = sct.extract_fname(fname_automatic_ml)
        path_template_ml, file_template_ml, ext_template_ml = sct.extract_fname(fname_template_ml)

        # Register multilabel images together
        cmd_reg = 'sct_register_multimodal -i ' + fname_template_ml + ' -d ' + fname_automatic_ml + ' -param ' + self.param.param_reg
        if 'centermass' in self.param.param_reg:
            fname_template_ml_seg = sct.add_suffix(fname_template_ml, '_bin')
            sct.run('sct_maths -i ' + fname_template_ml + ' -bin 0 -o ' + fname_template_ml_seg)

            fname_automatic_ml_seg = sct.add_suffix(fname_automatic_ml, '_bin')
            # sct.run('sct_maths -i '+fname_automatic_ml+' -thr 50 -o '+fname_automatic_ml_seg)
            sct.run('sct_maths -i ' + fname_automatic_ml + ' -bin 50 -o ' + fname_automatic_ml_seg)

            cmd_reg += ' -iseg ' + fname_template_ml_seg + ' -dseg ' + fname_automatic_ml_seg

        sct.run(cmd_reg)
        fname_warp_multilabel_template2auto = 'warp_' + file_template_ml + '2' + file_automatic_ml + '.nii.gz'
        fname_warp_multilabel_auto2template = 'warp_' + file_automatic_ml + '2' + file_template_ml + '.nii.gz'

        self.fname_warp_template2gm = sct.extract_fname(self.fname_warp_template2target)[1] + '_reg_gm' + sct.extract_fname(self.fname_warp_template2target)[2]
        # fname_warp_multilabel_template2auto = pad_im(fname_warp_multilabel_template2auto, nx, ny, nz, xi, xf, yi, yf, zi, zf)
        # fname_warp_multilabel_auto2template = pad_im(fname_warp_multilabel_auto2template, nx, ny, nz, xi, xf, yi, yf, zi, zf)

        sct.run('sct_concat_transfo -w ' + file_warp_template2target + ext_warp_template2target + ',' + fname_warp_multilabel_template2auto + ' -d ' + file_gm + ext_gm + ' -o ' + self.fname_warp_template2gm)

        if self.fname_warp_target2template is not None:
            if self.fname_template_dest is None:
                path_script = os.path.dirname(__file__)
                path_sct = os.path.dirname(path_script)
                if self.template == 'MNI-Poly-AMU':
                    self.fname_template_dest = path_sct + '/data/MNI-Poly-AMU/template/MNI-Poly-AMU_T2.nii.gz'
                elif self.template == 'PAM50':
                    self.fname_template_dest = path_sct + '/data/PAM50/template/PAM50_t2.nii.gz'

            self.fname_warp_gm2template = sct.extract_fname(self.fname_warp_target2template)[1] + '_reg_gm' + sct.extract_fname(self.fname_warp_target2template)[2]
            sct.run('sct_concat_transfo -w ' + fname_warp_multilabel_auto2template + ',' + file_warp_target2template + ext_warp_target2template + ' -d ' + self.fname_template_dest + ' -o ' + self.fname_warp_gm2template)

        os.chdir('..')

        # sct.generate_output_file(tmp_dir+fname_warp_multilabel_template2auto, self.param.output_folder+'warp_template_multilabel2automatic_seg_multilabel.nii.gz')
        # sct.generate_output_file(tmp_dir+fname_warp_multilabel_auto2template, self.param.output_folder+'warp_automatic_seg_multilabel2template_multilabel.nii.gz')

        sct.generate_output_file(tmp_dir + self.fname_warp_template2gm, self.param.output_folder + self.fname_warp_template2gm)
        if self.fname_warp_target2template is not None:
            sct.generate_output_file(tmp_dir + self.fname_warp_gm2template, self.param.output_folder + self.fname_warp_gm2template)

        if self.param.qc:
            fname_grid_warped = visualize_warp(tmp_dir + fname_warp_multilabel_template2auto, rm_tmp=self.param.remove_tmp)
            path_grid_warped, file_grid_warped, ext_grid_warped = sct.extract_fname(fname_grid_warped)
            sct.generate_output_file(fname_grid_warped, self.param.output_folder + file_grid_warped + ext_grid_warped)

        if self.param.remove_tmp:
            sct.run('rm -rf ' + tmp_dir, error_exit='warning')

    def validation(self, fname_manual_gmseg, fname_sc_seg):
        path_manual_gmseg, file_manual_gmseg, ext_manual_gmseg = sct.extract_fname(fname_manual_gmseg)
        path_sc_seg, file_sc_seg, ext_sc_seg = sct.extract_fname(fname_sc_seg)

        # Create tmp folder and copy files in it
        tmp_dir = sct.tmp_create()
        sct.run('cp ' + fname_manual_gmseg + ' ' + tmp_dir + file_manual_gmseg + ext_manual_gmseg)
        sct.run('cp ' + fname_sc_seg + ' ' + tmp_dir + file_sc_seg + ext_sc_seg)
        sct.run('cp ' + self.param.output_folder + self.fname_warp_template2gm + ' ' + tmp_dir + self.fname_warp_template2gm)
        os.chdir(tmp_dir)

        sct.run('sct_warp_template -d ' + fname_manual_gmseg + ' -w ' + self.fname_warp_template2gm + ' -qc 0 -a 0')
        if 'MNI-Poly-AMU_GM.nii.gz' in os.listdir('label/template/'):
            im_new_template_gm = Image('label/template/MNI-Poly-AMU_GM.nii.gz')
            im_new_template_wm = Image('label/template/MNI-Poly-AMU_WM.nii.gz')
        else:
            im_new_template_gm = Image('label/template/PAM50_gm.nii.gz')
            im_new_template_wm = Image('label/template/PAM50_wm.nii.gz')

        im_new_template_gm = thr_im(im_new_template_gm, self.param.thr, self.param.thr)
        im_new_template_wm = thr_im(im_new_template_wm, self.param.thr, self.param.thr)

        self.im_template_gm = thr_im(self.im_template_gm, self.param.thr, self.param.thr)
        self.im_template_wm = thr_im(self.im_template_wm, self.param.thr, self.param.thr)

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
        sct.run('sct_maths -i ' + file_sc_seg + ext_sc_seg + ' -sub ' + file_manual_gmseg + ext_manual_gmseg + ' -o ' + fname_manual_wmseg)

        # Compute Hausdorff distance
        status, output_old_hd = sct.run('sct_compute_hausdorff_distance -i ' + fname_old_template_gm + ' -r ' + file_manual_gmseg + ext_manual_gmseg + ' -t 1  -v 1')
        status, output_new_hd = sct.run('sct_compute_hausdorff_distance -i ' + fname_new_template_gm + ' -r ' + file_manual_gmseg + ext_manual_gmseg + ' -t 1  -v 1')

        hd_name = 'hd_md_multilabel_reg.txt'
        hd_fic = open(hd_name, 'w')
        hd_fic.write('The "diff" columns are comparisons between regular template registration and corrected template registration according to SC internal structure\n'
                     'Diff = metric_regular_reg - metric_corrected_reg\n')
        hd_fic.write('#Slice, HD, HD diff, MD, MD diff\n')

        no_ref_slices = []

        init_hd = "Hausdorff's distance  -  First relative Hausdorff's distance median - Second relative Hausdorff's distance median(all in mm)\n"
        old_gm_hd = output_old_hd[output_old_hd.find(init_hd) + len(init_hd):].split('\n')
        new_gm_hd = output_new_hd[output_new_hd.find(init_hd) + len(init_hd):].split('\n')

        for i in range(len(old_gm_hd) - 3):  # last two lines are informations
            i_old, val_old = old_gm_hd[i].split(':')
            i_new, val_new = new_gm_hd[i].split(':')
            i_old = int(i_old[-2:])
            i_new = int(i_new[-2:])

            assert i == i_old == i_new, 'ERROR: when comparing Hausdorff distances, slice numbers differs.'
            hd_old, med1_old, med2_old = val_old.split('-')
            hd_new, med1_new, med2_new = val_new.split('-')

            if float(hd_old) == 0.0:
                no_ref_slices.append(i)
                hd_fic.write(str(i) + ', NO MANUAL SEGMENTATION\n')
            else:
                md_new = max(float(med1_new), float(med2_new))
                md_old = max(float(med1_old), float(med2_old))

                hd_fic.write(str(i) + ', ' + hd_new + ', ' + str(float(hd_old) - float(hd_new)) + ', ' + str(md_new) + ', ' + str(md_old - md_new) + '\n')
        hd_fic.close()

        # Compute Dice coefficient
        # --- DC old template
        try:
            status_old_gm, output_old_gm = sct.run('sct_dice_coefficient -i ' + file_manual_gmseg + ext_manual_gmseg + ' -d ' + fname_old_template_gm + ' -2d-slices 2', error_exit='warning', raise_exception=True)
        except Exception:
            # put the result and the reference in the same space using a registration with ANTs with no iteration:
            corrected_manual_gmseg = file_manual_gmseg + '_in_old_template_space' + ext_manual_gmseg
            sct.run('isct_antsRegistration -d 3 -t Translation[0] -m MI[' + fname_old_template_gm + ',' + file_manual_gmseg + ext_manual_gmseg + ',1,16] -o [reg_ref_to_res,' + corrected_manual_gmseg + '] -n BSpline[3] -c 0 -f 1 -s 0')
            # sct.run('sct_maths -i '+corrected_manual_gmseg+' -thr 0.1 -o '+corrected_manual_gmseg)
            sct.run('sct_maths -i ' + corrected_manual_gmseg + ' -bin 0.1 -o ' + corrected_manual_gmseg)
            status_old_gm, output_old_gm = sct.run('sct_dice_coefficient -i ' + corrected_manual_gmseg + ' -d ' + fname_old_template_gm + '  -2d-slices 2', error_exit='warning')

        try:
            status_old_wm, output_old_wm = sct.run('sct_dice_coefficient -i ' + fname_manual_wmseg + ' -d ' + fname_old_template_wm + ' -2d-slices 2', error_exit='warning', raise_exception=True)
        except Exception:
            # put the result and the reference in the same space using a registration with ANTs with no iteration:
            path_manual_wmseg, file_manual_wmseg, ext_manual_wmseg = sct.extract_fname(fname_manual_wmseg)
            corrected_manual_wmseg = file_manual_wmseg + '_in_old_template_space' + ext_manual_wmseg
            sct.run('isct_antsRegistration -d 3 -t Translation[0] -m MI[' + fname_old_template_wm + ',' + fname_manual_wmseg + ',1,16] -o [reg_ref_to_res,' + corrected_manual_wmseg + '] -n BSpline[3] -c 0 -f 1 -s 0')
            # sct.run('sct_maths -i '+corrected_manual_wmseg+' -thr 0.1 -o '+corrected_manual_wmseg)
            sct.run('sct_maths -i ' + corrected_manual_wmseg + ' -bin 0.1 -o ' + corrected_manual_wmseg)
            status_old_wm, output_old_wm = sct.run('sct_dice_coefficient -i ' + corrected_manual_wmseg + ' -d ' + fname_old_template_wm + '  -2d-slices 2', error_exit='warning')

        # --- DC new template
        try:
            status_new_gm, output_new_gm = sct.run('sct_dice_coefficient -i ' + file_manual_gmseg + ext_manual_gmseg + ' -d ' + fname_new_template_gm + ' -2d-slices 2', error_exit='warning', raise_exception=True)
        except Exception:
            # put the result and the reference in the same space using a registration with ANTs with no iteration:
            corrected_manual_gmseg = file_manual_gmseg + '_in_new_template_space' + ext_manual_gmseg
            sct.run('isct_antsRegistration -d 3 -t Translation[0] -m MI[' + fname_new_template_gm + ',' + file_manual_gmseg + ext_manual_gmseg + ',1,16] -o [reg_ref_to_res,' + corrected_manual_gmseg + '] -n BSpline[3] -c 0 -f 1 -s 0')
            # sct.run('sct_maths -i '+corrected_manual_gmseg+' -thr 0.1 -o '+corrected_manual_gmseg)
            sct.run('sct_maths -i ' + corrected_manual_gmseg + ' -bin 0.1 -o ' + corrected_manual_gmseg)
            status_new_gm, output_new_gm = sct.run('sct_dice_coefficient -i ' + corrected_manual_gmseg + ' -d ' + fname_new_template_gm + '  -2d-slices 2', error_exit='warning')

        try:
            status_new_wm, output_new_wm = sct.run('sct_dice_coefficient -i ' + fname_manual_wmseg + ' -d ' + fname_new_template_wm + ' -2d-slices 2', error_exit='warning', raise_exception=True)
        except Exception:
            # put the result and the reference in the same space using a registration with ANTs with no iteration:
            path_manual_wmseg, file_manual_wmseg, ext_manual_wmseg = sct.extract_fname(fname_manual_wmseg)
            corrected_manual_wmseg = file_manual_wmseg + '_in_new_template_space' + ext_manual_wmseg
            sct.run('isct_antsRegistration -d 3 -t Translation[0] -m MI[' + fname_new_template_wm + ',' + fname_manual_wmseg + ',1,16] -o [reg_ref_to_res,' + corrected_manual_wmseg + '] -n BSpline[3] -c 0 -f 1 -s 0')
            # sct.run('sct_maths -i '+corrected_manual_wmseg+' -thr 0.1 -o '+corrected_manual_wmseg)
            sct.run('sct_maths -i ' + corrected_manual_wmseg + ' -bin 0.1 -o ' + corrected_manual_wmseg)
            status_new_wm, output_new_wm = sct.run('sct_dice_coefficient -i ' + corrected_manual_wmseg + ' -d ' + fname_new_template_wm + '  -2d-slices 2', error_exit='warning')

        dice_name = 'dice_multilabel_reg.txt'
        dice_fic = open(dice_name, 'w')
        dice_fic.write('The "diff" columns are comparisons between regular template registration and corrected template registration according to SC internal structure\n'
                     'Diff = metric_corrected_reg - metric_regular_reg\n')
        dice_fic.write('#Slice, WM DC, WM diff, GM DC, GM diff\n')

        init_dc = '2D Dice coefficient by slice:\n'

        old_gm_dc = output_old_gm[output_old_gm.find(init_dc) + len(init_dc):].split('\n')
        old_wm_dc = output_old_wm[output_old_wm.find(init_dc) + len(init_dc):].split('\n')
        new_gm_dc = output_new_gm[output_new_gm.find(init_dc) + len(init_dc):].split('\n')
        new_wm_dc = output_new_wm[output_new_wm.find(init_dc) + len(init_dc):].split('\n')

        for i in range(len(old_gm_dc)):
            if i not in no_ref_slices:
                i_new_gm, val_new_gm = new_gm_dc[i].split(' ')
                i_new_wm, val_new_wm = new_wm_dc[i].split(' ')
                i_old_gm, val_old_gm = old_gm_dc[i].split(' ')
                i_old_wm, val_old_wm = old_wm_dc[i].split(' ')

                assert i == int(i_new_gm) == int(i_new_wm) == int(i_old_gm) == int(i_old_wm), 'ERROR: when comparing Dice coefficients, slice numbers differs.'

                dice_fic.write(str(i) + ', ' + val_new_wm + ', ' + str(float(val_new_wm) - float(val_old_wm)) + ', ' + val_new_gm + ', ' + str(float(val_new_gm) - float(val_old_gm)) + '\n')
            else:
                dice_fic.write(str(i) + ', NO MANUAL SEGMENTATION\n')
        dice_fic.close()
        os.chdir('..')

        sct.generate_output_file(tmp_dir + hd_name, self.param.output_folder + hd_name)
        sct.generate_output_file(tmp_dir + dice_name, self.param.output_folder + dice_name)

        if self.param.remove_tmp:
            sct.run('rm -rf ' + tmp_dir, error_exit='warning')


def thr_im(im, low_thr, high_thr):
    im.data[im.data > high_thr] = 1
    im.data[im.data <= low_thr] = 0
    return im


def crop_im(fname_im, fname_mask):
    fname_im_crop = sct.add_suffix(fname_im, '_crop')
    status, output_crop = sct.run('sct_crop_image -i ' + fname_im + ' -m ' + fname_mask + ' -o ' + fname_im_crop)
    output_list = output_crop.split('\n')
    xi, xf, yi, yf, zi, zf = 0, 0, 0, 0, 0, 0
    for line in output_list:
        if 'Dimension 0' in line:
            dim, i, xi, xf = line.split(' ')
        if 'Dimension 1' in line:
            dim, i, yi, yf = line.split(' ')
        if 'Dimension 2' in line:
            dim, i, zi, zf = line.split(' ')
    return fname_im_crop, int(xi), int(xf), int(yi), int(yf), int(zi), int(zf)


def pad_im(fname_im, nx_full, ny_full, nz_full,  xi, xf, yi, yf, zi, zf):
    fname_im_pad = sct.add_suffix(fname_im, '_pad')
    pad_xi = str(xi)
    pad_xf = str(nx_full - (xf + 1))
    pad_yi = str(yi)
    pad_yf = str(ny_full - (yf + 1))
    pad_zi = str(zi)
    pad_zf = str(nz_full - (zf + 1))
    pad = ','.join([pad_xi, pad_xf, pad_yi, pad_yf, pad_zi, pad_zf])
    if len(Image(fname_im).data.shape) == 5:
        status, output = sct.run('sct_image -i ' + fname_im + ' -mcs')
        s = 'Created file(s):\n-->'
        output_fnames = output[output.find(s) + len(s):].split('\n')[0].split("'")
        fname_comp_list = [output_fnames[i] for i in range(1, len(output_fnames), 2)]
        fname_comp_pad_list = []
        for fname_comp in fname_comp_list:
            fname_comp_pad = sct.add_suffix(fname_comp, '_pad')
            sct.run('sct_image -i ' + fname_comp + ' -pad-asym ' + pad + ' -o ' + fname_comp_pad)
            fname_comp_pad_list.append(fname_comp_pad)
        components = ','.join(fname_comp_pad_list)
        sct.run('sct_image -i ' + components + ' -omc -o ' + fname_im_pad)
        sct.check_file_exist(fname_im_pad, verbose=1)
    else:
        sct.run('sct_image -i ' + fname_im + ' -pad-asym ' + pad + ' -o ' + fname_im_pad)
    return fname_im_pad


def visualize_warp(fname_warp, fname_grid=None, step=3, rm_tmp=True):
    if fname_grid is None:
        from numpy import zeros
        tmp_dir = sct.tmp_create()
        im_warp = Image(fname_warp)
        os.chdir(tmp_dir)

        assert len(im_warp.data.shape) == 5, 'ERROR: Warping field does bot have 5 dimensions...'
        nx, ny, nz, nt, ndimwarp = im_warp.data.shape

        # nx, ny, nz, nt, px, py, pz, pt = im_warp.dim
        # This does not work because dimensions of a warping field are not correctly read : it would be 1,1,1,1,1,1,1,1

        sq = zeros((step, step))
        sq[step - 1] = 1
        sq[:, step - 1] = 1
        dat = zeros((nx, ny, nz))
        for i in range(0, dat.shape[0], step):
            for j in range(0, dat.shape[1], step):
                for k in range(dat.shape[2]):
                    if dat[i:i + step, j:j + step, k].shape == (step, step):
                        dat[i:i + step, j:j + step, k] = sq
        fname_grid = 'grid_' + str(step) + '.nii.gz'
        im_grid = Image(param=dat)
        grid_hdr = im_warp.hdr
        im_grid.hdr = grid_hdr
        im_grid.setFileName(fname_grid)
        im_grid.save()
        fname_grid_resample = sct.add_suffix(fname_grid, '_resample')
        sct.run('sct_resample -i ' + fname_grid + ' -f 3x3x1 -x nn -o ' + fname_grid_resample)
        fname_grid = tmp_dir + fname_grid_resample
        os.chdir('..')
    path_warp, file_warp, ext_warp = sct.extract_fname(fname_warp)
    grid_warped = path_warp + 'grid_warped_gm' + ext_warp
    sct.run('sct_apply_transfo -i ' + fname_grid + ' -d ' + fname_grid + ' -w ' + fname_warp + ' -o ' + grid_warped)
    if rm_tmp:
        sct.run('rm -rf ' + tmp_dir, error_exit='warning')
    return grid_warped


########################################################################################################################
def get_parser():
    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Registration function to improve the template registration by accounting for the gray and white matter shape using a multi-label approach. Output is a warping field from the template to the target image accounting for the gray matter shape. If -winv is used, output also includes the inverse warping field (from the target image to the template).')
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
                      description="Path to template (registered on target image) to select the GM and WM files from the template. Template files must be in path/template/ and include \"gm\" (resp. \"wm\") in the file name.",
                      mandatory=False,
                      default_value='label/')
    parser.add_option(name="-w",
                      type_value="file",
                      description="Warping field: [template -> anat]",
                      mandatory=True,
                      example='warp_template2t2star.nii.gz')
    parser.add_option(name="-winv",
                      type_value="file",
                      description="Input the inverse warping field [anat -> template] if you need to output the corrected inverse warping field.",
                      mandatory=False,
                      example='warp_t2star2template.nii.gz')


    parser.add_option(name="-param",
                      type_value="str",
                      description="Parameters for the multimodal registration between multilabel images",
                      mandatory=False,
                      default_value=Param().param_reg,
                      example='step=1,algo=slicereg,metric=MeanSquares,step=2,algo=syn,metric=MeanSquares,iter=2:step=3,algo=bsplinesyn,metric=MeanSquares,iter=5,smooth=1')
    parser.add_option(name="-template-original",
                      type_value="file",
                      description="File of the template in it's original space. Only needed if you are using a custom template and flag -winv. It will be used as a destination to concatenate the inverse warping field.",
                      mandatory=False,
                      example='.../custom_template/custom_template_t2.nii.gz')

    parser.usage.addSection('\nOUTPUT OTIONS')
    parser.add_option(name="-ofolder",
                      type_value="folder_creation",
                      description="Path to an output folder",
                      mandatory=False,
                      example='multilabel_registration/')

    parser.usage.addSection('\nVALIDATION: Use both flags for validation.')
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

    parser.usage.addSection('\nMISC')
    # parser.add_option(name="-apply-warp",
    #                   type_value='multiple_choice',
    #                   description="Application of the warping field (option '-w') on the template (option '-t'). 0: do not apply it. 1: apply it.",
    #                   mandatory=False,
    #                   example=['0', '1'],
    #                   default_value='0')
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
    sct.start_stream_logger()
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])
    ml_param = Param()

    fname_gm = arguments['-gm']
    fname_wm = arguments['-wm']
    path_template = arguments['-t']
    if not sct.check_folder_exist(path_template):
        sct.printv(parser.usage.generate(error='ERROR: label/ folder does not exist. Please specify the path to the template using flag -t'))
    fname_warp_template = arguments['-w']

    fname_warp_target2template = None
    fname_manual_gmseg = None
    fname_sc_seg = None
    fname_template_dest = None

    if '-param' in arguments:
        ml_param.param_reg = arguments['-param']
    if '-manual-gm' in arguments:
        fname_manual_gmseg = arguments['-manual-gm']
    if '-sc' in arguments:
        fname_sc_seg = arguments['-sc']
    if '-winv' in arguments:
        fname_warp_target2template = arguments['-winv']
    if '-template-original' in arguments:
        fname_template_dest = arguments['-template-original']
    if '-ofolder' in arguments:
        ml_param.output_folder = sct.slash_at_the_end(arguments['-ofolder'], 1)
    if '-qc' in arguments:
        ml_param.qc = int(arguments['-qc'])
    if '-r' in arguments:
        ml_param.remove_tmp = int(arguments['-r'])
    if '-v' in arguments:
        ml_param.verbose = int(arguments['-v'])

    apply_warp = 0
    if '-apply-warp' in arguments:
        apply_warp = int(arguments['-apply-warp'])

    if (fname_manual_gmseg is not None and fname_sc_seg is None) or (fname_manual_gmseg is None and fname_sc_seg is not None):
        sct.printv(parser.usage.generate(error='ERROR: you need to specify both arguments : -manual-gm and -sc.'))

    ml_reg = MultiLabelRegistration(fname_gm, fname_wm, path_template, fname_warp_template, param=ml_param, fname_warp_target2template=fname_warp_target2template, apply_warp_template=apply_warp, fname_template_dest=fname_template_dest)
    ml_reg.register()
    if fname_manual_gmseg is not None:
        ml_reg.validation(fname_manual_gmseg, fname_sc_seg)
