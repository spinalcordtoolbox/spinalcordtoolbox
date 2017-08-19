#!/usr/bin/env python

# Analyze lesions
#
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Charley
# Modified: 2017-08-19
#
# About the license: see the file LICENSE.TXT

import os
import shutil
import sys
import numpy as np
from math import sqrt
from skimage.measure import label
import pandas as pd
import pickle

from sct_maths import binarise
from msct_image import Image
from msct_parser import Parser
from sct_image import set_orientation, get_orientation
from sct_utils import (add_suffix, extract_fname, printv, run, printv,
                       slash_at_the_end, Timer, tmp_create, get_absolute_path)
from sct_straighten_spinalcord import smooth_centerline
from msct_types import Centerline

def get_parser():
    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Compute statistics on lesions of the input binary file (1 for lesion, 0 for background). The function assigns an ID value to each lesion (1, 2, 3, etc.) and outputs morphometric measures for each lesion:'
                                    '\n- volume [mm^3]'
                                    '\n- length [mm]: length along the Superior-Inferior axis'
                                    '\n- max_equivalent_diameter [mm]: maximum diameter of the lesion, when approximating the lesion as a circle in the axial cross-sectional plane orthogonal to the spinal cord'
                                    '\n\nIf an image (e.g. T2w or T1w image, texture image) is provided, it computes the mean and standard deviation values of this image within each lesion.'
                                    '\n\nIf a registered template is provided, it computes:'
                                    '\n- the distribution of each lesion depending on each vertebral level and on each region of the template (eg GM, WM, WM tracts).'
                                    '\n- the proportion of ROI (eg vertebral level, GM, WM) occupied by lesion.'
                                    '\nN.B. If the proportion of lesion in each region (e.g., WM and GM) does not sum up to 100%, it means that the registered template does not fully cover the lesion, in that case you might want to check the registration results.')
    parser.add_option(name="-m",
                      type_value="file",
                      description="Lesion mask to analyze",
                      mandatory=True,
                      example='t2_lesion.nii.gz')
    parser.add_option(name="-s",
                      type_value="file",
                      description="Spinal cord centerline or segmentation file, which will be used to correct morphometric measures with cord angle with respect to slice.",
                      mandatory=False,
                      example='t2_seg.nii.gz')
    parser.add_option(name="-i",
                      type_value="file",
                      description="Image from which to extract average values within lesions (e.g. T2w or T1w image, texture image).",
                      mandatory=False,
                      example='t2.nii.gz')
    parser.add_option(name="-f",
                      type_value="str",
                      description="Path to folder containing the atlas/template registered to the anatomical image.",
                      mandatory=False,
                      example="./label")
    parser.add_option(name="-ofolder",
                      type_value="folder_creation",
                      description="Output folder",
                      mandatory=False,
                      example='./')
    parser.add_option(name="-r",
                      type_value="multiple_choice",
                      description="Remove temporary files.",
                      mandatory=False,
                      default_value='1',
                      example=['0', '1'])
    parser.add_option(name="-v",
                      type_value='multiple_choice',
                      description="Verbose: 0 = nothing, 1 = classic, 2 = expended",
                      mandatory=False,
                      example=['0', '1', '2'],
                      default_value='1')

    return parser

def pve_weighted_avg(im_mask_data, im_atlas_data):
	return im_mask_data * im_atlas_data

  
def relative_ROIvol_in_mask(im_mask_data, im_atlas_roi_data, p_lst, im_template_vert_data=None, vert_level=None):
	#
	#   Goal:
	#         This function computes the percentage of ROI occupied by binary mask
	#         --> ie volume of the intersection between {im_mask and im_roi} divided by the volume of roi
	#         If im_template_vert and vert are specified, the ROI is restricted to the given vert_level
	#         The PVE is handled by the method 'weighted_average'
	#
	#   Inputs:
	#           - im_mask_data - type=NumPyArray - binary mask (eg lesions)
	#           - im_atlas_roi_data - type=NumPyArray - ROI in the same space as im_mask
	#			- p_lst - type=list of float
	#           - im_template_vert_data - type=NumPyArray - vertebral template in the same space as im_mask
	#           - vert_level - type=int - vertebral level ID to restrict the ROI
	#

	if im_template_vert_data is not None and vert_level is not None:
		im_atlas_roi_data[np.where(im_template_vert_data != vert_level)] = 0.0
		im_mask_data[np.where(im_template_vert_data != vert_level)] = 0.0

	im_mask_roi_data_wa = pve_weighted_avg(im_mask_data, im_atlas_roi_data)

	vol_tot_roi = np.sum(im_atlas_roi_data) * p_lst[0] * p_lst[1] * p_lst[2]
	# vol_mask_tot = np.sum(im_mask_data) * p_lst[0] * p_lst[1] * p_lst[2]
	vol_mask_roi_wa = np.sum(im_mask_roi_data_wa) * p_lst[0] * p_lst[1] * p_lst[2]

	# printv('\tTotal volume of ROI = '+str(round(vol_tot_roi,2))+' mm^3', verbose=0, type='info')
	# printv('\tVolume of im_mask in ROI = '+str(round(vol_mask_roi_wa,2))+' mm^3', verbose=0, type='info')
	# printv('\tPercentage of ROI occupied by im_mask = '+str(round(vol_mask_roi_wa/vol_tot_roi,2)*100.0)+'%', verbose=0, type='info')

	return vol_mask_roi_wa, vol_tot_roi


class AnalyzeLeion:
	def __init__(self, fname_mask, fname_sc, fname_ref, path_template, path_ofolder, verbose):

		self.fname_mask = fname_mask
		print np.unique(Image(fname_mask).data)
		if not set(np.unique(Image(fname_mask).data)) == set([0.0, 1.0]):
			if set(np.unique(Image(fname_mask).data)) == set([0.0]):
				printv('WARNING: Empty masked image', self.verbose, 'warning')
			else:
  				printv("ERROR input file %s is not binary file with 0 and 1 values" % fname_mask, 1, 'error')

		self.fname_sc = fname_sc
		self.fname_ref = fname_ref
		self.path_template = path_template
		self.path_ofolder = path_ofolder
		self.verbose = verbose

		# create tmp directory
		self.tmp_dir = tmp_create(verbose=verbose)  # path to tmp directory

		# lesion file where each lesion has a different value
		self.fname_label = add_suffix(self.fname_mask, '_label')

		# initialization of measure sheet
		measure_lst = ['label', 'volume [mm3]', 'length [mm]', 'max_equivalent_diameter [mm]']
		if self.fname_ref is not None:
			for measure in ['mean', 'std']:
				measure_lst.append(measure+'_'+extract_fname(self.fname_ref)[1])
		measure_dct = {}
		for column in measure_lst:
			measure_dct[column] = None
		self.measure_pd = pd.DataFrame(data=measure_dct,index=range(0),columns=measure_lst)

		# orientation of the input image
		self.orientation = None

		# angle correction
		self.angles = np.zeros(Image(self.fname_mask).dim[2])

		# volume object
		self.volumes = None

		# initialization of proportion measures, related to registrated atlas
		if self.path_template is not None:
			self.path_atlas = self.path_template+'atlas/'
			self.path_levels = self.path_template+'template/PAM50_levels.nii.gz'
		else:
			self.path_atlas, self.path_levels = None, None
		self.vert_lst = None
		self.atlas_roi_lst = None
		self.distrib_matrix_dct = {}

		# output names
		self.pickle_name = extract_fname(self.fname_mask)[1]+'_analyzis.pkl'
		self.excel_name = extract_fname(self.fname_mask)[1]+'_analyzis.xlsx'

	def analyze(self):
		self.ifolder2tmp()

		# Orient input image(s) to RPI
		self.orient2rpi()

		# Label connected regions of the masked image
		self.label_lesion()

		# Compute angle for CSA correction
		self.angle_correction()

		# # Compute lesion volume, equivalent diameter, (S-I) length, max axial nominal diameter
		# # if registered template provided: across vertebral level, GM, WM, within WM/GM tracts...
		# self.measure()

		# # Compute mean, median, min, max value in each labeled lesion
		# if self.fname_ref is not None:
		#   self.measure_within_im()

		# # reorient data if needed
		# self.reorient()

		# # print averaged results
		# self.show_total_results()

		# # save results in excel and pickle files
		# self.pack_measures()

		# # save results to ofolder
		# self.tmp2ofolder()


	def tmp2ofolder(self):

		os.chdir('..') # go back to original directory

		printv('\nSave results files...', self.verbose, 'normal')
		printv('\n... measures saved in the files:', self.verbose, 'normal')
		printv('\n  - '+self.path_ofolder+self.excel_name, self.verbose, 'normal')
		if self.pickle_name is not None:
			printv('\n  - '+self.path_ofolder+self.pickle_name, self.verbose, 'normal')

		for file in [self.fname_label, self.excel_name, self.pickle_name]:
			if file is not None:
				shutil.copy(self.tmp_dir+file, self.path_ofolder+file)

	def pack_measures(self):

		writer = pd.ExcelWriter(self.excel_name)
		self.data_pd.to_excel(writer,sheet_name='measures', index=False, engine='xlsxwriter')
		if self.path_template is not None:
			for sheet_name in self.dct_matrix:
				if '#' in sheet_name:
					df = self.dct_matrix[sheet_name].copy()
					df = df.append(df.sum(numeric_only=True, axis=0), ignore_index=True)
					df['total'] = df.sum(numeric_only=True, axis=1)
					df.iloc[-1, df.columns.get_loc('vert')] = 'total'
					df.to_excel(writer,sheet_name=sheet_name, index=False, engine='xlsxwriter')
				else:
					self.dct_matrix[sheet_name].to_excel(writer,sheet_name=sheet_name, index=False, engine='xlsxwriter')

			self.dct_matrix['measures'] = self.data_pd
			with open(self.pickle_name, 'wb') as handle:
				pickle.dump(self.dct_matrix, handle)

		writer.save()



	def show_total_results(self):

		printv('\n\nAveraged measures...', self.verbose, 'normal')
		printv('  Volume = '+str(round(np.mean(self.data_pd['volume [mm3]']),2))+'+/-'+str(round(np.std(self.data_pd['volume [mm3]']),2))+' mm^3', self.verbose, type='info')
		printv('  (S-I) Length = '+str(round(np.mean(self.data_pd['length [mm]']),2))+'+/-'+str(round(np.std(self.data_pd['length [mm]']),2))+' mm', self.verbose, type='info')
		printv('  Equivalent Diameter = '+str(round(np.mean(self.data_pd['max_equivalent_diameter [mm]']),2))+'+/-'+str(round(np.std(self.data_pd['max_equivalent_diameter [mm]']),2))+' mm', self.verbose, type='info')

		if 'GM [%]' in self.data_pd:
			printv('  Proportion of lesions in WM / GM = '+str(round(np.mean(self.data_pd['WM [%]']),2))+'% / '+str(round(np.mean(self.data_pd['GM [%]']),2))+'%', self.verbose, type='info')

		printv('\nTotal volume = '+str(round(np.sum(self.data_pd['volume [mm3]']),2))+' mm^3', self.verbose, 'info')
		printv('Lesion count = '+str(len(self.data_pd['volume [mm3]'].values)), self.verbose, 'info')

	def reorient(self):
		if not self.orientation == 'RPI':
			printv('\nOrient output image to initial orientation...', self.verbose, 'normal')
			self._orient(self.fname_label, self.orientation)

	def measure_within_im(self):
		printv('\nCompute reference image features...', self.verbose, 'normal')
		im_label_data, im_ref_data = Image(self.fname_label).data, Image(self.fname_ref).data

		for lesion_label in [l for l in np.unique(im_label_data) if l]:
			im_label_data_cur = im_label_data == lesion_label
			im_label_data_cur[np.where(im_ref_data==0)] = 0 # if the ref object is eroded compared to the labeled object
			mean_cur, std_cur  = np.mean(im_ref_data[np.where(im_label_data_cur)]), np.std(im_ref_data[np.where(im_label_data_cur)])

			label_idx = self.data_pd[self.data_pd.label==lesion_label].index
			self.data_pd.loc[label_idx, 'mean_'+extract_fname(self.fname_ref)[1]] = mean_cur
			self.data_pd.loc[label_idx, 'std_'+extract_fname(self.fname_ref)[1]] = std_cur
			printv('Mean+/-std of lesion #'+str(lesion_label)+' in '+extract_fname(self.fname_ref)[1]+' file: '+str(round(mean_cur,2))+'+/-'+str(round(std_cur,2)), self.verbose, type='info')

	def _measure_volume(self, im_data, p_lst, idx):

		for zz in range(im_data.shape[2]):
			self.volumes[zz,idx-1] = np.sum(im_data[:,:,zz]) * p_lst[0] * p_lst[1] * p_lst[2]

		vol_tot_cur = np.sum(self.volumes[:,idx-1])
		self.data_pd.loc[idx, 'volume [mm3]'] = vol_tot_cur
		printv('  Volume : '+str(round(vol_tot_cur,2))+' mm^3', self.verbose, type='info')

	def _measure_length(self, im_data, p_lst, idx):

		length_cur = np.sum([np.cos(self.angles[zz]) * p_lst[2] for zz in list(np.unique(np.where(im_data)[2]))])

		self.data_pd.loc[idx, 'length [mm]'] = length_cur
		printv('  (S-I) length : '+str(round(length_cur,2))+' mm', self.verbose, type='info')

	def _measure_diameter(self, im_data, p_lst, idx):

		area_lst = []
		for zz in range(im_data.shape[2]):
			area_lst.append(np.sum(im_data[:,:,zz]) * np.cos(self.angles[zz]) * p_lst[0] * p_lst[1])
			diameter_cur = sqrt(max(area_lst)/(4*pi))

		self.data_pd.loc[idx, 'max_equivalent_diameter [mm]'] = diameter_cur
		printv('  Max. equivalent diameter : '+str(round(diameter_cur,2))+' mm', self.verbose, type='info')

	def _regroup_per_tracts(self, dct, lim_lst):
		res_mask, res_tot = [dct[t][0] for t in dct if t >=lim_lst[0] and t<=lim_lst[1]], [dct[t][1] for t in dct if t >=lim_lst[0] and t<=lim_lst[1]]
		return np.sum(res_mask)*100.0/np.sum(res_tot)

	def measure(self):
		im_lesion = Image(self.fname_label)
		im_lesion_data = im_lesion.data
		p_lst = im_lesion.dim[3:6]

		label_lst = [l for l in np.unique(im_lesion_data) if l]

		if self.path_template is not None:
			if os.path.isfile(self.path_levels):
				img_vert = Image(self.path_levels)
				im_vert_data = img_vert.data
				self.vert_lst = [v for v in np.unique(im_vert_data) if v]
				
			else:
				im_vert_data = None
				printv('WARNING: the file '+self.path_levels+' does not exist. Please make sure the template was correctly registered and warped (sct_register_to_template or sct_register_multimodal and sct_warp_template)', type='warning')
			
			# To open atlas images only once
			dct_atlas_data = {}
			for fname_atlas_roi in self.atlas_roi_lst:
				if fname_atlas_roi.endswith('.nii.gz'):
					tract_id = int(fname_atlas_roi.split('_')[-1].split('.nii.gz')[0])
					if tract_id < 36:
						img_cur = Image(fname_atlas_roi)
						img_cur_copy = img_cur.copy()
						dct_atlas_data[tract_id] = img_cur_copy.data

		self.volumes = np.zeros((im_lesion.dim[2],len(label_lst)))

		dct_lesion_tracts = {}
		for lesion_label in label_lst:
			im_lesion_data_cur = np.copy(im_lesion_data == lesion_label)
			printv('\nMeasures on lesion #'+str(lesion_label)+'...', self.verbose, 'normal')

			label_idx = self.data_pd[self.data_pd.label==lesion_label].index
			self._measure_volume(im_lesion_data_cur, p_lst, label_idx)
			self._measure_length(im_lesion_data_cur, p_lst, label_idx)
			self._measure_diameter(im_lesion_data_cur, p_lst, label_idx)

			if self.path_template is not None:
				sheet_name = 'lesion#'+str(lesion_label)+'_distribution'
				self.dct_matrix[sheet_name] = pd.DataFrame.from_dict({'vert': [str(v) for v in self.vert_lst]})

				for tract_id in dct_atlas_data:
					self.dct_matrix[sheet_name]['PAM50_'+str(tract_id).zfill(2)] = [0] * len(self.vert_lst)
				
				vol_mask_tot = 0.0
				for vert in self.vert_lst:
					im_vert_cur = np.copy(im_vert_data)
					im_vert_cur[np.where(im_vert_cur!=vert)] = 0.0
					if np.count_nonzero(im_vert_cur*np.copy(im_lesion_data_cur)):
						dct_tmp = {}
						idx = self.dct_matrix[sheet_name][self.dct_matrix[sheet_name].vert==str(vert)].index
						for tract_id in dct_atlas_data:
							res_lst = relative_ROIvol_in_mask(np.copy(im_lesion_data_cur), np.copy(dct_atlas_data[tract_id]), p_lst, np.copy(im_vert_data), vert)
							self.dct_matrix[sheet_name].loc[idx, 'PAM50_'+str(tract_id).zfill(2)] = res_lst[0]
							vol_mask_tot += res_lst[0]
				
				for vert in self.vert_lst:
					idx = self.dct_matrix[sheet_name][self.dct_matrix[sheet_name].vert==str(vert)].index
					for tract_id in dct_atlas_data:
						val = self.dct_matrix[sheet_name].loc[idx, 'PAM50_'+str(tract_id).zfill(2)].values[0]
						self.dct_matrix[sheet_name].loc[idx, 'PAM50_'+str(tract_id).zfill(2)] = val*100.0/vol_mask_tot				

		if self.path_template is not None:
			
			im_lesion_data_cur = np.copy(im_lesion_data > 0)

			sheet_name = 'ROI_occupied_by_lesion'
			self.dct_matrix[sheet_name] = pd.DataFrame.from_dict({'vert': [str(v) for v in self.vert_lst]})

			for tract_id in dct_atlas_data:
				self.dct_matrix[sheet_name]['PAM50_'+str(tract_id).zfill(2)] = [0] * len(self.vert_lst)

			for vert in self.vert_lst:
				im_vert_cur = np.copy(im_vert_data)
				im_vert_cur[np.where(im_vert_cur!=vert)] =0
				if np.count_nonzero(im_vert_cur*np.copy(im_lesion_data)):
					dct_tmp = {}
					idx = self.dct_matrix[sheet_name][self.dct_matrix[sheet_name].vert==str(vert)].index
					for tract_id in dct_atlas_data:
						res_lst = relative_ROIvol_in_mask(np.copy(im_lesion_data_cur), np.copy(dct_atlas_data[tract_id]), p_lst, np.copy(im_vert_data), vert)
						dct_tmp[tract_id] = res_lst
					
					self.dct_matrix[sheet_name].loc[idx, 'PAM50_GM'] = self._regroup_per_tracts(dct_tmp, [30,35])
					self.dct_matrix[sheet_name].loc[idx, 'PAM50_WM'] = self._regroup_per_tracts(dct_tmp, [0,29])
					self.dct_matrix[sheet_name].loc[idx, 'PAM50_DC'] = self._regroup_per_tracts(dct_tmp, [0,3])
					self.dct_matrix[sheet_name].loc[idx, 'PAM50_VF'] = self._regroup_per_tracts(dct_tmp, [14,29])
					self.dct_matrix[sheet_name].loc[idx, 'PAM50_LF'] = self._regroup_per_tracts(dct_tmp, [4,13])

					for tract_id in dct_atlas_data:
						self.dct_matrix[sheet_name].loc[idx, 'PAM50_'+str(tract_id).zfill(2)] = dct_tmp[tract_id][0]*100.0/dct_tmp[tract_id][1]

	def _normalize(self, vect):
		norm = np.linalg.norm(vect)
		return vect / norm

	def angle_correction(self):

		if self.fname_sc is not None:
			im_seg = Image(self.fname_sc)
			data_seg = im_seg.data
			X, Y, Z = (data_seg > 0).nonzero()
			min_z_index, max_z_index = min(Z), max(Z)

			# fit centerline, smooth it and return the first derivative (in physical space)
			x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv = smooth_centerline(self.fname_sc, algo_fitting='hanning', type_window='hanning', window_length=80, nurbs_pts_number=3000, phys_coordinates=True, verbose=self.verbose, all_slices=False)
			centerline = Centerline(x_centerline_fit, y_centerline_fit, z_centerline, x_centerline_deriv, y_centerline_deriv, z_centerline_deriv)

			# average centerline coordinates over slices of the image
			x_centerline_deriv_rescorr, y_centerline_deriv_rescorr, z_centerline_deriv_rescorr = centerline.average_coordinates_over_slices(im_seg)[3:]

			# compute Z axis of the image, in physical coordinate
			axis_Z = im_seg.get_directions()[2]

			# Empty arrays in which angle for each z slice will be stored
			self.angles = np.zeros(im_seg.dim[2])

			# for iz in xrange(min_z_index, max_z_index + 1):
			for zz in range(im_seg.dim[2]):
				if zz >= min_z_index and zz <= max_z_index:
					# in the case of problematic segmentation (e.g., non continuous segmentation often at the extremities), display a warning but do not crash
					try: # normalize the tangent vector to the centerline (i.e. its derivative)
						tangent_vect = self._normalize(np.array([x_centerline_deriv_rescorr[zz], y_centerline_deriv_rescorr[zz], z_centerline_deriv_rescorr[zz]]))

					except IndexError:
						printv('WARNING: Your segmentation does not seem continuous, which could cause wrong estimations at the problematic slices. Please check it, especially at the extremities.', type='warning')

					# compute the angle between the normal vector of the plane and the vector z
					self.angles[zz] = np.arccos(np.vdot(tangent_vect, axis_Z))

	def label_lesion(self):
		printv('\nLabel connected regions of the masked image...', self.verbose, 'normal')
		im = Image(self.fname_mask)
		im_2save = im.copy()
		im_2save.data = label(im.data, connectivity=2)
		im_2save.setFileName(self.fname_label)
		im_2save.save()

		self.measure_pd['label'] = [l for l in np.unique(im_2save.data) if l]
		printv('Lesion count = '+str(len(self.measure_pd['label'])), self.verbose, 'info')

	def _orient(self, fname, orientation):

		im = Image(fname)
		im = set_orientation(im, orientation, fname_out=fname)

	def orient2rpi(self):

		# save input image orientation
		self.orientation = get_orientation(Image(self.fname_mask))

		if not self.orientation == 'RPI':
			printv('\nOrient input image(s) to RPI orientation...', self.verbose, 'normal')
			self._orient(self.fname_mask, 'RPI')
			
			if self.fname_sc is not None:
				self._orient(self.fname_sc, 'RPI')
			if self.fname_ref is not None:
				self._orient(self.fname_ref, 'RPI')
			if self.path_template is not None:
				self._orient(self.path_levels, 'RPI')
				for fname_atlas in self.atlas_roi_lst:
					self._orient(fname_atlas, 'RPI')

	def ifolder2tmp(self):
		# copy input image
		if self.fname_mask is not None:
			shutil.copy(self.fname_mask, self.tmp_dir)
			self.fname_mask = ''.join(extract_fname(self.fname_mask)[1:])
		else:
			printv('ERROR: No input image', self.verbose, 'error')

		# copy seg image
		if self.fname_sc is not None:
			shutil.copy(self.fname_sc, self.tmp_dir)
			self.fname_sc = ''.join(extract_fname(self.fname_sc)[1:])

		# copy ref image
		if self.fname_ref is not None:
			shutil.copy(self.fname_ref, self.tmp_dir)
			self.fname_ref = ''.join(extract_fname(self.fname_ref)[1:])

		# copy registered template
		if self.path_template is not None:
			shutil.copy(self.path_levels, self.tmp_dir)
			self.path_levels = ''.join(extract_fname(self.path_levels)[1:])

			self.atlas_roi_lst = []
			for fname_atlas_roi in os.listdir(self.path_atlas):
				if fname_atlas_roi.endswith('.nii.gz'):
					tract_id = int(fname_atlas_roi.split('_')[-1].split('.nii.gz')[0])
					if tract_id < 36: # Not interested in CSF
						shutil.copy(self.path_atlas+fname_atlas_roi, self.tmp_dir)
						self.atlas_roi_lst.append(fname_atlas_roi)

		os.chdir(self.tmp_dir) # go to tmp directory

def main(args=None):
  if args is None:
    args = sys.argv[1:]

  # get parser
  parser = get_parser()
  arguments = parser.parse(args)

  # set param arguments ad inputted by user
  fname_mask = arguments["-m"]

  # SC segmentation
  if '-s' in arguments:
    fname_sc = arguments["-s"]
    if not os.path.isfile(fname_sc):
      fname_sc = None
      printv('WARNING: -s input file: "' + arguments['-s'] + '" does not exist.\n', 1, 'warning')
  else:
    fname_sc = None

  # Reference image
  if '-i' in arguments:
    fname_ref = arguments["-i"]
    if not os.path.isfile(fname_sc):
      fname_ref = None
      printv('WARNING: -i input file: "' + arguments['-i'] + '" does not exist.\n', 1, 'warning')
  else:
    fname_ref = None

  # Path to template
  if '-f' in arguments:
    path_template = slash_at_the_end(arguments["-f"], slash=1)
    if not os.path.isdir(path_template) and os.path.exists(path_template):
      path_template = None
      printv("ERROR output directory %s is not a valid directory" % path_template, 1, 'error')
  else:
    path_template = None

   # Output Folder
  if '-ofolder' in arguments:
    path_results = slash_at_the_end(arguments["-ofolder"], slash=1)
    if not os.path.isdir(path_results) and os.path.exists(path_results):
      printv("ERROR output directory %s is not a valid directory" % path_results, 1, 'error')
    if not os.path.exists(path_results):
      os.makedirs(path_results)
  else:
    path_results = './'

  # Remove temp folder
  if '-r' in arguments:
    rm_tmp = bool(int(arguments['-r']))
  else:
    rm_tmp = True

  # Verbosity
  if '-v' in arguments:
    verbose = int(arguments['-v'])
  else:
    verbose = '1'

  # create the Lesion constructor
  lesion_obj = AnalyzeLeion(fname_mask=fname_mask, 
                            fname_sc=fname_sc, 
                            fname_ref=fname_ref, 
                            path_template=path_template, 
                            path_ofolder=path_results,
                            verbose=verbose)

  # run the analyze
  lesion_obj.analyze()

  # # remove tmp_dir
  # if rm_tmp:
  #   shutil.rmtree(lesion_obj.tmp_dir)
        
  # # printv('\nDone! To view the labeled lesion file (one value per lesion), type:', verbose)
  # # if fname_ref is not None:
  # #   printv('fslview ' + path_results + fname_mask + ' ' + path_results + lesion_obj.fname_label + ' -l Red-Yellow -t 0.7 & \n', verbose, 'info')
  # # else:
  # #   printv('fslview ' + path_results + lesion_obj.fname_label + ' -l Red-Yellow -t 0.7 & \n', verbose, 'info')    
    
if __name__ == "__main__":
    main()