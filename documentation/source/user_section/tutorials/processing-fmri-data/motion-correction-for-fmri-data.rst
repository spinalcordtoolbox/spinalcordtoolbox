Motion correction for fMRI data
###############################

Now that we have a mask highlighting the spinal cord, we can apply motion correction to the volumes of the fMRI data.

.. code::

   sct_fmri_moco -i fmri.nii.gz -m mask_fmri.nii.gz  -qc ~/qc_singleSubj -qc-seg t2_seg_reg.nii.gz

:Input arguments:
   - ``-i`` : The input fMRI image.
   - ``-m`` : A mask used to limit the voxels considered by the motion correction algorithm.
   - ``-qc`` : Directory for Quality Control reporting. QC reports allow us to evaluate the results slice-by-slice.
   - ``-qc-seg`` :  Segmentation of spinal cord to improve cropping in QC report.

:Output files/folders:
   - ``fmri_moco.nii.gz`` : The motion-corrected 4D fMRI image.
   - ``fmri_moco_mean.nii.gz`` : The time-average of the motion-corrected 3D volumes of the fMRI image.
   - ``moco_params.tsv`` : A text file that provides a simplified overview of the motion correction, to be used for quality control. It contains the slicewise average of the axial (X-Y) translations for each 3D volume. (In reality, though, each slice of each volume will have had a different translation applied to it.)
   - ``moco_params_x.nii.gz`` : A 4D image with dimensions ``[1, 1, z, t]``. Each voxel contains the ``x`` translation corresponding to each ``z`` slice across each ``t`` volume.
   - ``moco_params_y.nii.gz`` : A 4D image with dimensions ``[1, 1, z, t]``. Each voxel contains the ``y`` translation corresponding to each ``z`` slice across each ``t`` volume.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/jn/2857-add-remaining-tutorials/processing-fmri-data/io-sct_fmri_moco.png
   :align: center