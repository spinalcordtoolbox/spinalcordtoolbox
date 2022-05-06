Motion correction for fMRI images
#################################

Now that we have a mask highlighting the spinal cord, we can apply motion correction to the volumes of the fMRI data.

The motion correction algorithm
-------------------------------

SCT features a complex motion correction algorithm, which is inspired by `[Xu et al., Neuroimage 2013] <https://pubmed.ncbi.nlm.nih.gov/23178538/>`_. The key aspects of this algorithm are as follows:

* **SliceReg:** Slice-wise registration regularized along the Z direction (based on the function antsSliceRegularizedRegistration from ANTs, and described in `[De Leener et al., Neuroimage 2017] <https://pubmed.ncbi.nlm.nih.gov/27720818/>`_).
* **Grouping:** The algorithm performs group-wise registration between groups of successive dMRI volumes. If your data has a very low signal-to-noise ratio (SNR), you can use the flag ``-g`` to increase the number of successive images that are averaged into a group in order to have sufficient SNR to estimate a reliable transformation.
* **Iterative average:** After registering a new group to the target image (which is usually the first DWI group), the target is averaged with the newly registered group in order to increase the SNR of the target image.
* **Outlier detection:** If a detected transformation is too large, it is ignored and the previous transformation is used instead.
* **Masking:** You can use the flag ``-m`` to provide a spinal cord mask, in order to estimate motion of the cord while ignoring the rest of the tissue.

Applying the algorithm
----------------------

To apply the algorithm, we use the ``sct_fmri_moco`` command:


.. code::

   sct_fmri_moco -i fmri.nii.gz -m mask_fmri.nii.gz -qc ~/qc_singleSubj -qc-seg t2_seg_reg.nii.gz

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

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/processing-fmri-data/io-sct_fmri_moco.png
   :align: center