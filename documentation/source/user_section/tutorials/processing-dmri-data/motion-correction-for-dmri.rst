Motion correction for dMRI images
#################################

SCT features a complex motion correction algorithm, which is inspired by `[Xu et al., Neuroimage 2013] <https://pubmed.ncbi.nlm.nih.gov/23178538/>`_. The key aspects of this algorithm are as follows:

.. TODO: The explanation of the motion correction algorithm is a little confusing to me. Rewrite into a step-by-step process? Make it more brief? Leave some of the explanation for the ``-h`` description?

    * **SliceReg:** Slice-wise registration regularized along the Z direction (based on the function antsSliceRegularizedRegistration from ANTs, and described in [De Leener et al., Neuroimage 2017]).
    * **Grouping:** there is the possibility to group successive volumes in order to have sufficient SNR to estimate a reliable transformation. If your data are very low SNR you can increase the number of successive images that are averaged into group with the flag ``-g``.
    * **Iterative average:** after registering a new group to the target image (which is usually the first DWI group), the target is averaged with the newly registered group in order to increase the SNR of the target image.
    * **Outlier detection:** if a detected transformation is too large, it is ignored and the previous transformation is used instead.
    * **Masking:** in order to estimate motion of the cord, ignoring the rest of the tissue, there is the possibility to include a mask with the flag ``-m``.

.. code::

   sct_dmri_moco -i dmri_crop.nii.gz -bvec bvecs.txt

:Input arguments:
   - ``-i`` : The input dMRI image.
   - ``-bvec`` : A text file with three lines, each containing a value for each volume in the input image. Together, the the three sets of values represent the ``(x, y, z)`` coordinates of the b-vectors, which indicate the direction of the diffusion encoding for each volume of the dMRI image.

:Output files/folders:
   - ``dmri_crop_moco.nii.gz`` : The motion-corrected 4D dMRI image.
   - ``dmri_crop_moco_b0_mean.nii.gz`` : The time-average of the motion-corrected 3D volumes with ``b == 0``.
   - ``dmri_crop_moco_dwi_mean.nii.gz`` : The time-average of the motion-corrected 3D volumes with ``b != 0``. (This image is what will be used for the template registration.)
   - ``moco_params.tsv`` : A text file that provides a simplified overview of the motion correction, to be used for quality control. It contains the slicewise average of the axial (X-Y) translations for each 3D volume. (In reality, though, each slice of each volume will have had a different translation applied to it.)
   - ``moco_params_x.nii.gz`` : A 4D image with dimensions ``[1, 1, z, t]``. Each voxel contains the ``x`` translation corresponding to each ``z`` slice across each ``t`` volume.
   - ``moco_params_y.nii.gz`` : A 4D image with dimensions ``[1, 1, z, t]``. Each voxel contains the ``y`` translation corresponding to each ``z`` slice across each ``t`` volume.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/jn/2857-add-remaining-tutorials/processing-dmri-data/io-sct_dmri_moco.png
   :align: center