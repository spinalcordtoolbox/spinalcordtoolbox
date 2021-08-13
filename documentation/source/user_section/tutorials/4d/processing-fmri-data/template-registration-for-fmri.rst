Registering fMRI data to the PAM50 template
###########################################

Now that we have the motion-corrected time-averaged fMRI image, we can use it to register the template to the fMRI image space.

.. code::

   sct_register_multimodal -i "${SCT_DIR}/data/PAM50/template/PAM50_t2s.nii.gz" \
                           -d fmri_moco_mean.nii.gz \
                           -dseg t2_seg_reg.nii.gz \
                           -param step=1,type=im,algo=syn,metric=CC,iter=5,slicewise=0 \
                           -initwarp ../t2s/warp_template2t2s.nii.gz \
                           -initwarpinv ../t2s/warp_t2s2template.nii.gz \
                           -qc ~/qc_singleSubj

.. TODO: I don't understand the choices made for ``-dseg`` and ``-iseg``.

   * Why do we supply a segmentation at all if we're using ``type=im``? Won't this ignore the segmentation image?
   * Why do we supply t2_seg_reg.nii.gz for ``-dseg`` when ``-d`` is fmri_moco_mean? These don't seem to match up, since ``t2_seg_reg.nii.gz`` was registered to the **non-motion-corrected** fMRI image.
   * Why don't we supply ``-iseg``? Wouldn't "$SCT_DIR/data/PAM50/template/PAM50_cord.nii.gz" be suitable?

:Input arguments:
   - ``-i`` : Source image. Here, we select the T2* version of the PAM50 template, because the T2* contrast is the closest visual match to our fMRI data.
   - ``-d`` : Destination image.
   - ``-param`` : The parameter settings worth noting are:
      - ``type=im`` : Since fMRI cannot be segmented reliably (due to low contrast between the spinal cord and the surrounding cerebrospinal fluid), we rely on just the anatomical images (``-i`` and ``-d``) rather than segmentation images (``-iseg`` and ``-dseg``).
      - ``algo=syn`` : This algorithm helps to compensate for the the lack of segmentation during registration. (TODO: The presenter notes describes this as "ANTs Superpower". But, can we give a more specific reason for why this is needed here in particular? My intent is to answer the question "If syn is so good, why don't we use it elsewhere by default? Why is it suited specifically for this situation?)
      - ``iter=5``: We decrease the number of iterations (default ``10``) because the registration is sensitive to the artifacts (drop out) in the image. (TODO: The presenter notes could be more specific here, too. Why is this registration specifically sensitive to artifacts? Why don't we do this elsewhere? Is it due to fMRI data? Is it because we're not using a segmentation here?)
      - ``slicewise=0``: This setting regularizes the transformations across the Z axis. (TODO: The presenter notes say that this is needed, but they don't provide a justification for why.)
   - ``-initwarp`` : TODO: Is this necessary? I understand that it's meant to improve the registration with the GM/WM-informed warping field, but I worry that it makes this tutorial unnecessarily complex for people who will read it as a standalone tutorial. So, I'm wondering if the T2* warping fields should be kept to the GM/WM tutorials only.
   - ``-initwarpinv`` : TODO: Is this necessary? (See above.)
   - ``-qc`` : Directory for Quality Control reporting. QC reports allow us to evaluate the results slice-by-slice.

:Output files/folders:
   - ``PAM50_t2s_reg.nii.gz`` : The PAM50 template image, registered to the space of dMRI data.
   - ``fmri_moco_mean_reg.nii.gz`` : The mean fMRI image, registered to the space of the PAM50 template.
   - ``warp_PAM50_t2s2fmri_moco_mean.nii.gz`` : The warping field to transform the PAM50 template to the fMRI space.
   - ``warp_fmri_moco_mean2PAM50_t2s.nii.gz`` : The warping field to transform the fMRI data to the PAM50 template space.

Finally, we rename the warping fields for convenience, since the automatically generated filenames are a little verbose.

.. code::

   mv warp_PAM50_t2s2fmri_moco_mean.nii.gz warp_template2fmri.nii.gz
   mv warp_fmri_moco_mean2PAM50_t2s.nii.gz warp_fmri2template.nii.gz