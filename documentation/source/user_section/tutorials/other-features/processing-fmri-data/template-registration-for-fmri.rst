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
                           -owarp warp_template2fmri.nii.gz \
                           -owarpinv warp_fmri2template.nii.gz \
                           -qc ~/qc_singleSubj

:Input arguments:
   - ``-i`` : Source image. Here, we select the T2* version of the PAM50 template, because the T2* contrast is the closest visual match to our fMRI data.
   - ``-d`` : Destination image.
   - ``-dseg`` : Segmentation corresponding to the destination image. Note, however, that because we supply ``type=im`` to ``-param``, the segmentation will be ignored during registration. This is intentional, as the `t2` segmentation would only coarsely match our fMRI data. The reason we supply the segmentation anyway is because ``-dseg`` is necessary to generate a QC report; it is used to roughly crop around the cord for visualization purposes (so it doesn't need to be perfect).
   - ``-param`` : The parameter settings worth noting are:
      - ``type=im`` : Since fMRI cannot be segmented reliably (due to low contrast between the spinal cord and the surrounding cerebrospinal fluid), we rely on just the anatomical images (``-i`` and ``-d``) rather than segmentation images (``-iseg`` and ``-dseg``).
      - ``algo=syn`` : This algorithm helps to compensate for the the lack of segmentation during registration. (TODO: The presenter notes describes this as "ANTs Superpower". But, can we give a more specific reason for why this is needed here in particular? My intent is to answer the question "If syn is so good, why don't we use it elsewhere by default? Why is it suited specifically for this situation?)
      - ``iter=5``: We decrease the number of iterations (default ``10``) because the registration is sensitive to the artifacts (drop out) in the image. (TODO: The presenter notes could be more specific here, too. Why is this registration specifically sensitive to artifacts? Why don't we do this elsewhere? Is it due to fMRI data? Is it because we're not using a segmentation here?)
      - ``slicewise=0``: This setting regularizes the transformations across the Z axis. (TODO: The presenter notes say that this is needed, but they don't provide a justification for why.)
   - ``-initwarp`` : TODO: Is this necessary? I understand that it's meant to improve the registration with the GM/WM-informed warping field, but I worry that it makes this tutorial unnecessarily complex for people who will read it as a standalone tutorial. So, I'm wondering if the T2* warping fields should be kept to the GM/WM tutorials only.
   - ``-initwarpinv`` : TODO: Is this necessary? (See above.)
   - ``-owarp``: The name of the output warping field. This is optional, and is only specified here to make the output filename a little clearer. By default, the filename would be automatically generated from the filenames ``-i`` and ``-d``, which in this case would be the (less clear) ``warp_PAM50_t2s2fmri_moco_mean.nii.gz``.
   - ``-owarpinv`` : The name of the output inverse warping field. This is specified for the same reasons as ``-owarp``.
   - ``-qc`` : Directory for Quality Control reporting. QC reports allow us to evaluate the results slice-by-slice.

:Output files/folders:
   - ``PAM50_t2s_reg.nii.gz`` : The PAM50 template image, registered to the space of dMRI data.
   - ``fmri_moco_mean_reg.nii.gz`` : The mean fMRI image, registered to the space of the PAM50 template.
   - ``warp_template2fmri.nii.gz`` : The warping field to transform the PAM50 template to the fMRI space.
   - ``warp_fmri2template.nii.gz`` : The warping field to transform the fMRI data to the PAM50 template space.