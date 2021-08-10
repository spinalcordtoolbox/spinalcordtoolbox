GM-informed registration between the PAM50 template and T2* data
################################################################

First, we will need to generate the warping fields that represent the transformation between the T2* space and the PAM50 template space. Once we have that transformation, we can then use it to improve the registration results of MT data.

This registration step is similar to that of previous tutorials. However, a notable difference is that rather than using the full spinal cord segmentations for the arguments ``-iseg`` and ``-dseg``, we use the white matter segmentation. This is because its interior perimeter outlines the gray matter shape, while its exterior perimeter outlines the shape of the spinal cord itself, which better informs the registration algorithm.

.. code:: sh

   sct_register_multimodal -i "${SCT_DIR}/data/PAM50/template/PAM50_t2s.nii.gz" \
                           -iseg "${SCT_DIR}/data/PAM50/template/PAM50_wm.nii.gz" \
                           -d t2s.nii.gz \
                           -dseg t2s_wmseg.nii.gz \
                           -initwarp ../t2/warp_template2anat.nii.gz \
                           -initwarpinv ../t2/warp_anat2template.nii.gz \
                           -param step=1,type=seg,algo=rigid:step=2,type=seg,algo=bsplinesyn,slicewise=1,iter=3 \
                           -qc ~/qc_singleSubj

:Input arguments:
   - ``-i`` : Source image. Here, we select the T2* version of the PAM50 template.
   - ``-iseg`` : Segmentation for the source image. Here, we use the PAM50 segmented white matter volume, rather than the spinal cord volume. This allows us to account for both the cord and the gray matter shape.
   - ``-d`` : Destination image.
   - ``-dseg`` : Segmentation for the destination image. Here, we use the white matter segmentation for the same reasons as ``-iseg``.
   - ``-initwarp`` : Warping field used to initialize the source image. Here, we supply the ``warp_template2anat.nii.gz`` file from the previous T2 registration. (See: :ref:`template-registration`)
      - TODO: I'm concerned that the usage of the ``-initwarp`` flag is not clear enough. For users who are following this tutorial standalone, they may not have the full context of the previous tutorial, so the usage of ``-initwarp`` may appear to "come out of nowhere". But, even if they have the context, reusing intermediate results raises some questions:
          - Is ``-initwarp`` optional or mandatory?
          - What should the command look like without ``-initwarp``?
          - Will the registration results be significantly worse without ``-initwarp``?
          - In what cases should ``-initwarp`` be used, and in what cases can they perform registration directly?
   - ``-initwarpinv``: Warping field used to initialize the destination image. Here, we supply the inverse warping field, ``warp_anat2template.nii.gz`` from the previous T2 registration. (See: :ref:`template-registration`)
      - TODO: Same concerns as above.
   - ``-param`` :
      - TODO: Why are we using ``rigid`` specifically? The MT tutorial uses ``centermass`` instead... do we need to explain the discrepancy? (Not explained in SCT course.)
   - ``-qc`` : Directory for Quality Control reporting. QC reports allow us to evaluate the results slice-by-slice.

:Output files/folders:
   - ``PAM50_t2s_reg.nii.gz`` : The PAM50 template image, registered to the space of the T2s image.
   - ``t2s_reg.nii.gx``: The T2s image, registered to the space of the PAM50 template.
   - ``warp_PAM50_t2s2t2s.nii.gz`` : The warping field to transform the PAM50 template to the T2s space.
   - ``warp_t2s2PAM50_t2s.nii.gz`` : The warping field to transform the T2s image to the PAM50 template space.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/jn/2857-add-remaining-tutorials/improving-registration-with-gm-seg/io-sct_register_multimodal-t2s.png
   :align: center

Finally, it is also worth renaming the automatically generated warping fields for clarity.

.. code:: sh

   mv warp_PAM50_t2s2t2s.nii.gz warp_template2t2s.nii.gz
   mv warp_t2s2PAM50_t2s.nii.gz warp_t2s2template.nii.gz

.. TODO: I've excluded the ``sct_warp_template`` step here because the T2* warped template was not actually used for anything in later steps.

   Also, my intent (more generally) is to convey that the main result of registration is to produce warping fields, and that those warping fields are the thing that should be passed along to other tutorials. (i.e. later in the "metric tutorial" I want to communicate that the user needs a warping field from a previous registration, that way they can warp the template *for the purposes of metric extraction*.)
