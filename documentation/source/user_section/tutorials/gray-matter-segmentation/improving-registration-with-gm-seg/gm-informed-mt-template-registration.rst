.. _gm-informed-mt-registration:

Reusing the GM-informed warping field to improve MTI registration
#################################################################

Now that we have estimated a transformation between the template and the T2* data, we can use this warping field to improve a registration command from :ref:`the previous MT registration tutorial step <mt-registraton-with-anat>`.

First, we return to the ``mt`` directory.

.. code::

   cd ../mt

Next, we run ``sct_register_multimodal`` to compute the transformation between the MT space and the PAM50 template space.

.. code:: sh

   sct_register_multimodal -i "${SCT_DIR}/data/PAM50/template/PAM50_t2.nii.gz" \
                           -iseg "${SCT_DIR}/data/PAM50/template/PAM50_cord.nii.gz" \
                           -d mt1.nii.gz \
                           -dseg mt1_seg.nii.gz \
                           -param step=1,type=seg,algo=centermass:step=2,type=seg,algo=bsplinesyn,slicewise=1,iter=3 \
                           -m mask_mt1.nii.gz \
                           -initwarp ../t2s/warp_template2t2s.nii.gz \
                           -owarp warp_template2mt.nii.gz \
                           -qc ~/qc_singleSubj

:Input arguments:
   - The input and arguments are identical to the arguments from :ref:`the previous MT registration tutorial step <mt-registraton-with-anat>`, apart from one argument.
   - ``-initwarp`` : Here, we specify the T2* warping field generated in the previous step. We do this because information about the GM shape is enclosed in the transformation, producing registration results that are supposedly more accurate.

:Output files/folders:
   - ``PAM50_t2_reg.nii.gz`` : The PAM50 template image, registered to the space of the MT1 image.
   - ``warp_template2mt.nii.gz`` : The warping field to transform the PAM50 template to the MT1 space.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/improving-registration-with-gm-seg/io-sct_register_multimodal-mt.png
   :align: center

This transformation can be used to warp the template to the MT space, which allows metrics to be extracted for specific vertebral levels, WM/GM tracts, and more.