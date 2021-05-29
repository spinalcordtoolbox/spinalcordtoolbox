.. _mt-registraton-without-anat:

Registeration Option 2: Direct registration to the template
###########################################################

In the case that you have only the MT data without the anatomical data, you can still perform registration. To do so, all you will need to do is apply the same vertebral labeling and template registration steps that were covered in :ref:`template-registration`.

First, we create one or two labels in the metric space. For example, if you know that your FOV is centered at C3/C4 disc, then you can create a label automatically with:

.. code:: sh

   sct_label_utils -i mt1_seg.nii.gz -create-seg-mid 4 -o label_c3c4.nii.gz

Then, you can register to the template. Note: In case the metric image has axial resolution with thick slices, we recommend to do the registration in the subject space (instead of the template space), without cord straightening.

.. code:: sh

   sct_register_to_template -i mt1.nii.gz -s mt1_seg.nii.gz -ldisc label_c3c4.nii.gz -ref subject \
                            -param step=1,type=seg,algo=centermassrot:step=2,type=seg,algo=bsplinesyn,slicewise=1

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/registration_to_template/io-mt-sct_register_multimodal-template.png
   :align: center
   :figwidth: 65%

   Input/output images for ``sct_register_to_template`` using MT1 data.

.. important::

   Only use this method if you don't also have anatomical data. If you do have anatomical data, we recommend that you stick with :ref:`mt-registraton-with-anat`. By reusing the registration results, you ensure that you use a consistent transformation between each contrast in your analysis.