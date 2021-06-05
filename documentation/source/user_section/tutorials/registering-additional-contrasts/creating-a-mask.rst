Creating a mask around the segmentation
#######################################

Next, we will create a `binary mask <https://homepages.inf.ed.ac.uk/rbf/HIPR2/mask.htm>`_ to focus on the region of interest, which will increase the accuracy of the registration. Importantly, this mask is used to exclude the tissue surrounding the spinal cord, because it can move independently of the cord and negatively impact the registration.

.. code:: sh

   sct_create_mask -i mt1.nii.gz -p centerline,mt1_seg.nii.gz -size 35mm -f cylinder -o mask_mt1.nii.gz

:Input arguments:
   - ``-i`` : Input image.
   - ``-p`` : Process to generate mask. By specifying 'centerline,mt1_seg.nii.gz', we tell the command to create a mask centered around the spinal cord centerline by using the segmentation file 'mt1_seg.nii.gz'
   - ``-size`` : Size of the mask in the axial plane. (You can also specify size in pixels by omitting 'mm'.)
   - ``-f`` : Shape of the mask.
   - ``-qc`` : Directory for Quality Control reporting. QC reports allow us to evaluate the results slice-by-slice.

:Output files/folders:
   - ``mt1_seg.nii.gz`` : 3D binary mask of the segmented spinal cord

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/registration_to_template/io-mt-sct_create_mask.png
   :align: center
   :figwidth: 65%

   Input/output images for ``sct_create_mask``.