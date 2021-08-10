.. TODO:

   Is this one-page tutorial necessary? It is basically just telling users that the ``sct_smooth_spinalcord`` tool exists. (Compared to other tutorials, which demonstrate multi-step workflows.)

   So, I am thinking that maybe this page will be unnecessary once we organize the "Command-Line Tools" page into one-page-per-script. We could simply have all of this information on the dedicated "sct_smooth_spinalcord" page instead, and save the "Tutorials" for complex workflows only.

.. _spinalcord-smoothing:

Spinal cord smoothing as a preprocessing operation
##################################################

SCT provides a spinal cord smoothing function that takes into account the curvature of the cord, preserving the boundary between the spinal cord and cerebrospinal fluid. This is useful in a variety of situations:

* It can be used to improve sensitivity of fMRI results while minimizing contamination from CSF
* It can also be used to obtain more reliable cord segmentations, because smoothing will sharpen the edge of the cord and will blur out possible artifacts at the cord/CSF interface.

.. code::

   sct_smooth_spinalcord -i t1.nii.gz -s t1_seg.nii.gz

:Input arguments:
   - ``-i`` : The input image.
   - ``-s`` : A spinal cord segmentation mask corresponding to the input image. This is needed as ``sct_smooth_spinalcord`` performs a 1D smoothing operation following the cord centerline (as opposed to the cortical surface smoothing in FreeSurfer, which is 2D).

:Output files/folders:
   - ``t1_smooth.nii`` : The input image, smoothed along the spinal cord. (TODO: Why are we outputting the uncompressed ``.nii`` file here?
   - ``warp_curve2straight.nii.gz`` : ``sct_smooth_spinalcord`` involves an intermediate straightening step, so this is the 4D warping field that defines the transform from the original curved anatomical image to the straightened image.
   - ``warp_straight2curve.nii.gz`` : ``sct_smooth_spinalcord`` involves an intermediate straightening step, so this is the 4D warping field that defines the inverse transform from the straightened anatomical image back to the original curved image.
   - ``straightening.cache`` : SCT functions that require straightening will check for this file. If it is present in the working directory, ``straight_ref.nii.gz`` and the two warping fields will be re-used, saving processing time. (TODO: ``straight_ref.nii.gz`` is not output, making this seemingly uselesss. Is this a bug?)

After smoothing, the apparent noise is reduced, while the cord edges are preserved, allowing a more reliable segmentation on a second pass.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/jn/2857-add-remaining-tutorials/spinalcord-smoothing/io-sct_smooth_spinalcord.png
   :align: center