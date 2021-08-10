Overcoming the partial volume effect
####################################

Each metric extraction method provided by SCT accounts for the partial volume effect in different ways.

Binary mask-based methods
*************************

``-method bin``: Average within binary ROI
------------------------------------------

Because of its simplicity, the traditional method to quantify metrics is to use a binary mask: voxels labeled as “1” (i.e. in the mask) are selected and values within those voxels are averaged.

This method does not account for the partial volume effect whatsoever, and thus the resulting metric could be biased by the surrounding tissues, as demonstrated on the previous page.

``-method wa``: Weighted average
--------------------------------

Instead, we could turn the binary masks into weighted masks by effectively “weighting” the contribution of voxels at the interface (e.g., mask value = 0.1) vs. voxels well within the tissue of interest (e.g., mask value = 0.9).

This method is only useful if you have an existing binary mask for each region of interest. Also, this method only considers each mask in isolation, rather than considering the relations between adjacent masks. So, while it would help to minimize the partial volume effect, it would not comprehensively solve the problem.

Atlas-based methods
*******************

Instead of using binary masks, we can use the white and gray matter atlas contained within the PAM50 template.

.. TODO: There is some additional info I haven't included from the presenter's notes:

   "Here, we benefit from the fact that the metric is measured within thousands of voxels, where the partial volume for each compartment (e.g., white matter, gray matter, CSF) is known.""

   I'm not 100% certain I understand this quotation. Questions:

   - Thousands of voxels? How so? Isn't a single tract barely even covered by one voxel? Is it the *atlas* that includes thousands of voxels?
   - "partial volume is known"? How so? Is this referring to the individual images for each tract in the atlas?

``-method ml``: Maximum Likelihood
----------------------------------

The partial volume information from the atlas can be combined with Gaussian mixture modeling and maximum likelihood estimation to estimate the “true” value within the region of interest (e.g. a white matter tract). This approach assumes that within each compartment, the metric is homogeneous.

``-method map``: Maximum A Posteriori
-------------------------------------

Because Maximum Likelihood estimation is sensitive to noise, especially in small tracts, we recommend using a modified version of Maximum Likelihood called Maximum a Posteriori. This method uses a prior based on the average metric value within the CSF, WM and GM compartments.

The ``map`` method is the most robust to noise in small tracts. This was further validated using bootstrap simulations based on a synthetic MRI phantom. For more details, see `[Lévy et al., Neuroimage 2015] <https://pubmed.ncbi.nlm.nih.gov/26099457/>`_ (construction of the phantom, effect of noise, contrast) and `[De Leener et al., Neuroimage 2017; Appendix] <https://pubmed.ncbi.nlm.nih.gov/27720818/>`_ (effect of spatial resolution).

.. TODO: Is the prior taken from the atlas (not specific to any particular metric)? Or is it computed uniquely for each image?

.. note:: The methods ``bin`` and ``wa`` can be used with any binary mask. However, the methods ``ml`` and ``map`` require you to warp the white matter atlas to the coordinate space of your data, as is shown on the next page.