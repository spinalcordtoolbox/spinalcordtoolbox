Overcoming the partial volume effect
####################################

Because of its simplicity, the traditional method to quantify metrics is to use a binary mask: voxels labeled as “1” (i.e. in the mask) are selected and values within those voxels are averaged to produce, e.g., the average FA inside the spinal cord. As mentioned in the previous slide, this method suffers from partial volume effect and the resulting metric could be biased by the surrounding tissues.

Instead of using binary masks, we could use weighted masks, which effectively “weight” the contribution of voxels at the interface (e.g., mask value = 0.1) vs. voxels well within the tissue of interest (e.g., mask value = 0.9). While this method (available as -method wa in SCT) minimizes PVE, it does not solve it.

Alternatively, methods using Gaussian mixture modeling can be used to estimate the “true” value within the region of interest. Here, we benefit from the fact that the metric is measured within thousands of voxels, where the partial volume for each compartment (e.g., white matter, gray matter, CSF) is known. We then use a maximum likelihood estimation (-method ml) to estimate the metric value within each compartment. This approach assumes that within each compartment, the metric is homogeneous.

Because Maximum Likelihood estimation is sensitive to noise, especially in small tracts, we recommend using the Maximum a Posteriori (-method map), which uses a prior based on the average metric value within the CSF, WM and GM compartment.

Note that methods ml and map need prior warping of the WM atlas to your data, whereas you could use bin and wa with any binary mask.