/*! \file Docu_Correction.h
 * \page correction_tips Correction Tips
 * \brief What to do if the propagated segmentation fails or contains local errors ?
 *
 * Due to contrast variations in MR imaging protocols, the contrast between the spinal cord and the cerebro-spinal fluid (CSF) can differ between MR volumes. Therefore, our propagated segmentation method may fail sometimes in presence of artefact, low contrast, etc. Here above, we propose some protocols to correct some segmentation failures.
 * 
 * \section orient Bad orientation and lack of CSF
 * The computation of the spinal cord orientation, at each iteration of the propagation, can fail in lack of spinal cord/CSF contrast. Particularly, this situation can lead to an local over-segmentation or even to a propagation which has stopped too soon, resulting in a partial spinal cord segmentation.
 * 
 * Two correction protocols can be used to improve the segmentation : <b>add centerline information</b> and <b>correct the image</b>
 * 
 * \subsection centerline Add the spinal cord centerline
 * The spinal cord orientation is computed at each propagation iteration by minimizing/maximizing (depending on the contrast type) the sum of gradient magnitude at vertices positions. Bad contrast or error propagation can make orientation computation difficult.
 * 
 * Centerline information can be provided (using "-init-centerline" parameter) to assure a correct orientation of the propagated deformable model. Spinal cord centerline can be a nifti image, with non-null values on centerline voxels. The orientation of the spinal cord will then be computed using a B-spline approximating the set of points extracted from this input image. Propagation will start at the center of the centerline (this can be change using "-init" parameter) and stop at its edges. Centerline can also be provided by a texte file, where each row contain x, y and z world coordinates (not pixel coordnates) of a point of the spinal cord, from the bottom to the top of the spinal cord.
 *
 * \subsection image_correct Correction of the image
 * MR images can sometimes present local absence of contrast, making the spinal cord segmentation impossible. Therefore, this situation can only be resolved by correcting the initial image. FSL View (http://fsl.fmrib.ox.ac.uk/fsl/fslview/) is a powerfull software allowing to easily create a mask or correct an image.
 * 
 * You can enhance the contrast between the spinal ord and the CSF by changing the values of voxels when necessary.
 * 
 * 
 * \section detect Wrong spinal crod detection
 * In precence of other circular or elliptical structures in the image and at low contrast-to-noise ratio, the spinal cord detection module can fail. You can verify the detection by using "-detect-display" parameter, writing a PNG axial image with a white cross at the detected spinal cord position.
 * 
 * In case of failure, two choices are possible:
 * 1. You change the starting position of the propagation (and the detection) in the image, by using "-init" parameter. You can provide a fraction (between 0 and 1) of the image in the inferior-superior director or the number of the desired slice.
 * 2. Using "-init-mask" parameter, you can provide a binary mask (an MR volume, same header than input image) containing three non-null voxel, at the center of the spinal cord, separated by 4-5 mm in the axial direction. The center point will be used as the starting position and the two other point will be used to compute the spinal cord orientation in the top and bottom direction. At the same time, you should provide the initial spinal cord radius approximation using the parameter "-detect-radius". The default value is 4 mm.
 *
 */
