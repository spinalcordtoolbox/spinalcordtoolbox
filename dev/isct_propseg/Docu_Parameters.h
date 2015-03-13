/*! \file Docu_Parameters.h
 * \page parameters_adjustment Parameters of the method
 * \brief How to adjust the parameters of the method ?
 *
 * IN CONSTRUCTION
 *
 * Depending on the clinical context or the image acquisition parameters, the method may need adjustment on its parameters to reach its full potential. This page provide such documentation on some parameter's adjustments.
 * 
 * Take also a look at the \ref correction_tips "Correction Tips" page where several advices for segmentation correction (e.g. providing centerline) are presented.
 *
 * \section init Initialisation
 * Several parameters are linked with the initialisation module. This initialisation make an elliptical Hough transform on multiple axial slices in order to compute the spinal cord position and orientation (in both directions). The default position of the initialisation is the middle slice in the inferior-superior direction but this position can be set using "-init" parameter.
 * 
 * In case of low contrast or bad performance of the Hough transform, two paremeters can be adjusted : the number of slices ("-detect-n") and the gap btween these slices ("-detect-gap"). More slices will increase the power of the spinal cord detection while an adapted gap (depending on the image resolution) will increase the precision of the direction computation.
 * 
 * The initial radius of circles/ellipses used by the Hough transform can be set using the parameter "-detect-radius".
 * 
 * The result of the spinal cord detection can be visualized using the parameter "-detection-display" which save the spinal cord position on the initial slice in a PNG image.
 *
 */