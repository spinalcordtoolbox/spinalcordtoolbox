.. _registration-to-template:

Tutorial 2: Registering T2 images to the PAM50 template
#######################################################

This tutorial demonstrates a multi-step pipeline to register an anatomical MRI scan to the PAM50 Template. While T2 images are used for this tutorial, each step is applicable to multiple contrast types (T1, T2, T2*).

Before starting this tutorial
*****************************

1. Read through the following pages to familiarize yourself with key SCT concepts:

    * :ref:`pam50`: An overview of the PAM50 template's features, as well as context for why the template is used.
    * :ref:`warping-fields`: Background information on the image transformation format used by the registration process.
    * :ref:`qc`: Primer for SCT's Quality Control interface. After each step of this tutorial, you will be able to open a QC report that lets you easily evaluate the results of each command.

2. Download and unzip `sct_course_london20.zip <https://osf.io/bze7v/?action=download>`_.
3. Open a terminal and navigate to the ``sct_course_london20/single_subject/data/t2/`` directory.

----------

.. _segmentation-section:

Step 1: Segmenting the spinal cord
**********************************

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/registration_to_template/registration-pipeline-1.png
   :align: center
   :figwidth: 100%

First, we will run the ``sct_deepseg_sc`` command to segment the spinal cord from the anatomical image.

.. code:: sh

   sct_deepseg_sc -i t2.nii.gz -c t2 -qc ~/qc_singleSubj

   # Input arguments:
   #   - i: Input image
   #   - c: Contrast of the input image
   #   - qc: Directory for Quality Control reporting. QC reports allow us to evaluate the segmentation slice-by-slice

   # Output files/folders:
   #   - t2_seg.nii.gz: 3D binary mask of the segmented spinal cord

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/spinalcord_segmentation/t2_propseg_before_after.png
   :align: center
   :figwidth: 65%

   Input/output images for ``sct_deepseg_sc``.


----------


.. _vert-labeling-section:

Step 2: Vertebral/disc labeling
*******************************

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/registration_to_template/registration-pipeline-2.png
   :align: center
   :figwidth: 100%

Next, the segmented spinal cord must be labeled to provide reference markers for matching the PAM50 template to subject's MRI.

Labeling conventions
====================

SCT accepts two different conventions for vertebral labels:

1. **Vertebral levels**: For this type of label file, labels should be placed as though the vertebrae were projected onto the spinal cord, with the label centered in the middle of each vertebral level.
2. **Intervertebral discs**: For this type of label file, labels should placed on the posterior tip of each disc.

   * **Note:** For disc labeling, SCT generates additional labels for the pontomedullary groove (label 49) and pontomedullary junction (label 50). However, for now these are not used in any subsequent steps.

For image registration, you can provide either vertebral body labels or disc labels, as the decision does not significantly impact the performance of the registration.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/registration_to_template/vertebral-labeling-conventions.png
   :align: center
   :figwidth: 600px

   Conventions for vertebral and disc labels.

Labeling algorithm
==================

The steps of the labeling algorithm are as follows:

  #. **Straightening**: The spinal cord is straightened to make it easier to use a moving window-based approach in a subsequent step.
  #. **C2-C3 disc detection:** The C2-C3 disc is used as a starting point because it is a distinct disc that is easy to detect (compared to, say, the T7-T9 discs, which are indistinct compared to one another).
  #. **Labeling of neighbouring discs**: The neighbouring discs are found using a similarity measure with the PAM50 template at each specific level.
  #. **Un-straightening**: Finally, the spinal cord and the labeled segmentation are both un-straightened, and the labels are saved to image files.

The vertebral/disc labeling algorithm has the following features:

  - **Contrast-independent**: Can be used on images regardless of their contrast type.
  - **Produces both label types:** Labels are produced for both vertebral levels and intervertebral discs.
  - **Robust to missing discs:** The labeling algorithm uses several priors from the template, including the probabilistic distance between adjacent discs and the size of the vertebral discs. These priors allow it to be robust enough to handle cases where instrumentation results in missing discs or susceptibility artifacts. *(See the figure below.)*

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/registration_to_template/instrumentation-missing-discs.png
   :align: center
   :figwidth: 400px

   ``sct_label_vertebrae`` is able to label vertebral levels despite missing discs due to instrumentation.

To apply this labeling algorithm, we use the following command:

.. code:: sh

   sct_label_vertebrae -i t2.nii.gz -s t2_seg.nii.gz -c t2 -qc ~/qc_singleSubj

   # Input arguments:
   #   - i: Input image
   #   - s: Segmented spinal cord corresponding to the input image
   #   - c: Contrast of the input image
   #   - qc: Directory for Quality Control reporting. QC reports allow us to evaluate the results slice-by-slice.

   # Output files/folders:
   #   - t2_seg_labeled.nii.gz: Image containing the labeled spinal cord. Each voxel of the segmented spinal cord is
   #                            labeled with a vertebral level as though the vertebrae were projected onto the spinal
   #                            cord. The convention for label values is C3-->3, C4-->4, etc.
   #   - t2_seg_labeled_discs.nii.gz: Image containing single-voxel intervertebral disc labels (without the segmented
   #                                  spinal cord). Each label is centered within the disc. The convention for label
   #                                  values is C2/C3-->3, C3/C4-->4, etc. This file also contains additional labels
   #                                  (such as the pontomedullary junction and groove), but these are not yet used.
   #   - straight_ref.nii.gz: The straightened input image produced by the intermediate straightening step. Can be
   #                          re-used by other SCT functions that need a straight reference space.
   #   - warp_curve2straight.nii.gz: The 4D warping field that defines the transform from the original curved
   #                                 anatomical image to the straightened image.
   #   - warp_straight2curve.nii.gz: The 4D warping field that defines the inverse transform from the straightened
   #                                 anatomical image back to the original curved image.
   #   - straightening.cache: If sct_label_vertebrae is run another time, the presence of this file (plus
   #                          straight_ref.nii.gz and the two warping fields) will cause the straightening step to be
   #                          skipped, thus saving processing time.

The most relevant output files are ``t2_seg_labeled.nii.gz`` and ``t2_seg_labeled_discs.nii.gz``. Either of them can be used for the template registration and/or for computing metrics along the cord.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/registration_to_template/io-sct_label_vertebrae.png
   :align: center
   :figwidth: 65%

   Input/output images for ``sct_label_vertebrae``.

.. _choosing-labels:

Choosing which labels to use for registration
=============================================

There are several options available to you when choosing the number of labels used for registration:

* **3+ labels**: This is the most accurate method for matching the vertebral levels of the anatomical image to the levels of the template. However, there is a key downside to this approach: Because more than two points must be matched, the level-matching transformation cannot be affine. As a result, the output warping field will be undefined for regions above the top label and below the bottom label.
* **2 labels:** In practice, the difference in accuracy between using 3+ labels and 2 labels is often negligible. Using 2 labels also has the added benefit of allowing for an affine level-matching transformation, which means the template-to-image warping field will be defined for the entire image. For these reasons, we strongly recommend starting with 2 labels for your registration.
* **1 label:** If your image covers only 1 vertebrae, you can still provide a single label. Note that the transformation in this case will be limited to a Z-axis translation, as an affine transformation can't be determined for a single point.

As starting with 2 labels is recommended, you will need to extract them from the labels that were automatically generated in the previous step. To discard the extra vertebral levels, we use ``sct_label_utils`` to create a new label image containing only 2 of the labels. These points are used to match the levels of the subject to the levels of the template, and correspond to the top and bottom vertebrae we wish to use for image registration.

.. code:: sh

   sct_label_utils -i t2_seg_labeled.nii.gz -vert-body 3,9 -o t2_labels_vert.nii.gz

   # Input arguments:
   #   - i: Input image containing a spinal cord labeled with vertebral levels
   #   - vert-body: The top and bottom vertebral levels to create new point labels for. Choose labels based on
   #                your region of interest. For example, here we have chosen '3,9', which corresponds to C3 and T1.
   #   - o: Output filename

   # Output files/folders:
   #   - t2_labels_vert.nii.gz: Image containing the 2 single-voxel vertebral labels

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/registration_to_template/io-sct_label_utils.png
   :align: center
   :figwidth: 65%

   Input/output images for ``sct_label_utils``.

Manual labeling
===============

If the fully automated labeling approach fails for any of your images, you can also manually perform some or all of the steps using ``sct_label_utils -create-viewer``. This tool lets you select labels using a GUI coordinate picker. There are two main approaches you can take:

   * **Manual C2-C3 labeling**: Manually labeling the C2-C3 disc can help initialize the automated disc detection. You would label the posterior tip of the C2-C3 disc using ``sct_label_utils``, then provide the resulting label image to ``sct_label_vertebrae`` with the ``-initlabel`` argument. This will skip the automatic C2-C3 detection, but leave the rest of the automated steps.
   * **Fully manual labeling**: In this case, you bypass the automatic labeling of ``sct_label_vertebrae`` and manually select 1, 2, or more labels according to the recommendations in :ref:`choosing-labels`.

.. note::

   For manual labeling, consider labeling inteverbral discs as opposed to vertebral bodies, as it is often easier to accurately select the posterior tip of the disc with a mouse pointer.

----------


.. _registration-section:

Step 3: Registering the anatomical image to the PAM50 template
**************************************************************

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/registration_to_template/registration-pipeline-3.png
   :align: center
   :figwidth: 100%

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/registration_to_template/thin-plate-straightening.png
   :align: right
   :figwidth: 300px

   Spinal cord straighting using thin-plate spline interpolation.

Now that we have the labeled spinal cord, we can register the anatomical image to the template. The steps of the registration algorithm are as follows:

1. **Straightening:** The straightening works by finding, for each point along the spinal cord, the mathematical transformation to go from a curved centerline to a straight centerline.

   * The straightening algorithm computes the orthogonal plane at each point along the centerline, then constructs a straight space in the output using thin-plate spline interpolation. This allows the inner geometry of the cord to be preserved.
   * The straightening algorithm outputs a forward and a backward warping field (ITK-compatible), which can be concatenated with subsequent transformations, as will be seen later.

2. **Vertebrae-matching transformation**: Once straightened, the next step involves a transformation to match the vertebral levels of the subject to that of the template. If 2 labels are provided, this transformation will be affine; if 3+ labels are provided, this transformation will be non-affine. (Note: This step focuses only on matching the coordinates of the labels, and does not consider the shape of the spinal cord, which is handled by the next step.)
3. **Shape-matching transformation**: A multi-step nonrigid deformation is estimated to match the subject’s cord shape to the template. By default, two steps are used: the first handles large deformations, while the second applies fine adjustments.

.. important::

   SCT provides many additional nonrigid deformation algorithms beyond the default configuration. You can visit the :ref:`customizing-registration-section` page to learn how to optimize the registration procedure for your particular contrast, resolution, and spinal cord geometry.

To apply the registration algorithm, the following command is used:

.. code:: sh

   sct_register_to_template -i t2.nii.gz -s t2_seg.nii.gz -l t2_labels_vert.nii.gz -c t2 -qc ~/qc_singleSubj

   # Input arguments:
   #   - i: Input image
   #   - s: Segmented spinal cord corresponding to the input image
   #   - l: One or two labels located at the center of the spinal cord, on the mid-vertebral slice
   #   - c: Contrast of the image. Specifying this determines which image from the template will be used.
   #        (e.g. t2 --> PAM50_t2.nii.gz)
   #   - qc: Directory for Quality Control reporting. QC reports allow us to evaluate the results slice-by-slice.

   # Output files/folders:
   #   - anat2template.nii.gz: The anatomical subject image (in this case, t2.nii.gz) warped to the template space.
   #   - template2anat.nii.gz: The template image (in this case, PAM50_t2.nii.gz) warped to the anatomical subject
   #                           space.
   #   - warp_anat2template.nii.gz: The 4D warping field that defines the transform from the anatomical image to the
   #                                template image.
   #   - warp_template2anat.nii.gz: The 4D warping field that defines the inverse transform from the template image to
   #                                the anatomical image.

The most relevant of the output files is ``warp_template2anat.nii.gz``, which will be used to transform the unbiased PAM50 template into the subject space (i.e. to match the ``t2.nii.gz`` anatomical image).

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/registration_to_template/io-sct_register_to_template.png
   :align: center
   :figwidth: 65%

   Input/output images for ``sct_register_to_template``.


----------


.. _transforming-template-section:

Step 4: Transforming template objects into the subject space
************************************************************

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/registration_to_template/registration-pipeline-4.png
   :align: center
   :figwidth: 100%

Once the transformations are estimated, we can apply the resulting warping field to the template to bring it into to the subject’s native space.

.. code:: sh

   sct_warp_template -d t2.nii.gz -w warp_template2anat.nii.gz -a 0 -qc ~/qc_singleSubj

   # Input arguments:
   #   - d: Destination image the template will be warped to.
   #   - w: Warping field (template space to anatomical space).
   #   - a: Whether or not to also warp the white matter atlas. (If '-a 1' is specified, 'label/atlas' will also be output.)
   #   - qc: Directory for Quality Control reporting. QC reports allow us to evaluate the results slice-by-slice.

   # Output:
   #   - label/template/: This directory contains the entirety of the PAM50 template, transformed into the subject
   #                      space (i.e. the t2.nii.gz anatomical image).

The output directory (``label/template``) contains 15 template objects, which can then be used to compute metrics for different regions of the spinal cord. An in-depth description of the template objects can be found on the :ref:`pam50` page.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/registration_to_template/io-sct_warp_template.png
   :align: center
   :figwidth: 65%

   Input/output images for ``sct_warp_template``.

----------

Next: Computing metrics
***********************

:ref:`compute-metrics-section` is a follow-on tutorial for using the warped template to perform quantitative analysis.