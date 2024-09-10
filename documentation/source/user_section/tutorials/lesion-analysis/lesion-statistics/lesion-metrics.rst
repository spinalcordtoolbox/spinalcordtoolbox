Compute lesion morphometric measures
####################################

Various morphometrics
---------------------

Here, we will see how to compute various morphometric measures on segmented lesions. For example, the number of lesions, lesion length, lesion volume, etc. The statistics will be output in an Excel (XLS) file.

In the case of multiple lesions, the function assigns an ID value to each lesion (1, 2, 3, etc.) and then outputs morphometric measures for each lesion to an XLS file.

The following morphometric measures are computed:

* ``volume [mm^3]`` : volume of the lesion
* ``length [mm]`` : length of the lesion along the superior-inferior (SI) axis

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/lesion-analysis/intramedullary-lesion-length.png
  :align: center
  :figwidth: 60%

* ``max_equivalent_diameter [mm]`` : maximum diameter of the lesion, when approximating the lesion as a circle in the axial plane
* ``max_axial_damage_ratio []`` : maximum axial damage ratio defined as the ratio of the lesion area divided by the spinal cord area. The ratio is computed in the axial plane for each slice and and the maximum ratio is reported.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/lesion-analysis/axial-damage-ratio.png
  :align: center
  :figwidth: 60%

TODO: Ask Andrew Smith for permission to use the figure

* ``dorsal_bridge_width [mm]`` : dorsal tissue bridges defined as the width of dorsal spared tissue (i.e. towards the posterior direction of the AP axis) at the minimum distance from the intramedullary lesion edge to the boundary between the spinal cord and cerebrospinal fluid
* ``ventral_bridge_width [mm]`` : ventral tissue bridges defined as the width of ventral spared tissue (i.e. towards the anterior direction of the AP axis) at the minimum distance from the intramedullary lesion edge to the boundary between the spinal cord and cerebrospinal fluid

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/master/lesion-analysis/tissue-bridges.png
  :align: center
  :figwidth: 60%

Running ``sct_analyze_lesion``
------------------------------

To compute the statistics, run the following command

.. code:: sh

   sct_analyze_lesion -m t2_lesion_seg.nii.gz -s t2_sc_seg.nii.gz -qc ./qc

:Input arguments:
   - ``-m`` : 3D binary mask of the segmented lesion
   - ``-s`` : 3D binary mask of the segmented spinal cord
   - ``-qc`` : Directory for Quality Control reporting. QC report contains a figure for the tissue bridges

:Output files:
   - ``t2_lesion_analysis.xls`` : XLS file containing the morphometric measures
   - ``t2_lesion_analysis.pkl`` : Python Pickle file containing the morphometric measures
   - ``t2_lesion_label.nii.gz`` : 3D mask of the segmented lesion with lesion IDs (1, 2, 3, etc.)

Details:

* **maximum axial damage ratio:** `Smith, A.C., et al. Spinal Cord (2021) <https://doi.org/10.1038/s41393-020-00561-w>`_
* **tissue bridges:** `Enamundram, N.K.*, Valošek, J.*, et al. arXiv (2024) <https://doi.org/10.48550/arXiv.2407.17265>`_, `Huber, E., et al. Ann Neurol. (2017) <https://doi.org/10.1002/ana.24932>`_, `Pfyffer, D., et al. Lancet Neurol. (2024) <https://doi.org/10.1016/S1474-4422%2824%2900173-X>`_

TODO: check with collaborators what references to include for maximum axial damage ratio and tissue bridges