Gray matter segmentation algorithm: ``sct_deepseg_gm``
######################################################

For segmenting the gray matter, SCT features the function ``sct_deepseg_gm``, which is based a deep learning architecture trained from 232 subjects (~4000 slices).

* **Algorithm:** Deep learning with dilated convolutions `[Perone et al., Sci Report 2018] <https://www.nature.com/articles/s41598-018-24304-3>`_
* **Pros:** High accuracy, robust to pathologies
* **Cons:** Restricted to T2*-like contrasts (GM bright, WM dark)

``sct_deepseg_gm`` obtained the best Dice score amongst all other methods that participated in the GM challenge `[Prados et al., Neuroimage 2017] <https://pubmed.ncbi.nlm.nih.gov/28286318/>`_.

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/jn/2857-add-remaining-tutorials/gm-wm-segmentation/gm-challenge.png
   :align: center

   Results of the GM Challenge

