Manual labeling
###############

If the fully automated labeling approach fails for any of your images, you can also manually perform some or all of the steps using ``sct_label_utils -create-viewer``. This tool lets you select labels using a GUI coordinate picker. There are two main approaches you can take:

   * **Manual C2-C3 labeling**: Manually labeling the C2-C3 disc can help initialize the automated disc detection. You would label the posterior tip of the C2-C3 disc using ``sct_label_utils``, then provide the resulting label image to ``sct_label_vertebrae`` with the ``-initlabel`` argument. This will skip the automatic C2-C3 detection, but leave the rest of the automated steps.
   * **Fully manual labeling**: In this case, you bypass the automatic labeling of ``sct_label_vertebrae`` and manually select 1, 2, or more labels according to the recommendations in :ref:`choosing-labels`.

.. note::

   For manual labeling, consider labeling inteverbral discs as opposed to vertebral bodies, as it is often easier to accurately select the posterior tip of the disc with a mouse pointer.