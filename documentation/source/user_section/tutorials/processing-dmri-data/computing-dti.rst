Computing DTI for motion corrected dMRI data
############################################

Here, we compute the diffusion tensor. SCT relies on the excellent dipy library for computing dMRI metrics `[Garyfallidis et al., Front Neuroinform 2014] <https://pubmed.ncbi.nlm.nih.gov/24600385/>`_.

.. code::

   sct_dmri_compute_dti -i dmri_crop_moco.nii.gz -bval bvals.txt -bvec bvecs.txt

:Input arguments:
   - ``-i`` : The input dMRI image.
   - ``-bval`` : A text file containing a b-value for each volume in the dMRI image, indicating the diffusion weightings for each of the volumes in the dMRI image.
   - ``-bvec`` : A text file with three lines, each containing a value for each volume in the input image. Together, the the three sets of values represent the ``(x, y, z)`` coordinates of the b-vectors, which indicate the direction of the diffusion encoding for each volume of the dMRI image.

.. note::

   You can also supply the ``-method restore`` option to estimate the tensors using **"RESTORE: robust estimation of tensors by outlier rejection"** `[Chang, Magn Reson Med 2005] <https://pubmed.ncbi.nlm.nih.gov/15844157/>`_.

:Output files/folders:
   - ``dti_FA.nii.gz`` : Fractional anisotropy (FA) diffusion tensor image.
   - ``dti_AD.nii.gz`` : Axial diffusivity (AD) diffusion tensor image.
   - ``dti_MD.nii.gz`` : Mean diffusivity (MD) diffusion tensor image.
   - ``dti_RD.nii.gz`` : Radial diffusivity (RD) diffusion tensor image.

.. TODO: Why doesn't sct_dmri_compute_dti output "Geodesic anisotropy (GA)"?

.. figure:: https://raw.githubusercontent.com/spinalcordtoolbox/doc-figures/jn/2857-add-remaining-tutorials/processing-dmri-data/io-sct_dmri_compute_dti.png
   :align: center

Now that we have the diffusion tensor images, we can register the dMRI data to the PAM50 template so that we can use the template to extract these metrics from specific regions of the image.
