Pottslab - A Matlab toolbox for jump-sparse recovery using Potts functionals

Authors:
    The toolbox was written by 
        M. Storath  (Ecole Polytechnique Federale de Lausanne)
    The algorithms were developed in collaboration with
        L. Demaret  (Helmholtz Center Munich)
        A. Weinmann (Helmholtz Center Munich)

Description:
    Pottslab is a Matlab toolbox for the reconstruction of 
    jump-sparse and images using Potts functionals.
    Applications include denoising of piecewise constant signals and 
    image segmentation.

System Requirements:
    Some functions require the Image Processing Toolbox
    Some functions may require Matlab 2014a

Quickstart:
    - Run the script "installPottslab.m", it should set all necessary paths
    - Run a demo from the Demos folder

Troubleshooting:
   * Problem: OutOfMemoryException
   * Solution: Increase Java heap space in the Matlab preferences

   * Problem: Undefined variable "pottslab" or class "pottslab.JavaTools.minL2Potts"
   * Solution: 
        - Run setPLJavaPath.m
        - Maybe you need to install Java 1.7 (see e.g. http://undocumentedmatlab.com/blog/using-java-7-in-matlab-r2013a-and-earlier)

Web:
    http://pottslab.de

Terms of use:
  * Pottslab is under the GNU-GPL3 (see file License.txt).
  * Please cite our corresponding publications (see file HowToCite.txt).

Development platform:
    This software was developed and tested under Matlab 2014a and Java 1.7 
    on Mac OS 10.9.

Comments and suggestions:
    martin.storath@epfl.ch



  