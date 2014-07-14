
# Spinal Cord Toolbox

Comprehensive and open-source library of analysis tools for MRI of the spinal cord.

- [Installation](https://sourceforge.net/p/spinalcordtoolbox/wiki/installation/)
- [List of tools](http://sourceforge.net/p/spinalcordtoolbox/wiki/tools/)
- [Getting started](https://sourceforge.net/p/spinalcordtoolbox/wiki/get_started/)


## License

The MIT License (MIT)

Copyright (c) 2014 École Polytechnique, Université de Montréal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Changes

1.0.2 (2014-07-13)

- NEW: virtual machine
- BUG: fixed sct_check_dependencies for Linux
- BUG: fix VM failure of sct_register_to_template (issue #41)
- OPT: sct_register_to_template.py now registers the straight spinal cord to the template using sct_register_multimodal.py, which uses the spinal cord segmentation for more accurate results.

1.0.1 (2014-07-03)

- INST: toolbox now requires matplotlib

1.0 (2014-06-15)

- first public release!

0.7 (2014-06-14)

- NEW: dMRI moco
- INST: libraries are now statically compiled
- OPT: propseg: now results are reproducible (i.e. removed pseudo-randomization)

0.6 (2014-06-13)

- Debian + OSX binaries
- BUG: fixed registration2template issue when labels were larger than 9
- BUG: fixed bug on PropSeg when the image contains a lot of null slices
- INST: now installer write on bashrc and links bash_profile to bashrc
- BUG: removed random parts in PropSeg

0.5 (2014-06-03)

- now possible to get both template2EPI and EPI2template warping fields
- fixed major bug in registration (labels were cropped)
- NEW: probabilistic location of spinal levels
- NEW: binaries for Debian/Ubuntu

0.4 (2014-05-28)

- NEW: installer for ANTs (currently only for OSX)
- fixed bugs

0.3 (2014-05-26)

- major changes in sct_register_multimodal
- fixed bugs

0.2 (2014-05-18)

- NEW: nonlocal means denoising filter
- NEW: sct_smooth_spinalcord --> smoothing along centerline
- fixed bugs

0.1 (2014-05-03)

- first beta version!
