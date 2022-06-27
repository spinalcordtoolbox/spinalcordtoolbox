#!/usr/bin/env python3
# *_* coding: utf-8 *_*

"""
    Partial reproduction of the 'fslhd' command-line utility, rewritten in Python.

    For full attribution, the original header of the 'fslhd.cc' source file is
    reproduced in full below:

//     fslhd.cc - show image header
//     Steve Smith, Mark Jenkinson and Matthew Webster, FMRIB Image Analysis Group
//     Copyright (C) 2000-2019 University of Oxford
/*  Part of FSL - FMRIB's Software Library
    http://www.fmrib.ox.ac.uk/fsl
    fsl@fmrib.ox.ac.uk

    Developed at FMRIB (Oxford Centre for Functional Magnetic Resonance
    Imaging of the Brain), Department of Clinical Neurology, Oxford
    University, Oxford, UK


    LICENCE

    FMRIB Software Library, Release 6.0 (c) 2018, The University of
    Oxford (the "Software")

    The Software remains the property of the Oxford University Innovation
    ("the University").

    The Software is distributed "AS IS" under this Licence solely for
    non-commercial use in the hope that it will be useful, but in order
    that the University as a charitable foundation protects its assets for
    the benefit of its educational and research purposes, the University
    makes clear that no condition is made or to be implied, nor is any
    warranty given or to be implied, as to the accuracy of the Software,
    or that it will be suitable for any particular purpose or for use
    under any specific conditions. Furthermore, the University disclaims
    all responsibility for the use which is made of the Software. It
    further disclaims any liability for the outcomes arising from using
    the Software.

    The Licensee agrees to indemnify the University and hold the
    University harmless from and against any and all claims, damages and
    liabilities asserted by third parties (including claims for
    negligence) which arise directly or indirectly from the use of the
    Software or the sale of any products based on the Software.

    No part of the Software may be reproduced, modified, transmitted or
    transferred in any form or by any means, electronic or mechanical,
    without the express permission of the University. The permission of
    the University is not required if the said reproduction, modification,
    transmission or transference is done without financial return, the
    conditions of this Licence are imposed upon the receiver of the
    product, and all original and amended source code is included in any
    transmitted product. You may be held legally responsible for any
    copyright infringement that is caused or encouraged by your failure to
    abide by these terms and conditions.

    You are not permitted under this Licence to use this Software
    commercially. Use for which any financial return is received shall be
    defined as commercial use, and includes (1) integration of all or part
    of the source code or the Software into a product for sale or license
    by or on behalf of Licensee to third parties or (2) use of the
    Software or any derivative of it for research with the final aim of
    developing software products for sale or license to a third party or
    (3) use of the Software or any derivative of it for research with the
    final aim of developing non-software products for sale or license to a
    third party, or (4) use of the Software to provide any service to an
    external organisation for which payment is received. If you are
    interested in using the Software commercially, please contact Oxford
    University Innovation ("OUI"), the technology transfer company of the
    University, to negotiate a licence. Contact details are:
    fsl@innovation.ox.ac.uk quoting Reference Project 9564, FSL.*/
"""

import nibabel as nib

INTENT_STRINGS = {
    0: "Unknown",
    2: "Correlation statistic",
    3: "T-statistic",
    4: "F-statistic",
    5: "Z-score",
    6: "Chi-squared distribution",
    7: "Beta distribution",
    8: "Binomial distribution",
    9: "Gamma distribution",
    10: "Poisson distribution",
    11: "Normal distribution",
    12: "F-statistic noncentral",
    13: "Chi-squared noncentral",
    14: "Logistic distribution",
    15: "Laplace distribution",
    16: "Uniform distribition",
    17: "T-statistic noncentral",
    18: "Weibull distribution",
    19: "Chi distribution",
    20: "Inverse Gaussian distribution",
    21: "Extreme Value distribution",
    22: "P-value",
    23: "Log P-value",
    24: "Log10 P-value",
    1001: "Estimate",
    1002: "Label index",
    1003: "NeuroNames index",
    1004: "General matrix",
    1005: "Symmetric matrix",
    1006: "Displacement vector",
    1007: "Vector",
    1008: "Pointset",
    1009: "Triangle",
    1010: "Quaternion",
    1011: "Dimensionless number",
    2001: "Time series",
    2002: "Node index",
    2003: "RGB vector",
    2004: "RGBA vector",
    2005: "Shape"
}

ORIENTATION_STRINGS = {
    None: "Unknown",
    'R': "Left-to-Right",
    'L': "Right-to-Left",
    'A': "Posterior-to-Anterior",
    'P': "Anterior-to-Posterior",
    'S': "Inferior-to-Superior",
    'I': "Superior-to-Inferior"
}

DATATYPE_STRINGS = {
    0: "UNKNOWN",
    1: "BINARY",
    256: "INT8",
    2: "UINT8",
    4: "INT16",
    512: "UINT16",
    8: "INT32",
    768: "UINT32",
    1024: "INT64",
    1280: "UINT64",
    16: "FLOAT32",
    64: "FLOAT64",
    1536: "FLOAT128",
    32: "COMPLEX64",
    1792: "COMPLEX128",
    2048: "COMPLEX256",
    128: "RGB24",
    2034: "RGBA32"
}

XFORM_STRINGS = {
    0: "Unknown",
    1: "Scanner Anat",
    2: "Aligned Anat",
    3: "Talairach",
    4: "MNI_152",
    5: "Template"
}

UNIT_STRINGS = {
    0: "Unknown",
    1: 'm',
    2: 'mm',
    3: 'um',
    8: 's',
    16: 'ms',
    24: 'us',
    32: 'Hz',
    40: 'ppm',
    48: 'rad/s'
}

SLICE_ORDER_STRINGS = {
    0: "Unknown",
    1: "sequential_increasing",
    2: "sequential_decreasing",
    3: "alternating_increasing",
    4: "alternating_decreasing",
    5: "alternating_increasing_2",
    6: "alternating_decreasing_2",
}


def generate_nifti_fields(header):
    """
    Generate nifti header fields using methods found in fslhd.
    """
    nib_fields = {k: v[()] for k, v in dict(header).items()}
    qform = header.get_qform()
    sform = header.get_sform()
    slope, inter = header.get_slope_inter()

    return {
        'sizeof_hdr': nib_fields['sizeof_hdr'],
        'data_type': DATATYPE_STRINGS[nib_fields['datatype']],
        'dim0': nib_fields['dim'][0],
        'dim1': nib_fields['dim'][1],
        'dim2': nib_fields['dim'][2],
        'dim3': nib_fields['dim'][3],
        'dim4': nib_fields['dim'][4],
        'dim5': nib_fields['dim'][5],
        'dim6': nib_fields['dim'][6],
        'dim7': nib_fields['dim'][7],
        'vox_units': UNIT_STRINGS[nib_fields['xyzt_units'] & 0x07],
        'time_units': UNIT_STRINGS[nib_fields['xyzt_units'] & 0x38],
        'datatype': nib_fields['datatype'],
        'nbyper': nib_fields['bitpix']//8,
        'bitpix': nib_fields['bitpix'],
        # NB: nib.load calls Nifti1Header._chk_qfc(fix=True) internally, which sets pixdim[0] = 1 if the value is wrong.
        # fslhd doesn't apply any fixing, so there could be a discrepancy in what's displayed.
        'pixdim0': f"{nib_fields['pixdim'][0]:.6f}",
        'pixdim1': f"{nib_fields['pixdim'][1]:.6f}",
        'pixdim2': f"{nib_fields['pixdim'][2]:.6f}",
        'pixdim3': f"{nib_fields['pixdim'][3]:.6f}",
        'pixdim4': f"{nib_fields['pixdim'][4]:.6f}",
        'pixdim5': f"{nib_fields['pixdim'][5]:.6f}",
        'pixdim6': f"{nib_fields['pixdim'][6]:.6f}",
        'pixdim7': f"{nib_fields['pixdim'][7]:.6f}",
        # NB: When printing the header as-is, nibabel uses nib_fields['vox_offset'], which == header.pair_vox_offset.
        # But, header.pair_vox_offset doesn't match the output of fslhd, so use header.single_vox_offset instead.
        'vox_offset': header.single_vox_offset,
        'cal_max': f"{nib_fields['cal_max']:.6f}",
        'cal_min': f"{nib_fields['cal_min']:.6f}",
        'scl_slope': f"{slope:.6f}" if isinstance(slope, float) else slope,
        'scl_inter': f"{inter:.6f}" if isinstance(inter, float) else inter,
        'phase_dim': int((nib_fields['dim_info'] >> 2) & 3),
        'freq_dim': int(nib_fields['dim_info'] & 3),
        'slice_dim': int((nib_fields['dim_info'] >> 4) & 3),
        'slice_name': SLICE_ORDER_STRINGS[nib_fields['slice_code']].title(),
        'slice_code': nib_fields['slice_code'],
        'slice_start': nib_fields['slice_start'],
        'slice_end': nib_fields['slice_end'],
        'slice_duration': f"{nib_fields['slice_duration']:.6f}",
        'toffset': f"{nib_fields['toffset']:.6f}",
        'intent': INTENT_STRINGS[nib_fields['intent_code']],
        'intent_code': nib_fields['intent_code'],
        'intent_name': nib_fields['intent_name'].decode('utf-8'),
        'intent_p1': f"{nib_fields['intent_p1']:.6f}",
        'intent_p2': f"{nib_fields['intent_p2']:.6f}",
        'intent_p3': f"{nib_fields['intent_p3']:.6f}",
        'qform_name': XFORM_STRINGS[nib_fields['qform_code']],
        'qform_code': nib_fields['qform_code'],
        'qto_xyz:1': "".join([f"{v:.6f} " for v in qform[0]]),
        'qto_xyz:2': "".join([f"{v:.6f} " for v in qform[1]]),
        'qto_xyz:3': "".join([f"{v:.6f} " for v in qform[2]]),
        'qto_xyz:4': "".join([f"{v:.6f} " for v in qform[3]]),
        'qform_xorient': ORIENTATION_STRINGS[nib.aff2axcodes(qform)[0]],
        'qform_yorient': ORIENTATION_STRINGS[nib.aff2axcodes(qform)[1]],
        'qform_zorient': ORIENTATION_STRINGS[nib.aff2axcodes(qform)[2]],
        'sform_name': XFORM_STRINGS[nib_fields['sform_code']],
        'sform_code': nib_fields['sform_code'],
        'sto_xyz:1': "".join([f"{v:.6f} " for v in sform[0]]),
        'sto_xyz:2': "".join([f"{v:.6f} " for v in sform[1]]),
        'sto_xyz:3': "".join([f"{v:.6f} " for v in sform[2]]),
        'sto_xyz:4': "".join([f"{v:.6f} " for v in sform[3]]),
        'sform_xorient': ORIENTATION_STRINGS[nib.aff2axcodes(sform)[0]],
        'sform_yorient': ORIENTATION_STRINGS[nib.aff2axcodes(sform)[1]],
        'sform_zorient': ORIENTATION_STRINGS[nib.aff2axcodes(sform)[2]],
        'file_type': f"NIFTI-{int(nib_fields['magic'].decode('utf-8')[2])}+",
        'file_code': int(nib_fields['magic'].decode("utf-8")[2]),
        'descrip': nib_fields['descrip'].decode('utf-8'),
        'aux_file': nib_fields['aux_file'].decode('utf-8')
    }
