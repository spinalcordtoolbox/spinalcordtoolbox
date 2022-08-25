#!/usr/bin/env python
#########################################################################################
#
# Perform various types of processing from the spinal cord segmentation (e.g. extract centerline, compute CSA, etc.).
# (extract_centerline) extract the spinal cord centerline from the segmentation. Output file is an image in the same
# space as the segmentation.
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Benjamin De Leener, Julien Touati, Gabriel Mangeat
# Modified: 2014-07-20 by jcohenadad
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: the import of scipy.misc imsave was moved to the specific cases (orth and ellipse) in order to avoid issue #62. This has to be cleaned in the future.

import sys
import os
import logging
import argparse
from typing import Sequence

import pandas as pd
import numpy as np
from matplotlib.ticker import MaxNLocator

from spinalcordtoolbox.aggregate_slicewise import aggregate_per_slice_or_level, save_as_csv, func_wa, func_std, \
    func_sum, merge_dict, normalize_csa
from spinalcordtoolbox.process_seg import compute_shape
from spinalcordtoolbox.scripts import sct_maths
from spinalcordtoolbox.csa_pmj import get_slices_for_pmj_distance
from spinalcordtoolbox.centerline.core import ParamCenterline
from spinalcordtoolbox.image import add_suffix, splitext
from spinalcordtoolbox.reports.qc import generate_qc
from spinalcordtoolbox.utils.shell import SCTArgumentParser, Metavar, ActionCreateFolder, parse_num_list, display_open
from spinalcordtoolbox.utils.sys import init_sct, set_loglevel, __sct_dir__
from spinalcordtoolbox.utils.fs import get_absolute_path

logger = logging.getLogger(__name__)


class SeparateNormArgs(argparse.Action):
    """Separates predictors from their values and puts the results in a dict"""
    def __call__(self, parser, namespace, values, option_string=None):
        pred = values[::2]
        val = values[1::2]
        if len(pred) != len(val):
            parser.error("Values for normalization need to be specified for each predictor.")
        try:
            data_subject = {p: float(v) for p, v in zip(pred, val)}
        except ValueError as e:
            parser.error(f"Non-numeric value passed to '-normalize': {e}")
        setattr(namespace, self.dest, data_subject)


def get_parser():
    """
    :return: Returns the parser with the command line documentation contained in it.
    """
    # Initialize the parser
    parser = SCTArgumentParser(
        description=(
            "Compute the following morphometric measures based on the spinal cord segmentation:\n"
            "  - area [mm^2]: Cross-sectional area, measured by counting pixels in each slice. Partial volume can be "
            "accounted for by inputing a mask comprising values within [0,1]. Can be normalized when specifying the flag -normalize\n"
            "  - angle_AP, angle_RL: Estimated angle between the cord centerline and the axial slice. This angle is "
            "used to correct for morphometric information.\n"
            "  - diameter_AP, diameter_RL: Finds the major and minor axes of the cord and measure their length.\n"
            "  - eccentricity: Eccentricity of the ellipse that has the same second-moments as the spinal cord. "
            "The eccentricity is the ratio of the focal distance (distance between focal points) over the major axis "
            "length. The value is in the interval [0, 1). When it is 0, the ellipse becomes a circle.\n"
            "  - orientation: angle (in degrees) between the AP axis of the spinal cord and the AP axis of the "
            "image\n"
            "  - solidity: CSA(spinal_cord) / CSA_convex(spinal_cord). If perfect ellipse, it should be one. This "
            "metric is interesting for detecting non-convex shape (e.g., in case of strong compression)\n"
            "  - length: Length of the segmentation, computed by summing the slice thickness (corrected for the "
            "centerline angle at each slice) across the specified superior-inferior region.\n"
            "\n"
            "To select the region to compute metrics over, choose one of the following arguments:\n"
            "   1. '-z': Select axial slices based on slice index.\n"
            "   2. '-pmj' + '-pmj-distance' + '-pmj-extent': Select axial slices based on distance from pontomedullary "
            "junction.\n"
            "      (For options 1 and 2, you can also add '-perslice' to compute metrics for each axial slice, rather "
            "than averaging.)\n"
            "   3. '-vert' + '-vertfile': Select a region based on vertebral labels instead of individual slices.\n"
            "      (For option 3, you can also add '-perlevel' to compute metrics for each vertebral level, rather "
            "than averaging.)"
        )
    )

    mandatory = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatory.add_argument(
        '-i',
        metavar=Metavar.file,
        required=True,
        help="Mask to compute morphometrics from. Could be binary or weighted. E.g., spinal cord segmentation."
             "Example: seg.nii.gz"
    )

    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        "-h",
        "--help",
        action="help",
        help="Show this help message and exit."
    )
    optional.add_argument(
        '-o',
        metavar=Metavar.file,
        default='csa.csv',
        help="Output file name (add extension)."
    )
    optional.add_argument(
        '-append',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1],
        default=0,
        help="Append results as a new line in the output csv file instead of overwriting it."
    )
    optional.add_argument(
        '-z',
        metavar=Metavar.str,
        type=parse_num_list,
        default='',
        help="Slice range to compute the metrics across. Example: 5:23"
    )
    optional.add_argument(
        '-perslice',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1],
        default=0,
        help="Set to 1 to output one metric per slice instead of a single output metric. Please note that when "
             "methods ml or map is used, outputing a single metric per slice and then averaging them all is not the "
             "same as outputting a single metric at once across all slices."
    )
    optional.add_argument(
        '-vert',
        metavar=Metavar.str,
        type=parse_num_list,
        help="Vertebral levels to compute the metrics across. Example: 2:9 for C2 to T2. If you also specify a range of "
             "slices with flag `-z`, the intersection between the specified slices and vertebral levels will be "
             "considered."
    )
    optional.add_argument(
        '-vertfile',
        metavar=Metavar.str,
        default=os.path.join('.', 'label', 'template', 'PAM50_levels.nii.gz'),
        help="Vertebral labeling file. Only use with flag -vert.\n"
        "The input and the vertebral labelling file must in the same voxel coordinate system "
        "and must match the dimensions between each other. "
    )
    optional.add_argument(
        '-perlevel',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1],
        default=0,
        help="Set to 1 to output one metric per vertebral level instead of a single output metric. This flag needs "
             "to be used with flag -vert."
    )
    optional.add_argument(
        '-angle-corr',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1],
        default=1,
        help="Angle correction for computing morphometric measures. When angle correction is used, the cord within "
             "the slice is stretched/expanded by a factor corresponding to the cosine of the angle between the "
             "centerline and the axial plane. If the cord is already quasi-orthogonal to the slab, you can set "
             "-angle-corr to 0."
    )
    optional.add_argument(
        '-centerline-algo',
        choices=['polyfit', 'bspline', 'linear', 'nurbs'],
        default='bspline',
        help="Algorithm for centerline fitting. Only relevant with -angle-corr 1."
    )
    optional.add_argument(
        '-centerline-smooth',
        metavar=Metavar.int,
        type=int,
        default=30,
        help="Degree of smoothing for centerline fitting. Only use with -centerline-algo {bspline, linear}."
    )
    optional.add_argument(
        '-pmj',
        metavar=Metavar.file,
        type=get_absolute_path,
        help="Ponto-Medullary Junction (PMJ) label file. "
             "Example: pmj.nii.gz"
    )
    optional.add_argument(
        '-pmj-distance',
        type=float,
        metavar=Metavar.float,
        help="Distance (mm) from Ponto-Medullary Junction (PMJ) to the center of the mask used to compute morphometric "
             "measures. (To be used with flag '-pmj'.)"
    )
    optional.add_argument(
        '-pmj-extent',
        type=float,
        metavar=Metavar.float,
        default=20.0,
        help="Extent (in mm) for the mask used to compute morphometric measures. Each slice covered by the mask is "
             "included in the calculation. (To be used with flag '-pmj' and '-pmj-distance'.)"
    )
    optional.add_argument(
        '-normalize',
        metavar=Metavar.list,
        action=SeparateNormArgs,
        nargs="+",
        help="Normalize CSA values ('MEAN(area)').\n"
             "Two models are available:\n"
             "    1. sex, brain-volume, thalamus-volume.\n"
             "    2. sex, brain-volume.\n"
             "Specify each value for the subject after the corresponding predictor.\n"
             "Example:\n    -normalize sex 0 brain-volume 960606.0 thalamus-volume 13942.0 \n"
             "*brain-volume and thalamus-volume are in mm^3. For sex, female: 0, male: 1.\n"
             "\n"
             "The models were generated using T1w brain images from 804 healthy (non-pathological) participants "
             "ranging from 48 to 80 years old, taken from the UK Biobank dataset.\n"
             "For more details on the subjects and methods used to create the models, go to: "
             "https://github.com/sct-pipeline/ukbiobank-spinalcord-csa#readme \n"  # TODO add ref of the paper
             "Given the risks and lack of consensus surrounding CSA normalization, we recommend thoroughly reviewing "
             "the literature on this topic before applying this feature to your data.\n"
    )
    optional.add_argument(
        '-qc',
        metavar=Metavar.folder,
        type=os.path.abspath,
        action=ActionCreateFolder,
        help="The path where the quality control generated content will be saved."
             " The QC report is only available for PMJ-based CSA (with flag '-pmj')."
    )
    optional.add_argument(
        '-qc-image',
        metavar=Metavar.str,
        help="Input image to display in QC report. Typically, it would be the "
             "source anatomical image used to generate the spinal cord "
             "segmentation. This flag is mandatory if using flag '-qc'."
    )
    optional.add_argument(
        '-qc-dataset',
        metavar=Metavar.str,
        help="If provided, this string will be mentioned in the QC report as the dataset the process was run on."
    )
    optional.add_argument(
        '-qc-subject',
        metavar=Metavar.str,
        help="If provided, this string will be mentioned in the QC report as the subject the process was run on."
    )
    optional.add_argument(
        '-v',
        metavar=Metavar.int,
        type=int,
        choices=[0, 1, 2],
        default=1,
        # Values [0, 1, 2] map to logging levels [WARNING, INFO, DEBUG], but are also used as "if verbose == #" in API
        help="Verbosity. 0: Display only errors/warnings, 1: Errors/warnings + info messages, 2: Debug mode"
    )

    return parser


def _make_figure(metric, fit_results):
    """
    Make a graph showing CSA and angles per slice.
    :param metric: Dictionary of metrics
    :param fit_results: class centerline.core.FitResults()
    :return: image object
    """
    import tempfile
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    fname_img = tempfile.NamedTemporaryFile().name + '.png'
    z, csa, angle_ap, angle_rl = [], [], [], []
    for key, value in metric.items():
        z.append(key[0])
        csa.append(value['MEAN(area)'])
        angle_ap.append(value['MEAN(angle_AP)'])
        angle_rl.append(value['MEAN(angle_RL)'])

    z_ord = np.argsort(z)
    z, csa, angle_ap, angle_rl = (
        [np.array(x)[z_ord] for x in (z, csa, angle_ap, angle_rl)]
    )

    # Make figure
    fig = Figure(figsize=(8, 7), tight_layout=True)  # 640x700 pix
    FigureCanvas(fig)
    # If -angle-corr was set to 1, fit_results exists and centerline fitting results are displayed
    if fit_results is not None:
        ax = fig.add_subplot(311)
        ax.plot(z, csa, 'k')
        ax.plot(z, csa, 'k.')
        ax.grid(True)
        ax.set_ylabel('CSA [$mm^2$]')
        ax.set_xticklabels([])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax = fig.add_subplot(312)
        ax.grid(True)
        ax.plot(z, angle_ap, 'b', label='_nolegend_')
        ax.plot(z, angle_ap, 'b.')
        ax.plot(z, angle_rl, 'r', label='_nolegend_')
        ax.plot(z, angle_rl, 'r.')
        ax.legend(['Rotation about AP axis', 'Rotation about RL axis'])
        ax.set_ylabel('Angle [$deg$]')
        ax.set_xticklabels([])
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax = fig.add_subplot(313)
        ax.grid(True)
        # find a way to condense the following lines
        zmean_list, xmean_list, xfit_list, ymean_list, yfit_list, zref_list = [], [], [], [], [], []
        for i, value in enumerate(fit_results.data.zref):
            if value in z:
                zmean_list.append(fit_results.data.zmean[i])
                xmean_list.append(fit_results.data.xmean[i])
                xfit_list.append(fit_results.data.xfit[i])
                ymean_list.append(fit_results.data.ymean[i])
                yfit_list.append(fit_results.data.yfit[i])
                zref_list.append(fit_results.data.zref[i])
        ax.plot(zmean_list, xmean_list, 'b.', label='_nolegend_')
        ax.plot(zref_list, xfit_list, 'b')
        ax.plot(zmean_list, ymean_list, 'r.', label='_nolegend_')
        ax.plot(zref_list, yfit_list, 'r')
        ax.legend(['Fitted (RL)', 'Fitted (AP)'])
        ax.set_ylabel('Centerline [$vox$]')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        ax = fig.add_subplot(111)
        ax.plot(z, csa, 'k')
        ax.plot(z, csa, 'k.')
        ax.grid(True)
        ax.set_ylabel('CSA [$mm^2$]')

    ax.set_xlabel('Slice (Inferior-Superior direction)')
    fig.savefig(fname_img)

    return fname_img


def main(argv: Sequence[str]):
    parser = get_parser()
    arguments = parser.parse_args(argv)
    verbose = arguments.v
    set_loglevel(verbose=verbose)

    # Initialization
    group_funcs = (('MEAN', func_wa), ('STD', func_std))  # functions to perform when aggregating metrics along S-I

    fname_segmentation = get_absolute_path(arguments.i)

    file_out = os.path.abspath(arguments.o)
    append = bool(arguments.append)
    if arguments.vert is not None:
        levels = arguments.vert
        fname_vert_level = arguments.vertfile
    else:
        levels = []
        fname_vert_level = None
    perlevel = bool(arguments.perlevel)
    slices = arguments.z
    perslice = bool(arguments.perslice)
    angle_correction = bool(arguments.angle_corr)
    param_centerline = ParamCenterline(
        algo_fitting=arguments.centerline_algo,
        smooth=arguments.centerline_smooth,
        minmax=True)
    fname_pmj = arguments.pmj
    distance_pmj = arguments.pmj_distance
    extent_pmj = arguments.pmj_extent
    path_qc = arguments.qc
    qc_dataset = arguments.qc_dataset
    qc_subject = arguments.qc_subject

    if distance_pmj is not None and fname_pmj is None:
        parser.error("Option '-pmj-distance' requires option '-pmj'.")
    if fname_pmj is not None and distance_pmj is None and not perslice:
        parser.error("Option '-pmj' requires option '-pmj-distance' or '-perslice 1'.")

    # update fields
    metrics_agg = {}

    metrics, fit_results = compute_shape(fname_segmentation,
                                         angle_correction=angle_correction,
                                         param_centerline=param_centerline,
                                         verbose=verbose)
    if fname_pmj is not None:
        im_ctl, mask, slices, centerline, length_from_pmj = get_slices_for_pmj_distance(fname_segmentation, fname_pmj,
                                                                                        distance_pmj, extent_pmj,
                                                                                        param_centerline=param_centerline, perslice=perslice,
                                                                                        verbose=verbose)

        # Save array of the centerline in a .csv file if verbose == 2
        if verbose == 2:
            fname_ctl_csv, _ = splitext(add_suffix(arguments.i, '_centerline_extrapolated'))
            np.savetxt(fname_ctl_csv + '.csv', centerline, delimiter=",")
    else:
        length_from_pmj = None
    for key in metrics:
        if key == 'length':
            # For computing cord length, slice-wise length needs to be summed across slices
            metrics_agg[key] = aggregate_per_slice_or_level(metrics[key], slices=slices,
                                                            levels=levels,
                                                            distance_pmj=distance_pmj, perslice=perslice,
                                                            perlevel=perlevel, fname_vert_level=fname_vert_level,
                                                            group_funcs=(('SUM', func_sum),), length_pmj=length_from_pmj)
        else:
            # For other metrics, we compute the average and standard deviation across slices
            metrics_agg[key] = aggregate_per_slice_or_level(metrics[key], slices=slices,
                                                            levels=levels,
                                                            distance_pmj=distance_pmj, perslice=perslice,
                                                            perlevel=perlevel, fname_vert_level=fname_vert_level,
                                                            group_funcs=group_funcs, length_pmj=length_from_pmj)
    metrics_agg_merged = merge_dict(metrics_agg)
    # Normalize CSA values (MEAN(area))
    if arguments.normalize is not None:
        data_subject = pd.DataFrame([arguments.normalize])
        path_model = os.path.join(__sct_dir__, 'data', 'csa_normalization_models',
                                  '_'.join(sorted(data_subject.columns)) + '.csv')
        if not os.path.isfile(path_model):
            parser.error('Invalid choice of predictors in -normalize. Please specify sex and brain-volume or sex, brain-volume and thalamus-volume.')
        # Get normalization model
        # Models are generated with https://github.com/sct-pipeline/ukbiobank-spinalcord-csa/blob/master/pipeline_ukbiobank/cli/compute_stats.py
        # TODO update link with release tag.
        data_predictors = pd.read_csv(path_model, index_col=0)
        # Add interaction term
        data_subject['inter-BV_sex'] = data_subject['brain-volume']*data_subject['sex']
        for line in metrics_agg_merged.values():
            line['MEAN(area)'] = normalize_csa(line['MEAN(area)'], data_predictors, data_subject)

    save_as_csv(metrics_agg_merged, file_out, fname_in=fname_segmentation, append=append)
    # QC report (only for PMJ-based CSA)
    if path_qc is not None:
        if fname_pmj is not None:
            if arguments.qc_image is not None:
                fname_mask_out = add_suffix(arguments.i, '_mask_csa')
                fname_ctl = add_suffix(arguments.i, '_centerline_extrapolated')
                fname_ctl_smooth = add_suffix(fname_ctl, '_smooth')
                if verbose != 2:
                    from spinalcordtoolbox.utils.fs import tmp_create
                    path_tmp = tmp_create()
                    fname_mask_out = os.path.join(path_tmp, fname_mask_out)
                    fname_ctl = os.path.join(path_tmp, fname_ctl)
                    fname_ctl_smooth = os.path.join(path_tmp, fname_ctl_smooth)
                # Save mask
                mask.save(fname_mask_out)
                # Save extrapolated centerline
                im_ctl.save(fname_ctl)
                # Generated centerline smoothed in RL direction for visualization (and QC report)
                sct_maths.main(['-i', fname_ctl, '-smooth', '10,1,1', '-o', fname_ctl_smooth])

                generate_qc(fname_in1=get_absolute_path(arguments.qc_image),
                            # NB: For this QC figure, the centerline has to be first in the list in order for the centerline
                            # to be properly layered underneath the PMJ + mask. However, Sagittal.get_center_spit
                            # is called during QC, and it uses `fname_seg[-1]` to center the slices. `fname_mask_out`
                            # doesn't work for this, so we have to repeat `fname_ctl_smooth` at the end of the list.
                            fname_seg=[fname_ctl_smooth, fname_pmj, fname_mask_out, fname_ctl_smooth],
                            args=sys.argv[1:],
                            path_qc=path_qc,
                            dataset=qc_dataset,
                            subject=qc_subject,
                            process='sct_process_segmentation')
            else:
                parser.error('-qc-image is required to display QC report.')
        else:
            logger.warning('QC report only available for PMJ-based CSA. QC report not generated.')

    display_open(file_out)


if __name__ == "__main__":
    init_sct()
    main(sys.argv[1:])
