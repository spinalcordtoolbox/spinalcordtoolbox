

import argparse
import logging
import numpy as np
import pandas as pd
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.template import get_slices_from_vertebral_levels


logger = logging.getLogger(__name__)

# 1. Detect complete levels
# 2 Interpolation perlevel

def get_parser():
    parser = argparse.ArgumentParser(
        description="Computes statistics about CSA and distances between PMJ, dics and nerve rootlets.")
    parser.add_argument('-segmentation', required=True, type=str,
                        help="image segmentation")
    parser.add_argument('-PAM50-seg-labeled', required=True, type=str,
                        help="image segmentation")
    parser.add_argument('-metrics-csv', required=True, type=str,
                        help="image segmentation")    
    return parser



def interpolate_metrics(PAM50_seg_labeled, metrics_csv, segmentation): # Remove segmentation
    # Check if levels are complete

    # read CSV file
    metrics = pd.read_csv(metrics_csv)
    # Get vertebral levels
    levels = np.unique(metrics['VertLevel'])
    
    # get slice thickness of segmentation
    im_seg = Image(segmentation)
    _, _, _, _, px, py, pz, _ = im_seg.dim # TO REMOVE
    # get slice thickness of PAM50
    im_seg_labeled_PAM50 = Image(PAM50_seg_labeled)
    _, _, _, _, px_PAM50, py_PAM50, pz_PAM50, _ = im_seg_labeled_PAM50.dim
    p_ratio = pz/pz_PAM50  # TO REMOVE
    slices_PAM50_all = []
    levels_PAM50_all = []
    df_metrics_PAM50_space = pd.DataFrame(columns=metrics.columns)
    # loop through levels
    for level in levels:
        # interpolate in the same number of slices
        slices_PAM50 = get_slices_from_vertebral_levels(im_seg_labeled_PAM50, level)
        slices_PAM50.reverse()
        nb_slices = len(slices_PAM50)
        # Add slice number of PAM50 space and vertebral level
        levels_PAM50_all.extend(np.repeat(level, nb_slices))
        slices_PAM50_all.extend(slices_PAM50)

        x_PAM50 = np.arange(0, nb_slices, 1)
        x = np.linspace(0, nb_slices - 1, len(metrics.loc[metrics['VertLevel']==level, 'MEAN(area)']))
        # Only keep metrics with MEAN
        columns = [i for i in metrics.columns.to_list() if 'MEAN' in i]
        for metric_name in columns:
            metric_values = metrics.loc[metrics['VertLevel']==level, metric_name].tolist()
            metrics_PAM50_space = np.interp(x_PAM50, x, metric_values)
            #csa_PAM50_space.append(metrics_PAM50_space)
    # linear interpolation

    df_metrics_PAM50_space['Slice (I->S)'] = slices_PAM50_all
    df_metrics_PAM50_space['VertLevel'] = levels_PAM50_all
    print(df_metrics_PAM50_space[['Slice (I->S)','VertLevel']])
    # TODO: unit test : check that the proper column is output (MEAN), number of columns with MEANS
    # TODO: put metrics in Metrics() class type
    # loop through keys and call aggregate_per_slice_or_level, merge_dict(metrics_agg) and save as csv

def main():
    parser = get_parser()
    args = parser.parse_args()
    #metrics_csv = "/mnt/c/Users/sb199/Projet3_data/DCM_norm_PAM50/sub-amu01/csa.csv"
    #PAM50_seg_labeled = "/mnt/c/Users/sb199/spinalcordtoolbox/data/PAM50/template/PAM50_levels.nii.gz"
    #segmentation = "/mnt/c/Users/sb199/Projet3_data/DCM_norm_PAM50/sub-amu01/sub-amu01_T1w_seg.nii.gz"
    interpolate_metrics(args.PAM50_seg_labeled, args.metrics_csv, args.segmentation)
    #interpolate_metrics(PAM50_seg_labeled, metrics_csv, segmentation)

if __name__ == '__main__':
    main()    
