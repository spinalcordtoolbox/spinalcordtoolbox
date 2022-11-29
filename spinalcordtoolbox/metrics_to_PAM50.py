

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



def interpolate_metrics(PAM50_seg_labeled, metrics_csv, segmentation):
    # read CSV file
    metrics = pd.read_csv(metrics_csv)
    # Get vertebral levels
    levels = np.unique(metrics['VertLevel'])
    
    # get slice thickness of segmentation
    im_seg = Image(segmentation)
    _, _, _, _, px, py, pz, _ = im_seg.dim
    # get slice thickness of PAM50
    im_seg_labeled_PAM50 = Image(PAM50_seg_labeled)
    _, _, _, _, px_PAM50, py_PAM50, pz_PAM50, _ = im_seg_labeled_PAM50.dim
    p_ratio = pz/pz_PAM50

    # loop through levels
    for level in levels:
        # put in same resolution
        #pd.loc[]
        # interpolate in the same number of slices
        nb_slices = len(get_slices_from_vertebral_levels(im_seg_labeled_PAM50, level))

    
    # linear interpolation

def main():
    parser = get_parser()
    args = parser.parse_args()
    interpolate_metrics(args.PAM50_seg_labeled, args.metrics_csv, args.segmentation)
if __name__ == '__main__':
    main()    
