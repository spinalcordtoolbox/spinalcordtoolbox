#
# Plot a single subject morphometric metrics in the PAM50 space per slice and vertebral levels
#
# You can use SCT's conda environment to run this script:
#       # Go to the SCT directory
#       cd $SCT_DIR
#       # Activate SCT conda environment
#       source ./python/etc/profile.d/conda.sh
#       conda activate venv_sct
#
# Example usage on a single subject:
#       python generate_figure_PAM50_quadrant_measures.py -file sub-001_T2w_metrics_PAM50.csv
#
# Authors: Jan Valosek
#

import os
import sys
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

METRICS = [
    'MEAN(area_quadrant_anterior_left)',
    'MEAN(area_quadrant_anterior_right)',
    'MEAN(area_quadrant_posterior_left)',
    'MEAN(area_quadrant_posterior_right)',
    'MEAN(area_half_left)',
    'MEAN(area_half_right)',
    'MEAN(area_half_anterior)',
    'MEAN(area_half_posterior)',
    'MEAN(area_total)',
    'MEAN(area)',
]


METRICS_COLORS = {
    'MEAN(area_quadrant_anterior_left)': 'lightblue',
    'MEAN(area_quadrant_anterior_right)': 'blue',
    'MEAN(area_quadrant_posterior_left)': '#FF6666',    # light red
    'MEAN(area_quadrant_posterior_right)': 'red',
    'MEAN(area_half_left)': '#90EE90',  # light green
    'MEAN(area_half_right)': 'green',
    'MEAN(area_half_anterior)': 'dodgerblue',
    'MEAN(area_half_posterior)': '#ADD8E6', # light blue
    'MEAN(area_total)': '#FFD700',  # gold
    'MEAN(area)': '#000000',  # black
}

METRICS_LINESTYLES = {
    'MEAN(area_quadrant_anterior_left)': 'solid',
    'MEAN(area_quadrant_anterior_right)': 'solid',
    'MEAN(area_quadrant_posterior_left)': 'solid',
    'MEAN(area_quadrant_posterior_right)': 'solid',
    'MEAN(area_half_left)': 'solid',
    'MEAN(area_half_right)': 'solid',
    'MEAN(area_half_anterior)': 'solid',
    'MEAN(area_half_posterior)': 'solid',
    'MEAN(area_total)': 'solid',
    'MEAN(area)': 'dashed',
}

METRICS_DTYPE = {
    'MEAN(diameter_AP)': 'float64',
    'MEAN(area)': 'float64',
    'MEAN(diameter_RL)': 'float64',
    'MEAN(eccentricity)': 'float64',
    'MEAN(solidity)': 'float64'
}

LABELS_FONT_SIZE = 14
TICKS_FONT_SIZE = 12


def get_parser():
    parser = argparse.ArgumentParser(
        description="Plot single subject morphometric metrics (multiple sessions) together with normative values computed "
                    "from normative database (spine-generic dataset in PAM50 space) per slice and vertebral levels")
    parser.add_argument('-file', required=True, type=str,
                        help="Path to a CSV file with morphometric metrics in PAM50 space. ")
    parser.add_argument('-path-out', required=False, type=str, default='figures',
                        help="Output directory name. Default: figures.")

    return parser


# Apply smoothing to the metric data
def smooth(y: np.ndarray, box_pts: int) -> np.ndarray:
    """
    Smooths a 1D array using a simple moving average (box filter).
    Inspired by: https://github.com/sct-pipeline/rootlets-informed-reg2template/blob/main/csa_analysis.py#L199
    Args:
        y (np.ndarray): Input 1D array to be smoothed.
        box_pts (int): Number of points in the moving average window. Needs to be >= 1.
    Returns:
        np.ndarray: Smoothed array of the same length as the input.
    """
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def load_single_subject_data(path_single_subject, df_spine_generic_min, df_spine_generic_max):
    """
    Load single subject data
    :param path_single_subject: path to single subject CSV file (from session1 or session2)
    :param df_spine_generic_min: minimum slice number from spine-generic dataset
    :param df_spine_generic_max: maximum slice number from spine-generic dataset
    :return:
    """
    df_single_subject = pd.read_csv(path_single_subject, dtype=METRICS_DTYPE)
    # Compute compression ratio (CR) as MEAN(diameter_AP) / MEAN(diameter_RL)
    df_single_subject['MEAN(compression_ratio)'] = df_single_subject['MEAN(diameter_AP)'] / \
                                                   df_single_subject['MEAN(diameter_RL)']
    # Multiply solidity by 100 to get percentage (sct_process_segmentation computes solidity in the interval 0-1)
    df_single_subject['MEAN(solidity)'] = df_single_subject['MEAN(solidity)'] * 100

    # Keep only slices from C1 to Th1 to match the slices of the spine-generic normative values
    df_single_subject = df_single_subject[(df_single_subject['Slice (I->S)'] >= df_spine_generic_min) &
                                          (df_single_subject['Slice (I->S)'] <= df_spine_generic_max)]

    return df_single_subject

def get_vert_indices(df):
    """
    Get indices of slices corresponding to mid-vertebrae
    Args:
        df (pd.dataFrame): dataframe with CSA values
    Returns:
        vert (pd.Series): vertebrae levels across slices
        ind_vert (np.array): indices of slices corresponding to the beginning of each level (=intervertebral disc)
        ind_vert_mid (np.array): indices of slices corresponding to mid-levels
    """
    # Get unique participant IDs
    subjects = df['Filename'].unique()
    # Get vert levels for one certain subject
    vert = df[df['Filename'] == subjects[0]]['VertLevel']
    # Get indexes of where array changes value
    ind_vert = vert.diff()[vert.diff() != 0].index.values
    # Get the beginning of C1
    ind_vert = np.append(ind_vert, vert.index.values[-1])
    ind_vert_mid = []
    # Get indexes of mid-vertebrae
    for i in range(len(ind_vert)-1):
        ind_vert_mid.append(int(ind_vert[i:i+2].mean()))

    return vert, ind_vert, ind_vert_mid


def create_lineplot(df, figure_path):
    """
    Create lineplot for individual metrics per vertebral levels.
    Args:
        df (pd.DataFrame): dataframe with single subject values
        figure_path (str): path to save the figure
    """
    mpl.rcParams['font.family'] = 'Arial'

    # Plot figures
    fig, axes = plt.subplots(1, 1, figsize=(6, 7))

    for index, metric in enumerate(METRICS):

        # Smooth the data to improve visualization
        df[metric] = smooth(df[metric].values, 5)
        sns.lineplot(ax=axes, x="Slice (I->S)", y=metric,
                     data=df, linewidth=1, color=METRICS_COLORS[metric],
                     linestyle=METRICS_LINESTYLES[metric],
                     label=metric)

    # Tweak y-axis limits
    # ymin, ymax = METRICS_YLIMITS[metric]
    # axes.set_ylim(ymin, ymax)
    # Remove first and last 4 slices from the x-axis to remove smoothing artifacts
    axes.set_xlim(df['Slice (I->S)'].iloc[4], df['Slice (I->S)'].iloc[-4])

    # axes.set_ylabel(METRIC_TO_AXIS[metric], fontsize=LABELS_FONT_SIZE)
    # axes.set_xlabel('PAM50 Axial Slice #', fontsize=LABELS_FONT_SIZE)
    # Remove xticks to hide PAM50 Axial Slice numbers
    axes.set_xticks([])
    axes.tick_params(axis='both', which='major', labelsize=TICKS_FONT_SIZE)
    axes.spines['right'].set_visible(False)
    axes.spines['left'].set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.spines['bottom'].set_visible(True)

    # Get ymin and ymax for the y-axis
    ymin, ymax = axes.get_ylim()

    vert, ind_vert, ind_vert_mid = get_vert_indices(df)
    for idx, x in enumerate(ind_vert[1:-1]):
        axes.axvline(df.loc[x, 'Slice (I->S)'], color='black', linestyle='--', alpha=0.3, zorder=0)
    for idx, x in enumerate(ind_vert_mid, 0):
        level = f'T{vert[x] - 7}' if vert[x] > 7 else f'C{vert[x]}'
        axes.text(df.loc[ind_vert_mid[idx], 'Slice (I->S)'],
                        ymin - (ymax-ymin)*0.05, level, horizontalalignment='center',
                        verticalalignment='bottom', color='black', fontsize=TICKS_FONT_SIZE)

    axes.invert_xaxis()
    axes.yaxis.grid(True)
    axes.set_axisbelow(True)
    # Remove x- and y-label
    axes.set_xlabel('')  # Remove x-axis label ('Slice (I->S)')
    axes.set_ylabel('Area [mm²]')

    # Decrease the font size of the legend
    axes.legend(loc='upper right', fontsize=TICKS_FONT_SIZE-6)

    # plt.suptitle(f"Morphometric measures for {fname_str.replace('_', ' ')} in PAM50 template space",
    #              fontweight='bold', fontsize=LABELS_FONT_SIZE, y=0.92)
    plt.savefig(figure_path, dpi=300, bbox_inches='tight')
    print(f'Figure saved: {figure_path}')


def main():
    parser = get_parser()
    args = parser.parse_args()
    file = args.file
    path_out = os.path.abspath(args.path_out)

    if not os.path.exists(path_out):
        os.makedirs(path_out)

    # Create a list of dataframes for each session file
    df = load_single_subject_data(file, 700, 980)
    if df.empty:
        print('WARNING: No slices found in the range C1-Th1 in the single subject data. Exiting...')
        sys.exit(1)

    # Sum 'MEAN(area_quadrant_anterior_left)' and 'MEAN(area_quadrant_posterior_left)' into left side
    # and 'MEAN(area_quadrant_anterior_right)' and 'MEAN(area_quadrant_posterior_right)' into right side
    df['MEAN(area_half_left)'] = df['MEAN(area_quadrant_anterior_left)'] + df['MEAN(area_quadrant_posterior_left)']
    df['MEAN(area_half_right)'] = df['MEAN(area_quadrant_anterior_right)'] + df['MEAN(area_quadrant_posterior_right)']

    # Sum 'MEAN(area_quadrant_anterior_left)' and 'MEAN(area_quadrant_anterior_right)' into anterior side
    # and 'MEAN(area_quadrant_posterior_left)' and 'MEAN(area_quadrant_posterior_right)' into posterior side
    df['MEAN(area_half_anterior)'] = df['MEAN(area_quadrant_anterior_left)'] + df['MEAN(area_quadrant_anterior_right)']
    df['MEAN(area_half_posterior)'] = df['MEAN(area_quadrant_posterior_left)'] + df['MEAN(area_quadrant_posterior_right)']

    # Sanity check - Sum all quadrants into total area to be compared with the total area (MEAN(area)) from sct_process_segmentation
    df['MEAN(area_total)'] = df['MEAN(area_quadrant_anterior_left)'] + \
                             df['MEAN(area_quadrant_anterior_right)'] + \
                             df['MEAN(area_quadrant_posterior_left)'] + \
                             df['MEAN(area_quadrant_posterior_right)']

    figure_fname = f'T2w_lineplot_PAM50_quadrant_measures.png'
    figure_path = os.path.join(path_out, figure_fname)
    create_lineplot(df, figure_path)


if __name__ == '__main__':
    main()
