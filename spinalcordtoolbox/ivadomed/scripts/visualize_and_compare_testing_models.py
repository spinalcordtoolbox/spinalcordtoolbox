###########################################################################################################
#
# This script makes a figure with violinplots and significance values to compare between models
#
#        python3 visualize_and_compare_testing_models.py --ofolders path/to/ofolder1 path/to/ofolder2
#                              --metric metric_to_use --metadata metadata_label string_to_match

# Konstantinos Nasiotis 2021
###########################################################################################################

import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import itertools
import seaborn as sns
from scipy.stats import ks_2samp
from ivadomed.utils import init_ivadomed
from pathlib import Path
from loguru import logger
import argparse

matplotlib.rcParams['toolbar'] = 'None'  # Remove buttons

if matplotlib.get_backend() == "agg":
    logger.warning("No backend can be used - Visualization will fail")
else:
    logger.info(f"Using: {matplotlib.get_backend()}  gui")


# ---------------------------------------------------------------------------------------------------------------------#


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ofolders", required=True, nargs="*", dest="ofolders",
                        help="List of log folders from different models.")
    parser.add_argument("--metric", default='dice_class0', nargs=1, type=str, dest="metric",
                        help="Metric from evaluation_3Dmetrics.csv to base the plots on.")
    parser.add_argument("--metadata", required=False, nargs=2, type=str, dest="metadata",
                        help="Selection based on metadata from participants.tsv:"
                             "(1) Label from column (2) string to match")
    return parser


def onclick(event, df):
    # Get the index of the selected violinplot datapoint
    # WARNING: More than one can be selected if they are very close to each other
    #          If that's the case, all will be displayed
    clicked_index = event.ind

    fig = plt.gca()
    # clicking outside the plot area produces a coordinate of None, so we filter those out.
    if None not in clicked_index:
        output_folders = df["EvaluationModel"].unique()
        nfolders = len(output_folders)

        # Remove the previously displayed subject(s)
        # This also takes care of the case where more than one subjects are displayed
        while len(fig.texts) > nfolders + np.math.factorial(nfolders) / (
                np.math.factorial(2) * np.math.factorial(nfolders - 2)):
            fig.texts.pop()

        # This is a hack to find the index of the Violinplot - There should be another way to get this from the
        # figure object
        bins = np.linspace(-1, 1, len(output_folders) + 1)
        i_output_folder = np.where(bins < event.mouseevent.xdata)
        i_output_folder = i_output_folder[-1][0]
        selected_output_folder = df[df["EvaluationModel"] == output_folders[i_output_folder]]

        for iSubject in range(0, len(clicked_index.tolist())):
            frame = plt.text(event.mouseevent.xdata, -0.08 - 0.08 * iSubject + event.mouseevent.ydata,
                             selected_output_folder["subject"][clicked_index[iSubject]], size=10,
                             ha="center", va="center",
                             bbox=dict(facecolor='red', alpha=0.5)
                             )
            # To show the artist
            matplotlib.artist.Artist.set_visible(frame, True)

        plt.show()


def visualize_and_compare_models(ofolders, metric="dice_class0", metadata=None):
    """
    This function allows violinplots visualization of multiple evaluation models simultaneously and performs a
    Kolmogorov–Smirnov significance test between each combination of models. The mean values of the datapoints for each
    violinplot is superimposed on the top.

    If only one model is selected as input, only the Violinplot will be presented (no test will be superimposed)

    The datapoints within each violinplot are interactive. The subject_id and MRI sequence of each point are displayed
    when clicked (as shown on the violinplot to the right of the example figure below).

    .. note::
        If more than 4 model outputs are selected to be compared, the significance tests are not displayed since the
        figure becomes very busy

    Usage example::

        visualize_and_compare_testing_models.py --ofolders ~/logs/logs_T1w ~/logs/logs_T2w
                                                --metric dice_class0 --metadata pathology ms

    .. image:: ../../images/visualize_and_compare_models.png
            :width: 600px
            :align: center


    Args:
        ofolders (list): list of folders that contain the outputs of the models to be compared, Flag: ``--ofolders``
        metric (str):      column of "results_eval/evaluation_3Dmetrics.csv" to be used on the plots
                           (default: dice_class0), Flag: ``--metric``
        metadata (list) - Optional:   Allows visualization of violinplots only from subjects that match the
                           metadata criteria.
                           2 elements - (1) column label of the dataframe.csv metadata so only subjects that belong to
                           that category will be used and (2) string to be matched, Flag: ``--metadata``, Example: "--metadata pathology ms"
    """

    # access CLI options
    logger.debug(f"ofolders: {ofolders}")
    logger.debug(f"metric: {metric}")
    if metadata is None:
        metadata = []
    if metadata:
        logger.debug(f"metadata: {metadata}")

    # Do a quick check that all the required files are present
    for folder in ofolders:
        if not Path(folder, 'results_eval', 'evaluation_3Dmetrics.csv').exists():
            logger.error(f"evaluation_3Dmetrics.csv file is not present within {Path(folder, 'results_eval')}")
            raise Exception('evaluation_3Dmetrics.csv missing')
        if not Path(folder, 'bids_dataframe.csv').exists():
            logger.error(f"bids_dataframe.csv file is not present within {folder}")
            raise Exception('bids_dataframe.csv missing')

    if len(ofolders) < 1:
        raise Exception('No folders were selected - Nothing to show')

    columnNames = ["EvaluationModel", metric, 'subject']
    df = pd.DataFrame([], columns=columnNames)

    for folder in ofolders:
        result = pd.read_csv(str(Path(folder, 'results_eval', 'evaluation_3Dmetrics.csv')))

        if metadata:
            participant_metadata = pd.read_table(str(Path(folder, 'bids_dataframe.csv')), sep=',')
            # Select only the subjects that satisfy the --metadata input
            selected_subjects = participant_metadata[participant_metadata[metadata[0]] == metadata[1]][
                "filename"].tolist()
            selected_subjects = [i.replace(".nii.gz", "") for i in selected_subjects]

            # Now select only the scores from these subjects
            result_subject_ids = result["image_id"]
            result = result.iloc[
                [i for i in range(len(result_subject_ids)) if result_subject_ids[i] in selected_subjects]]

            if result.empty:
                logger.warning(f"No subject meet the selected criteria - skipping plot for: {folder}")

        if not result.empty:
            scores = result[metric]

            folders = [Path(folder).resolve().name] * len(scores)
            subject_id = result["image_id"]
            combined = np.column_stack((folders, scores.astype(np.object, folders), subject_id)).T
            singleFolderDF = pd.DataFrame(combined, columnNames).T
            df = df.append(singleFolderDF, ignore_index=True)

    nFolders = len(ofolders)
    combinedNumbers = list(itertools.combinations(range(nFolders), 2))
    combinedFolders = list(itertools.combinations(ofolders, 2))

    # Pandas annoying issues
    df[metric] = df[metric].astype('float64')

    if not df.empty:

        # Plot all violinplots
        plt1 = sns.violinplot(x="EvaluationModel", y=metric, data=df, palette="muted", saturation=0.3,
                              inner='quartile', picker=True, pickradius=1)
        plt2 = sns.stripplot(x="EvaluationModel", y=metric, data=df, linewidth=0.5, edgecolor="black",
                             jitter=True, zorder=1, picker=True, pickradius=1)

        # Display the mean performance on top of every violinplot
        for i in range(len(ofolders)):
            # This will be used to plot the mean value on top of each individual violinplot
            temp = df[metric][df['EvaluationModel'] == Path(ofolders[i]).resolve().name]
            plt.text(i, df[metric].max() + 0.07, str((100 * temp.mean()).round() / 100), ha='center', va='top',
                     color='r', picker=True)

        if len(ofolders) > 1 and len(ofolders) < 5:
            # Perform a Kolmogorov–Smirnov test for all combinations of results & connect the corresponding Violinplots
            for i in range(len(combinedNumbers)):
                dataX = df[metric][df['EvaluationModel'] == Path(combinedFolders[i][0]).resolve().name]
                dataY = df[metric][df['EvaluationModel'] == Path(combinedFolders[i][1]).resolve().name]

                ks_test = ks_2samp(dataX, dataY)

                x1, x2 = combinedNumbers[i]

                y, h, col = df[metric].min() - 0.06 - 0.03 * i, -0.01, 'k'
                plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c=col, picker=True)

                # Show if the differentiation of the distributions is :
                # Not significant: ns, significant: *, very significant: ***
                if ks_test.pvalue >= 0.5:
                    plt.text((x1 + x2) * .5, y + h, "ns", ha='center', va='bottom', color=col, picker=True)
                elif 0.5 > ks_test.pvalue >= 0.01:
                    plt.text((x1 + x2) * .5, y + h, "*", ha='center', va='bottom', color='r', picker=True)
                elif ks_test.pvalue < 0.01:
                    plt.text((x1 + x2) * .5, y + h, "***", ha='center', va='bottom', color='r', picker=True)

        if metadata:
            plt.title("Metric:  " + metric + "\nMetadata:  " + metadata[0] + ":" + metadata[1], picker=True)
        else:
            plt.title("Metric:  " + metric, picker=True)

        plt.grid()
        plt.text(0, 0, " ")  # One empty entry is introduced here so it is popped in the function
        plt.gca().figure.canvas.mpl_connect('pick_event',
                                            lambda event: onclick(event, df))

        plt.show(block=True)

    else:
        logger.warning("No subjects meet the criteria selected for any model. "
                       "Probably you need to change the --metadata / --metric selection")


def main():
    init_ivadomed()

    parser = get_parser()
    args = parser.parse_args()

    # Run automate training
    visualize_and_compare_models(args.ofolders, args.metric, args.metadata)


if __name__ == '__main__':
    main()
