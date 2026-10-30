from __future__ import annotations
from tqdm import tqdm

from ivadomed.loader import film as imed_film
from ivadomed.loader.mri2d_segmentation_dataset import MRI2DSegmentationDataset
from ivadomed.object_detection import utils as imed_obj_detect
from ivadomed.keywords import ROIParamsKW, ContrastParamsKW, ModelParamsKW, MetadataKW, SubjectDictKW
import typing
if typing.TYPE_CHECKING:
    from ivadomed.loader.bids_dataframe import BidsDataframe
    from ivadomed.loader.slice_filter import SliceFilter
    from ivadomed.loader.patch_filter import PatchFilter
    import pandas as pd


class BidsDataset(MRI2DSegmentationDataset):
    """ BIDS specific dataset loader.

    Args:
        bids_df (BidsDataframe): Object containing dataframe with all BIDS image files and their metadata.
        subject_file_lst (list): Subject filenames list.
        target_suffix (list): List of suffixes for target masks.
        contrast_params (dict): Contains image contrasts related parameters.
        model_params (dict): Dictionary containing model parameters.
        slice_axis (int): Indicates the axis used to extract 2D slices from 3D NifTI files:
            "axial": 2, "sagittal": 0, "coronal": 1. 2D PNG/TIF/JPG files use default "axial": 2.
        nibabel_cache (bool): If the data should be cached in memory or not by nibabel to reduce repetitive disk loading.
        transform (list): Transformation list (length 2) composed of preprocessing transforms (Compose) and transforms
            to apply during training (Compose).
        metadata_choice (str): Choice between "mri_params", "contrasts", the name of a column from the
            participants.tsv file, None or False, related to FiLM.
        slice_filter_fn (SliceFilter): Class that filters slices according to their content.
        patch_filter_fn (PatchFilter): Class that filters patches according to their content.
        roi_params (dict): Dictionary containing parameters related to ROI image processing.
        multichannel (bool): If True, the input contrasts are combined as input channels for the model. Otherwise, each
            contrast is processed individually (ie different sample / tensor).
        object_detection_params (dict): Object dection parameters.
        task (str): Choice between segmentation or classification. If classification: GT is discrete values, \
            If segmentation: GT is binary mask.
        soft_gt (bool): If True, ground truths are not binarized before being fed to the network. Otherwise, ground
        truths are thresholded (0.5) after the data augmentation operations.
        is_input_dropout (bool): Return input with missing modalities.

    Attributes:
        filename_pairs (list): A list of tuples in the format (input filename list containing all modalities,ground \
            truth filename, ROI filename, metadata).
        metadata (dict): Dictionary containing FiLM metadata.
        soft_gt (bool): If True, ground truths are not binarized before being fed to the network. Otherwise, ground
            truths are thresholded (0.5) after the data augmentation operations.
        roi_params (dict): Dictionary containing parameters related to ROI image processing.

    """

    def __init__(self, bids_df: BidsDataframe, subject_file_lst: list, target_suffix: list, contrast_params: dict,
                 model_params: dict, slice_axis: int = 2, nibabel_cache: bool = True, transform: list = None,
                 metadata_choice: str = False, slice_filter_fn: SliceFilter = None, patch_filter_fn: PatchFilter = None,
                 roi_params: dict = None, multichannel: bool = False, object_detection_params: dict = None,
                 task: str = "segmentation", soft_gt: bool = False, is_input_dropout: bool = False):

        self.roi_params = roi_params if roi_params is not None else \
            {ROIParamsKW.SUFFIX: None, ROIParamsKW.SLICE_FILTER_ROI: None}
        self.soft_gt = soft_gt
        self.filename_pairs = []
        if metadata_choice == MetadataKW.MRI_PARAMS:
            self.metadata = {"FlipAngle": [], "RepetitionTime": [],
                             "EchoTime": [], "Manufacturer": []}

        # Sort subject_file_lst and create a sub-dataframe from bids_df containing only subjects from subject_file_lst
        subject_file_lst = sorted(subject_file_lst)
        df_subjects = bids_df.df[bids_df.df['filename'].isin(subject_file_lst)]

        # Create a dictionary with the number of subjects for each contrast of contrast_balance
        tot = {contrast: df_subjects['suffix'].str.fullmatch(contrast).value_counts()[True]
               for contrast in contrast_params[ContrastParamsKW.BALANCE].keys()}

        # Create a counter that helps to balance the contrasts
        c = {contrast: 0 for contrast in contrast_params[ContrastParamsKW.BALANCE].keys()}

        # Get a list of subject_ids for multichannel_subjects (prefix filename without modality suffix and extension)
        subject_ids = []
        for subject in subject_file_lst:
            subject_ids.append(subject.split('.')[0].split('_')[0])
        subject_ids = sorted(list(set(subject_ids)))

        # Create multichannel_subjects dictionary for each subject_id
        multichannel_subjects = {}
        idx_dict = {}
        if multichannel:
            num_contrast = len(contrast_params[ContrastParamsKW.CONTRAST_LST])
            for idx, contrast in enumerate(contrast_params[ContrastParamsKW.CONTRAST_LST]):
                idx_dict[contrast] = idx
            multichannel_subjects = {subject: {"absolute_paths": [None] * num_contrast,
                                               "deriv_path": None,
                                               "roi_filename": None,
                                               SubjectDictKW.METADATA: [None] * num_contrast} for subject in subject_ids}

        # Get all subjects path from bids_df for bounding box
        get_all_subj_path = bids_df.df[bids_df.df['filename']
                                .str.contains('|'.join(bids_df.get_subject_fnames()))]['path'].to_list()

        # Load bounding box from list of path
        bounding_box_dict = imed_obj_detect.load_bounding_boxes(object_detection_params,
                                                                get_all_subj_path,
                                                                slice_axis,
                                                                contrast_params[ContrastParamsKW.CONTRAST_LST])

        # Get all derivatives filenames from bids_df
        all_deriv = bids_df.get_deriv_fnames()

        # Create filename_pairs
        for subject in tqdm(subject_file_lst, desc="Loading dataset"):
            df_sub, roi_filename, target_filename, metadata = self.create_filename_pair(multichannel_subjects, subject,
                                                                                        c, tot, multichannel, df_subjects,
                                                                                        contrast_params, target_suffix,
                                                                                        all_deriv, bids_df, bounding_box_dict,
                                                                                        idx_dict, metadata_choice)
            # Fill multichannel dictionary
            # subj_id is the filename without modality suffix and extension
            if multichannel:
                multichannel_subjects = self.fill_multichannel_dict(multichannel_subjects, subject, idx_dict, df_sub,
                                                                    roi_filename, target_filename, metadata)
            else:
                self.filename_pairs.append(([df_sub['path'].values[0]],
                                            target_filename, roi_filename, [metadata]))

        if multichannel:
            for subject in multichannel_subjects.values():
                if None not in subject["absolute_paths"]:
                    self.filename_pairs.append((subject["absolute_paths"], subject["deriv_path"],
                                                subject["roi_filename"], subject[SubjectDictKW.METADATA]))

        if not self.filename_pairs:
            raise Exception('No subjects were selected - check selection of parameters on config.json (e.g. center '
                            'selected + target_suffix)')

        length = model_params[ModelParamsKW.LENGTH_2D] if ModelParamsKW.LENGTH_2D in model_params else []
        stride = model_params[ModelParamsKW.STRIDE_2D] if ModelParamsKW.STRIDE_2D in model_params else []

        super().__init__(self.filename_pairs, length, stride, slice_axis, nibabel_cache, transform, slice_filter_fn, patch_filter_fn,
                         task, self.roi_params, self.soft_gt, is_input_dropout)

    def get_target_filename(self, target_suffix: any, target_filename: any, derivative: any) -> None:
        for idx, suffix_list in enumerate(target_suffix):
            # If suffix_list is a string, then only one rater annotation per class is available.
            # Otherwise, multiple raters segmented the same class.
            if isinstance(suffix_list, list):
                for suffix in suffix_list:
                    if suffix in derivative:
                        target_filename[idx].append(derivative)
            elif suffix_list in derivative:
                target_filename[idx] = derivative


    def create_metadata_dict(self, metadata: dict, metadata_choice: any, df_sub: pd.DataFrame, bids_df: BidsDataframe) -> None:
        # add custom data to metadata
        if metadata_choice not in df_sub.columns:
            raise ValueError("The following metadata cannot be found: {}. "
                                "Invalid metadata choice.".format(metadata_choice))
        metadata[metadata_choice] = df_sub[metadata_choice].values[0]
        # Create metadata dict for OHE
        data_lst = sorted(set(bids_df.df[metadata_choice].dropna().values))
        metadata_dict = {}
        for idx, data in enumerate(data_lst):
            metadata_dict[data] = idx
        metadata[MetadataKW.METADATA_DICT] = metadata_dict

    def fill_multichannel_dict(self, multichannel_subjects: dict, subject: str, idx_dict: dict, df_sub: pd.DataFrame,
                               roi_filename: str, target_filename: str, metadata: any) -> dict:
        idx = idx_dict[df_sub['suffix'].values[0]]
        subj_id = subject.split('.')[0].split('_')[0]
        multichannel_subjects[subj_id]["absolute_paths"][idx] = df_sub['path'].values[0]
        multichannel_subjects[subj_id]["deriv_path"] = target_filename
        multichannel_subjects[subj_id][SubjectDictKW.METADATA][idx] = metadata
        if roi_filename:
            multichannel_subjects[subj_id]["roi_filename"] = roi_filename
        return multichannel_subjects


    def create_filename_pair(self, multichannel_subjects: dict, subject: str, c: dict, tot: dict, multichannel: any,
                             df_subjects: pd.DataFrame, contrast_params: dict, target_suffix: any, all_deriv: any,
                             bids_df: pd.DataFrame, bounding_box_dict: dict, idx_dict: dict, metadata_choice: str):
        df_sub = df_subjects.loc[df_subjects['filename'] == subject]

        # Training & Validation: do not consider the contrasts over the threshold contained in contrast_balance
        contrast = df_sub['suffix'].values[0]
        if contrast in (contrast_params[ContrastParamsKW.BALANCE].keys()):
            c[contrast] = c[contrast] + 1
            if c[contrast] / tot[contrast] > contrast_params[ContrastParamsKW.BALANCE][contrast]:
                return
        if isinstance(target_suffix[0], str):
            target_filename, roi_filename = [None] * len(target_suffix), None
        else:
            target_filename, roi_filename = [[] for _ in range(len(target_suffix))], None

        derivatives = bids_df.df[bids_df.df['filename']
                        .str.contains('|'.join(bids_df.get_derivatives(subject, all_deriv)))]['path'].to_list()

        for derivative in derivatives:
            self.get_target_filename(target_suffix, target_filename, derivative)
            if not (self.roi_params[ROIParamsKW.SUFFIX] is None) and self.roi_params[ROIParamsKW.SUFFIX] in derivative:
                roi_filename = [derivative]

        if (not any(target_filename)) or (not (self.roi_params[ROIParamsKW.SUFFIX] is None) and (roi_filename is None)):
            return

        metadata = df_sub.to_dict(orient='records')[0]
        metadata[MetadataKW.CONTRAST] = contrast

        if len(bounding_box_dict):
            # Take only one bounding box for cropping
            metadata[MetadataKW.BOUNDING_BOX] = bounding_box_dict[str(df_sub['path'].values[0])][0]

        if metadata_choice == MetadataKW.MRI_PARAMS:
            if not all([imed_film.check_isMRIparam(m, metadata, subject, self.metadata) for m in
                        self.metadata.keys()]):
                return

        elif metadata_choice and metadata_choice != MetadataKW.CONTRASTS and metadata_choice is not None:
            self.create_metadata_dict(metadata, metadata_choice, df_sub, bids_df)

        return df_sub, roi_filename, target_filename, metadata
