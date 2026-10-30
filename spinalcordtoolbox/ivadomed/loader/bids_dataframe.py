import copy
import itertools
import os

import bids as pybids
import pandas as pd
from loguru import logger
from pathlib import Path


class BidsDataframe:
    """
    This class aims to create a dataframe containing all BIDS image files in a list of path_data and their metadata.

    Args:
        loader_params (dict): Loader parameters, see :doc:`configuration_file` for more details.
        path_output (str): Output folder.
        derivatives (bool): If True, derivatives are indexed.
        split_method (str): split_method from Split Dataset parameters, see :doc:`configuration_file` for more details.
            Default: None. Used to remove unused subject files from the bids_dataframe.

    Attributes:
        path_data (list): Paths to the BIDS datasets.
        bids_config (str): Path to the custom BIDS configuration file.
        target_suffix (list of str): List of suffix of targetted structures.
        roi_suffix (str): List of suffix of ROI masks.
        extensions (list of str): List of file extensions of interest.
        contrast_lst (list of str): List of the contrasts of interest.
        derivatives (bool): If True, derivatives are indexed.
        split_method (str): split_method from Split Dataset parameters
        df (pd.DataFrame): Dataframe containing dataset information
    """

    def __init__(self, loader_params: dict, path_output: str, derivatives: bool, split_method: str = None) -> None:

        # paths_data from loader parameters
        self.paths_data = loader_params['path_data']

        # bids_config from loader parameters
        self.bids_config = None if 'bids_config' not in loader_params else loader_params['bids_config']

        # target_suffix and roi_suffix from loader parameters
        self.target_suffix = copy.deepcopy(loader_params['target_suffix'])

        # If `target_suffix` is a list of lists convert to list
        if any(isinstance(t, list) for t in self.target_suffix):
            self.target_suffix = list(itertools.chain.from_iterable(self.target_suffix))

        self.roi_suffix = loader_params['roi_params']['suffix']

        # If `roi_suffix` is not None, add to target_suffix
        if self.roi_suffix is not None:
            self.target_suffix.append(self.roi_suffix)

        self.bids_validate = loader_params.get('bids_validate', True)

        # extensions from loader parameters
        self.extensions = loader_params['extensions'] if loader_params['extensions'] else [".nii", ".nii.gz"]

        # contrast_lst from loader parameters
        self.contrast_lst = [] if 'contrast_lst' not in loader_params['contrast_params'] \
            else loader_params['contrast_params']['contrast_lst']

        # split_method
        self.split_method = split_method

        # derivatives
        self.derivatives = derivatives

        # Create dataframe
        self.df = pd.DataFrame()
        self.create_bids_dataframe()

        # Save dataframe as csv file
        self.save(str(Path(path_output, "bids_dataframe.csv")))

    def create_bids_dataframe(self) -> None:
        """Generate the dataframe."""

        for path_data in self.paths_data:
            path_data = Path(path_data, '')

            # For CT-scan files:
            # Force indexing of subject subfolders containing CT-scan files.
            # As of 20221026: Implementation based on potential CT datatypes (anat or ct), extensions and
            # modality suffixes discussed in BEP024 (https://bids.neuroimaging.io/bep024).
            # bids_config parameter with default config_bids.json must be used
            # (see: https://ivadomed.org/configuration_file.html#bids-config)
            # TODO: remove force indexing of CT-scan files when BEP024 is merged in BIDS.
            extension_ct = ('.nii.gz', '.nii')
            suffix_ct = ('ct', 'CT')
            force_index = []
            for path_object in path_data.glob('**/*'):
                if path_object.is_file():
                    subject_path_index = len(path_data.parts)
                    subject_path = path_object.parts[subject_path_index]
                    if (path_object.name.endswith(extension_ct) and path_object.name.split('.')[0].endswith(suffix_ct) and
                            (path_object.parent.name == "anat" or path_object.parent.name == "ct") and
                            subject_path.startswith('sub')):
                        force_index.append(str(Path(*path_object.parent.parts[subject_path_index:])))

            # Initialize BIDSLayoutIndexer and BIDSLayout
            # validate=True by default for both indexer and layout, BIDS-validator is not skipped
            indexer = pybids.BIDSLayoutIndexer(force_index=force_index, validate=self.bids_validate)
            if self.derivatives:
                self.write_derivatives_dataset_description(path_data)
            layout = pybids.BIDSLayout(str(path_data), config=self.bids_config, indexer=indexer,
                                       derivatives=self.derivatives)

            # Transform layout to dataframe with all entities and json metadata
            # As per pybids, derivatives don't include parsed entities, only the "path" column
            df_next = layout.to_df(metadata=True)

            # Add filename column
            df_next.insert(1, 'filename', df_next['path'].apply(os.path.basename))

            # Drop rows with json, tsv and LICENSE files in case no extensions are provided in config file for filtering
            df_next = df_next[~df_next['filename'].str.endswith(tuple(['.json', '.tsv', 'LICENSE']))]

            # The following command updates the dataframe by doing 2 things:
            # 1. Keep only subject files of chosen contrasts (for files that are not in the 'derivatives' folder)
            #    (ex: '<dataset_path>/sub-XX/anat/sub-XX_T1w.nii.gz' with contrast_lst:["T1w"])
            # 2. Keep only derivatives files of chosen target_suffix (for files that are in the 'derivatives' folder)
            #    (ex: '<dataset_path>/derivatives/labels/sub-XX/anat/sub-XX_T1w_seg-manual.nii.gz' with target_suffix:["_seg-manual"])
            df_next = df_next[(~df_next['path'].str.contains('derivatives')
                               & df_next['suffix'].str.contains('|'.join(self.contrast_lst)))
                               | (df_next['path'].str.contains('derivatives')
                               & (df_next['filename'].str.split('.').apply(lambda x: x[0])).str.endswith(tuple(self.target_suffix)))]

            # Update dataframe with files of chosen extensions
            df_next = df_next[df_next['filename'].str.endswith(tuple(self.extensions))]

            # Warning if no subject files are found in path_data
            if df_next[~df_next['path'].str.contains('derivatives')].empty:
                logger.warning(f"No subject files were found in '{path_data}' dataset. Skipping dataset.")
            else:
                # Add tsv files metadata to dataframe
                df_next = self.add_tsv_metadata(df_next, path_data, layout)

                # TODO: check if other files are needed for EEG and DWI

                # Merge dataframes
                self.df = pd.concat([self.df, df_next], join='outer', ignore_index=True)

        if self.df.empty:
            # Raise error and exit if no subject files are found in any path data
            raise RuntimeError("No subject files found. Check selection of parameters in config.json"
                               " and datasets compliance with BIDS specification.")

        # Drop duplicated rows based on all columns except 'path'
        # Keep first occurence
        columns = self.df.columns.to_list()
        columns.remove('path')
        self.df = self.df[~(self.df.astype(str).duplicated(subset=columns, keep='first'))]

        # Remove subject files without the "split_method" metadata if specified and keep all derivatives
        if self.split_method:
            files_remove = (self.df[(
                                # Path does not contain derivative string (i.e. we only target subject raw data files)
                                ~self.df['path'].str.contains('derivatives')
                                # and split method metadata is null (i.e. the subject must have the split_method metadata or will be excluded)
                                & self.df[self.split_method].isnull())]
                                # Get these filesnames and convert to list.
                                ['filename']).tolist()
            if files_remove:
                logger.warning(f"The following files don't have the '{self.split_method}' metadata indicated as the "
                               f"split_method in the configuration JSON file. Skipping these files: {files_remove}")
                # Removing from dataframe all filenames which contain any of the file from files_remove field.
                self.df = self.df[~self.df['filename'].str.contains('|'.join(files_remove))]

        # If indexing of derivatives is true
        if self.derivatives:

            # Get list of subject files with available derivatives
            has_deriv, deriv = self.get_subjects_with_derivatives()

            # Filter dataframe to keep subjects files with available derivatives only
            if has_deriv:
                self.df = self.df[self.df['filename'].str.contains('|'.join(has_deriv))
                                  | self.df['filename'].str.contains('|'.join(deriv))]
            else:
                # Raise error and exit if no derivatives are found for any subject files
                raise RuntimeError("Derivatives not found.")

        # Reset index
        self.df.reset_index(drop=True, inplace=True)

        # Drop columns with all null values
        self.df.dropna(axis=1, inplace=True, how='all')

    def add_tsv_metadata(self, df: pd.DataFrame, path_data: str, layout: pybids.BIDSLayout) -> pd.DataFrame:
        """Add tsv files metadata to dataframe.

        Args:
            df (pd.DataFrame): Dataframe containing dataset information
            path_data (str): Path to the BIDS dataset
            layout (pybids.BIDSLayout): pybids BIDSLayout of the indexed files of the path_data

        Returns:
            pd.DataFrame: Dataframe containing datasets information
        """

        # Drop columns with all null values before loading TSV metadata
        # Avoid conflicts with unused columns descriptions from TSV sidecar JSON files
        df.dropna(axis=1, inplace=True, how='all')

        # Add metadata from 'participants.tsv' file if present
        # Uses pybids function
        if layout.get_collections(level='dataset'):
            df_participants = layout.get_collections(level='dataset', merge=True).to_df()
            df_participants.insert(1, 'participant_id', "sub-" + df_participants['subject'])
            df_participants.drop(['suffix'], axis=1, inplace=True)
            df = pd.merge(df, df_participants, on='subject', suffixes=("_x", None), how='left')

        # Add metadata from 'samples.tsv' file if present
        # The 'participant_id' column is added only if not already present from the 'participants.tsv' file.
        # TODO: update to pybids function when the indexing of samples.tsv is integrated in pybids
        # (see: https://github.com/bids-standard/pybids/issues/843)
        fname_samples = Path(path_data, "samples.tsv")
        if fname_samples.exists():
            df_samples = pd.read_csv(str(fname_samples), sep='\t')
            df_samples['sample'] = df_samples['sample_id'].str.split("sample-").apply(lambda x: x[1])
            df_samples['subject'] = df_samples['participant_id'].str.split("sub-").apply(lambda x: x[1])
            columns = df_samples.columns.tolist()
            if 'participant_id' in df.columns:
                columns.remove('participant_id')
            df = pd.merge(df, df_samples[columns], on=['subject', 'sample'], suffixes=("_x", None), how='left')

        # Add metadata from all _sessions.tsv files, if present
        # Uses pybids function
        if layout.get_collections(level='subject'):
            df_sessions = layout.get_collections(level='subject', merge=True).to_df()
            df_sessions.drop(['suffix'], axis=1, inplace=True)
            df = pd.merge(df, df_sessions, on=['subject', 'session'], suffixes=("_x", None), how='left')

        # Add metadata from all _scans.tsv files, if present
        # TODO: implement reading _scans.tsv files using pybids "layout.get_collections(level='session')"
        # TODO: verify merge behavior with EEG and DWI scans files, tested with anat and micr only
        df_scans = pd.DataFrame()
        for path_object in Path(path_data).glob("**/*"):
            if path_object.is_file():
                if path_object.name.endswith("scans.tsv"):
                    df_temp = pd.read_csv(str(path_object), sep='\t')
                    df_scans = pd.concat([df_scans, df_temp], ignore_index=True)
        if not df_scans.empty:
            df_scans['filename'] = df_scans['filename'].apply(os.path.basename)
            df = pd.merge(df, df_scans, on=['filename'], suffixes=("_x", None), how='left')

        return df

    def get_subjects_with_derivatives(self) -> (list, list):
        """Get lists of subject filenames with available derivatives.

        Returns:
            list, list: subject filenames having derivatives, available derivatives filenames.
        """
        subject_fnames = self.get_subject_fnames()
        deriv_fnames = self.get_deriv_fnames()
        has_deriv = []
        deriv = []

        for subject_fname in subject_fnames:
            available = self.get_derivatives(subject_fname, deriv_fnames)
            if available:
                if self.roi_suffix is not None:
                    if self.roi_suffix in ('|'.join(available)):
                        has_deriv.append(subject_fname)
                        deriv.extend(available)
                    else:
                        logger.warning(f"Missing roi_suffix {self.roi_suffix} for {subject_fname}. Skipping file.")
                else:
                    has_deriv.append(subject_fname)
                    deriv.extend(available)
                for target in self.target_suffix:
                    if target not in str(available) and target != self.roi_suffix:
                        logger.warning(f"Missing target_suffix {target} for {subject_fname}")
            else:
                logger.warning(f"Missing derivatives for {subject_fname}. Skipping file.")

        return has_deriv, deriv

    def get_subject_fnames(self) -> list:
        """Get the list of subject filenames in dataframe.

        Returns:
            list: subject filenames.
        """
        return self.df[~self.df['path'].str.contains('derivatives')]['filename'].to_list()

    def get_deriv_fnames(self) -> list:
        """Get the list of derivative filenames in dataframe.

        Returns:
            list: derivative filenames.
        """
        return self.df[self.df['path'].str.contains('derivatives')]['filename'].tolist()

    def get_derivatives(self, subject_fname: str, deriv_fnames: list) -> list:
        """Return list of available derivative filenames for a subject filename.

        Args:
            subject_fname (str): Subject filename.
            deriv_fnames (list of str): List of derivative filenames.

        Returns:
            list: derivative filenames
        """
        prefix_fname = subject_fname.split('.')[0]
        return [d for d in deriv_fnames if prefix_fname in d]

    def save(self, path: str) -> None:
        """Save the dataframe into a csv file.

        Args:
            path (str): Path to csv file.
        """
        try:
            self.df.to_csv(path, index=False)
            logger.info(f"Dataframe has been saved in {path}.")
        except FileNotFoundError:
            logger.error(f"Wrong path, bids_dataframe.csv could not be saved in {path}.")

    def write_derivatives_dataset_description(self, path_data: str) -> None:
        """Writes default dataset_description.json file if not found in path_data/derivatives folder

        Args:
            path_data (str): Path to the BIDS dataset.
        """
        path_data = Path(path_data).absolute()

        filename = 'dataset_description'
        path_deriv_desc_file = Path(f'{path_data}/derivatives/{filename}.json')
        path_label_desc_file = Path(f'{path_data}/derivatives/labels/{filename}.json')
        # need to write default dataset_description.json file if not found
        if not path_deriv_desc_file.is_file() and not path_label_desc_file.is_file():

            logger.warning(f"{path_deriv_desc_file} not found. Will attempt to create a place holder "
                           f"description file for now at {path_deriv_desc_file}.")
            with path_deriv_desc_file.open(mode='w') as f:
                f.write(
                    '{"Name": "Example dataset", '
                    '"BIDSVersion": "1.0.2", '
                    '"GeneratedBy": [{"Name": "Example pipeline"}]}'
                )
