config = dict()

# Path to pickle file where the filenames used for the training are saved.
# The pickle contains a pandas dataframe with the following columns (one row per subject):
#   - "subject": subject name
#   - "img": absolute path towards the grayscale image
#   - "label": absolute path towards the label file containing one voxel with value=3, indicating the targeted voxel.
config['dataframe_database'] = ''

# Folder path where the preprocessed data will be saved.
# Please add a "/" at the end.
config['folder_out'] = ''

# Basename used to save the model in a file (eg model_c2c3 --> model_c2c3.yml)
config['model_name'] = ''

# Image contrast: either 't1', either 't2'
# Parameter used by sct_get_centerline during the preprocessing step.
config['contrast_centerline'] = ''