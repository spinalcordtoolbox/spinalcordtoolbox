config = dict()

# Path to pickle file that includes locations of images and ground-truth labels.
# The pickle contains a pandas dataframe with the following columns (one row per subject):
#   - "subject": subject name
#   - "img": absolute path to the grayscale image
#   - "label": absolute path to the label file containing one voxel with value=3, indicating the targeted voxel.
# To generate the panda file, follow the code below:
# >> import panda as pd
# >> d = {'subject': ['sub-01', 'sub-02'], 'img': ['/Users/JonDoe/data/sub-01/t2.nii.gz', '/Users/JonDoe/data/sub-02/t2_ax.nii.gz'], 'label': ['/Users/JonDoe/data/sub-01/t2_label.nii.gz', '/Users/JonDoe/data/sub-02/t2_ax_label.nii.gz']}
# >> df = pd.DataFrame(data=d)
# >> df.to_pickle(file_name)
config['dataframe_database'] = ''

# Folder where the preprocessed data will be saved.
# Please use absolute folder and add a "/" at the end.
config['folder_out'] = ''

# Basename used to save the model in a file.
# Example: basename 'model_c2c3' will generate model file: 'model_c2c3.yml'
config['model_name'] = ''

# Image contrast: 't1' or 't2'
# The contrast is used by sct_get_centerline during the preprocessing step.
config['contrast_centerline'] = ''