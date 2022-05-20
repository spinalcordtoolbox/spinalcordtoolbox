import os

import numpy as np
import nibabel as nib
from nilearn.image import resample_img

import torch
# import VoxelMorph with pytorch backend
os.environ['VXM_BACKEND'] = 'pytorch'
import voxelmorph as vxm

# Load preprocessed data (scaled between 0 and 1 and with the moving data in the space of the fixed one)
fixed = nib.load("data_processed_time_analysis/data2/sub-geneva06/anat/sub-geneva06_T1w.nii.gz")
moving = nib.load("data_processed_time_analysis/data2/sub-geneva06/anat/sub-geneva06_T2w.nii.gz")
# N.B.
# These data are in my local computer but any data could be used to perform the same analysis.
# It only needs to be scaled and set in a common space (e.g. using sct_register_multimodal with -identity 1)

# Define the input shape of the model (smallest divider of 16 above the input data shape)
# Ensure that the volumes can be used in the registration model
fx_img_shape = fixed.get_fdata().shape
mov_img_shape = moving.get_fdata().shape
max_img_shape = max(fx_img_shape, mov_img_shape)
new_img_shape = (int(np.ceil(max_img_shape[0] // 16)) * 16, int(np.ceil(max_img_shape[1] // 16)) * 16,
                 int(np.ceil(max_img_shape[2] // 16)) * 16)

# Pad the volumes to the max shape
fx_paded = resample_img(fixed, target_affine=fixed.affine, target_shape=new_img_shape, interpolation='continuous')
mov_paded = resample_img(moving, target_affine=moving.affine, target_shape=new_img_shape, interpolation='continuous')
input_shape = list(new_img_shape)

# Set the parameters of the registration model
reg_args = dict(
    inshape=input_shape,
    int_steps=5,
    int_downsize=2,
    unet_half_res=True,
    nb_unet_features=([256, 256, 256, 256], [256, 256, 256, 256, 256, 256])
)
# Create the PyTorch model and specify the device
device = 'cpu'

# ---- First Model ---- #
pt_first_model = vxm.networks.VxmDense(**reg_args)
trained_state_dict_first_model = torch.load('pt_cascaded_first_model.pt')
# Load the weights to the PyTorch model
weights_first_model = []
for k in trained_state_dict_first_model:
    weights_first_model.append(trained_state_dict_first_model[k])
i = 0
i_max = len(list(pt_first_model.named_parameters()))
torchparam = pt_first_model.state_dict()
for k, v in torchparam.items():
    if i < i_max:
        torchparam[k] = weights_first_model[i]
        i += 1
pt_first_model.load_state_dict(torchparam)
pt_first_model.eval()

# ---- Second Model ---- #
pt_second_model = vxm.networks.VxmDense(**reg_args)
trained_state_dict_second_model = torch.load('pt_cascaded_second_model.pt')
# Load the weights to the PyTorch model
weights_second_model = []
for k in trained_state_dict_second_model:
    weights_second_model.append(trained_state_dict_second_model[k])
i = 0
i_max = len(list(pt_second_model.named_parameters()))
torchparam = pt_second_model.state_dict()
for k, v in torchparam.items():
    if i < i_max:
        torchparam[k] = weights_second_model[i]
        i += 1
pt_second_model.load_state_dict(torchparam)
pt_second_model.eval()

# Prepare the data for inference
data_moving = np.expand_dims(mov_paded.get_fdata().squeeze(), axis=(0, -1)).astype(np.float32)
data_fixed = np.expand_dims(fx_paded.get_fdata().squeeze(), axis=(0, -1)).astype(np.float32)
# Set up tensors and permute for inference
input_moving = torch.from_numpy(data_moving).to(device).float().permute(0, 4, 1, 2, 3)
input_fixed = torch.from_numpy(data_fixed).to(device).float().permute(0, 4, 1, 2, 3)

# Predict using cascaded networks
moved, warp_tensor = pt_first_model(input_moving, input_fixed, registration=True)
warp_data_first = warp_tensor[0].permute(1, 2, 3, 0).detach().numpy()

# Saved the moved data after the first step of the process to directly observe the results
moved_first_reg_data = moved[0][0].detach().numpy()
moved_nifti = nib.Nifti1Image(moved_first_reg_data, fixed.affine)
nib.save(moved_nifti, 'moved_first_reg.nii.gz')

moved_final, warp_tensor = pt_second_model(moved, input_fixed, registration=True)
warp_data_second = warp_tensor[0].permute(1, 2, 3, 0).detach().numpy()
# Saved the moved data at the end of the registration process
moved_data = moved_final[0][0].detach().numpy()
nib.save(nib.Nifti1Image(moved_data, fixed.affine), 'moved.nii.gz')

# Warping field
# Modify the warp data so it can be used with sct_apply_transfo()
# (add a time dimension, change the sign of some axes and set the intent code to vector)

# Change the sign of the vectors and the order of the axes components to be correctly used with sct_apply_transfo
# and to to get the same results with sct_apply_transfo() and when using model.predict() or vxm.networks.Transform()
orientation_conv = "LPS"
fx_im_orientation = list(nib.aff2axcodes(fixed.affine))
opposite_character = {'L': 'R', 'R': 'L', 'A': 'P', 'P': 'A', 'I': 'S', 'S': 'I'}
perm = [0, 1, 2]
inversion = [1, 1, 1]
for i, character in enumerate(orientation_conv):
    try:
        perm[i] = fx_im_orientation.index(character)
    except ValueError:
        perm[i] = fx_im_orientation.index(opposite_character[character])
        inversion[i] = -1

# First Warping Field
# Add the time dimension
warp_data_exp = np.expand_dims(warp_data_first, axis=3)
warp_data_exp_copy = np.copy(warp_data_exp)
warp_data_exp[..., 0] = inversion[0] * warp_data_exp_copy[..., perm[0]]
warp_data_exp[..., 1] = inversion[1] * warp_data_exp_copy[..., perm[1]]
warp_data_exp[..., 2] = inversion[2] * warp_data_exp_copy[..., perm[2]]
warp = nib.Nifti1Image(warp_data_exp, fixed.affine)
warp.header['intent_code'] = 1007

# Save the warping field that can be later used with sct_apply_transfo
nib.save(warp, f'warp_field_first_reg.nii.gz')

# Second Warping Field
# Add the time dimension
warp_data_exp = np.expand_dims(warp_data_second, axis=3)
warp_data_exp_copy = np.copy(warp_data_exp)
warp_data_exp[..., 0] = inversion[0] * warp_data_exp_copy[..., perm[0]]
warp_data_exp[..., 1] = inversion[1] * warp_data_exp_copy[..., perm[1]]
warp_data_exp[..., 2] = inversion[2] * warp_data_exp_copy[..., perm[2]]
warp = nib.Nifti1Image(warp_data_exp, fixed.affine)
warp.header['intent_code'] = 1007

# Save the warping field that can be later used with sct_apply_transfo
nib.save(warp, f'warp_field_second_reg.nii.gz')