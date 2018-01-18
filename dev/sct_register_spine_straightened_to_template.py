#Tests for registrations - Template, anat_straightened


from nibabel import load, Nifti1Image, save

# Inputs
ext = '.nii.gz'
prefix_ants = 'tmp_'
suffix_ants = '_reg2template'

template_path = '/home/django/gleveque/data/spinalcordtoolbox_dev__testing/data/template/'
template_file = template_path + 'MNI-POLY-AMU_v1__T2'
template_file_masked = 'MNI-POLY-AMU_v1__T2_masked'
landmarks_template_file = 'landmarks_template_C1C2_T5T6'

anat_file = 'errsm_24_t2_reg_PSE'
anat_file_reorient = anat_file + '_reorient'
anat_file_reorient_masked = anat_file_reorient + '_masked'
landmarks_anat_file = 'landmarks_anat'
landmarks_anat_file_reorient = landmarks_anat_file + '_reorient'

# Preprocessing: cropping along SI axis outside the landmarks
# orientation for MNI_POLY_AMU_v1__T2: AP RL IS
cmd = 'fslswapdim ' + anat_file + ext + ' RL PA IS ' + anat_file_reorient + ext
print('>> '+ cmd)
status, PWD = getstatusoutput(cmd)

cmd = 'fslswapdim ' + landmarks_anat_file + ext + ' RL PA IS ' + landmarks_anat_file_reorient + ext
print('>> '+ cmd)
status, PWD = getstatusoutput(cmd)

# read nifti input file
anat_file_nibabel = load(anat_file_reorient + '.nii.gz')
# 3d array for each x y z voxel values for the input nifti image
anat = anat_file_nibabel.get_data()
shape = anat.shape
print "Anat volume dim.:"
print shape

landmarks_anat_file_nibabel = load(landmarks_anat_file_reorient + '.nii.gz')
# 3d array for each x y z voxel values for the input nifti image
landmarks_anat = landmarks_anat_file_nibabel.get_data()
shape = landmarks_anat.shape
print "Landmarks anat volume dim.:"
print shape

# read nifti input file
template_file_nibabel = load(template_file + '.nii.gz')
# 3d array for each x y z voxel values for the input nifti image
template = template_file_nibabel.get_data()
shape = template.shape
print "Template volume dim.:"
print shape

landmarks_template_file_nibabel = load(landmarks_template_file + '.nii.gz')
# 3d array for each x y z voxel values for the input nifti image
landmarks_template = landmarks_template_file_nibabel.get_data()
shape = landmarks_template.shape
print "Template volume dim.:"
print shape



# masking anat
X, Y, Z = (landmarks_anat > 0).nonzero()

if Z[0] > Z[1]:
    for i in range (0, shape[0]):
        for j in range(0, shape[1]):
            for k in range(Z[0], shape[2]):
                landmarks_anat[i][j][k] = 0

    for i in range (0, shape[0]):
        for j in range(0, shape[1]):
            for k in range(0, Z[1]):
                landmarks_anat[i][j][k] = 0

else:

    for i in range (0, shape[0]):
        for j in range(0, shape[1]):
            for k in range(Z[1], shape[2]):
                landmarks_anat[i][j][k] = 0

    for i in range (0, shape[0]):
        for j in range(0, shape[1]):
            for k in range(0, Z[0]):
                landmarks_anat[i][j][k] = 0

hdr = anat_file_nibabel.get_header()
img = Nifti1Image(landmarks_anat, None, hdr)
save(img, anat_file_reorient_masked + '.nii.gz')

cmd = 'isct_c3d ' + anat_file_reorient + ext + ' ' + anat_file_reorient_masked + ext + ' -copy-transform -o ' + anat_file_reorient_masked + ext
print('>> '+ cmd)
status, PWD = getstatusoutput(cmd)



# masking anat
X, Y, Z = (landmarks_template > 0).nonzero()

if Z[0] > Z[1]:
    for i in range (0, shape[0]):
        for j in range(0, shape[1]):
            for k in range(Z[0], shape[2]):
                landmarks_template[i][j][k] = 0

    for i in range (0, shape[0]):
        for j in range(0, shape[1]):
            for k in range(0, Z[1]):
                landmarks_template[i][j][k] = 0

else:

    for i in range (0, shape[0]):
        for j in range(0, shape[1]):
            for k in range(Z[1], shape[2]):
                landmarks_template[i][j][k] = 0

    for i in range (0, shape[0]):
        for j in range(0, shape[1]):
            for k in range(0, Z[0]):
                landmarks_template[i][j][k] = 0

hdr = anat_file_nibabel.get_header()
img = Nifti1Image(landmarks_template, None, hdr)
save(img, template_file_masked + '.nii.gz')

cmd = 'isct_c3d ' + template_file + ext + ' ' + template_file_masked + ext + ' -copy-transform -o ' + template_file_masked + ext
print('>> '+ cmd)
status, PWD = getstatusoutput(cmd)

# Registration computation with ANTS
cmd = 'ants 3 -o ' + prefix_ants + ' ' + '-m PSE[' + landmarks_template_file + ext + ',' + landmarks_anat_file_reorient + ext + ',' + landmarks_template_file + ext + ',' + landmarks_anat_file_reorient + ext + ',0.3,100,11,0,10,1000] ' + '-m MSQ[' + landmarks_template_file + ext + ',' + landmarks_anat_file_reorient + ext + ',0.3,0] ' + '-m MI[' + template_file_masked + ',' + anat_file_reorient_masked + ',0.4,32] ' + '--use-all-metrics-for-convergence 1 --use-Histogram-Matching 1 ' + '-t SyN[0.2] -r Gauss[0.5,0.5] -i 100x50x30 ' + '--rigid-affine false --affine-metric-type MI --number-of-affine-iterations 1000x1000x1000'
print('>> '+ cmd)
status, PWD = getstatusoutput(cmd)

# Applying the transform
cmd = 'WarpImageMultiTransform 3 ' + anat_file_reorient_masked + ext + ' ' + anat_file_reorient_masked + suffix_ants + ext + ' ' + prefix_ants + 'Warp.nii.gz ' + prefix_ants + 'Affine.txt -R ' + template_file + ext
print('>> '+ cmd)
status, PWD = getstatusoutput(cmd)

cmd = 'WarpImageMultiTransform 3 ' + anat_file_reorient + ext + ' ' + anat_file_reorient + suffix_ants + ext + ' ' + prefix_ants + 'Warp.nii.gz ' + prefix_ants + 'Affine.txt -R ' + template_file + ext
print('>> '+ cmd)
status, PWD = getstatusoutput(cmd)








