from nibabel import load


def find_missing_items(int_list):
    '''
    Finds missing integer within an unsorted list and return a list of
    missing items

    >>> find_missing_items([1, 2, 5, 6, 7, 10])
    [3, 4, 8, 9]

    >>> find_missing_items([3, 1, 2])
    []
    '''

    # Put the list in a set, find smallest and largest items
    original_set  = set(int_list)
    smallest_item = min(original_set)
    largest_item  = max(original_set)

    # Create a super set of all items from smallest to largest
    full_set = set(xrange(smallest_item, largest_item + 1))

    # Missing items are the ones that are in the full_set, but not in
    # the original_set
    return sorted(list(full_set - original_set))


FILE_VOLUME = '/home/django/gleveque/data/spinalcordtoolbox_dev_testing/scripts/isct_ANTSUseLandmarkImagesToGetBSplineDisplacementField/ortho_rigid.nii'

HORIZONTAL_LANDMARKS = '/home/django/gleveque/data/spinalcordtoolbox_dev_testing/scripts/errsm_24_t2_cropped_APRLIS_horizontal_landmarks.nii.gz'

# read nifti input file
img = load(FILE_VOLUME)
# 3d array for each x y z voxel values for the input nifti image
data = img.get_data()
shape = data.shape
print "Input volume dim.:"
print shape

X, Y, Z = (data > 0).nonzero()

a = len(X)

# for all points with non-zeros neighbors, force the neighbors to 0
for i in range(0,a):

    if data[X[i-1]][Y[i-1]][Z[i]] != 0:
        data[X[i-1]][Y[i-1]][Z[i]] = 0
    if data[X[i-1]][Y[i-1]][Z[i-1]] != 0:
        data[X[i-1]][Y[i-1]][Z[i-1]] = 0
    if data[X[i-1]][Y[i-1]][Z[i+1]] != 0:
        data[X[i-1]][Y[i-1]][Z[i+1]] = 0

    if data[X[i-1]][Y[i]][Z[i]] != 0:
        data[X[i-1]][Y[i]][Z[i]] = 0
    if data[X[i-1]][Y[i]][Z[i-1]] != 0:
        data[X[i-1]][Y[i]][Z[i-1]] = 0
    if data[X[i-1]][Y[i]][Z[i+1]] != 0:
        data[X[i-1]][Y[i]][Z[i+1]] = 0

    if data[X[i-1]][Y[i+1]][Z[i]] != 0:
        data[X[i-1]][Y[i+1]][Z[i]] = 0
    if data[X[i-1]][Y[i+1]][Z[i-1]] != 0:
        data[X[i-1]][Y[i+1]][Z[i-1]] = 0
    if data[X[i-1]][Y[i+1]][Z[i+1]] != 0:
        data[X[i-1]][Y[i+1]][Z[i+1]] = 0

    if data[X[i]][Y[i-1]][Z[i]] != 0:
        data[X[i]][Y[i-1]][Z[i]] = 0
    if data[X[i]][Y[i-1]][Z[i-1]] != 0:
        data[X[i]][Y[i-1]][Z[i-1]] = 0
    if data[X[i]][Y[i-1]][Z[i+1]] != 0:
        data[X[i]][Y[i-1]][Z[i+1]] = 0

    if data[X[i]][Y[i]][Z[i-1]] != 0:
        data[X[i]][Y[i]][Z[i-1]] = 0
    if data[X[i]][Y[i]][Z[i+1]] != 0:
        data[X[i]][Y[i]][Z[i+1]] = 0

    if data[X[i]][Y[i+1]][Z[i]] != 0:
        data[X[i]][Y[i+1]][Z[i]] = 0
    if data[X[i]][Y[i+1]][Z[i-1]] != 0:
        data[X[i]][Y[i+1]][Z[i-1]] = 0
    if data[X[i]][Y[i+1]][Z[i+1]] != 0:
        data[X[i]][Y[i+1]][Z[i+1]] = 0

    if data[X[i-1]][Y[i-1]][Z[i]] != 0:
        data[X[i-1]][Y[i-1]][Z[i]] = 0
    if data[X[i-1]][Y[i-1]][Z[i-1]] != 0:
        data[X[i-1]][Y[i-1]][Z[i-1]] = 0
    if data[X[i-1]][Y[i-1]][Z[i+1]] != 0:
        data[X[i-1]][Y[i-1]][Z[i+1]] = 0

    if data[X[i+1]][Y[i]][Z[i]] != 0:
        data[X[i+1]][Y[i]][Z[i]] = 0
    if data[X[i+1]][Y[i]][Z[i-1]] != 0:
        data[X[i+1]][Y[i]][Z[i-1]] = 0
    if data[X[i+1]][Y[i]][Z[i+1]] != 0:
        data[X[i+1]][Y[i]][Z[i+1]] = 0

    if data[X[i+1]][Y[i+1]][Z[i]] != 0:
        data[X[i+1]][Y[i+1]][Z[i]] = 0
    if data[X[i+1]][Y[i+1]][Z[i-1]] != 0:
        data[X[i+1]][Y[i+1]][Z[i-1]] = 0
    if data[X[i+1]][Y[i+1]][Z[i+1]] != 0:
        data[X[i+1]][Y[i+1]][Z[i+1]] = 0

# verify if orthogonal landmarks have disappeared and if so, remove the horizintal landmarks with corresponding missing value

# create an array of landmarks values to check if some values are not present
values = [0 for x in xrange(0,a)]
for i in range(0,a):
    values[i] = data[X[i]][Y[i]][Z[i]]

missing_values = find_missing_items(values)


if (len(missing_values) != 0):
    # read nifti input file
    img = load(HORIZONTAL_LANDMARKS)
    # 3d array for each x y z voxel values for the input nifti image
    data = img.get_data()
    shape = data.shape
    print "Input volume dim.:"
    print shape

    X, Y, Z = (data > 0).nonzero()

    a = len(X)
    b = len(missing_values)

    for i in range(0,a):
        for j in range(0,b):
            if data[X[i]][Y[i]][Z[i]] == missing_values[j]:
                data[X[i]][Y[i]][Z[i]] = 0


