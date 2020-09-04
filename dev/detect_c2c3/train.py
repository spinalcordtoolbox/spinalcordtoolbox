import os
import numpy as np
import pandas as pd
import sct_utils as sct
import matplotlib.pyplot as plt 
from matplotlib.patches import Circle
from spinalcordtoolbox.image import Image
from config_file import config


def run_optic(fname_in, contrast, ofolder):
    cmd = ['sct_get_centerline', '-i', fname_in, '-c', contrast, '-ofolder', ofolder]
    sct.run(cmd)


def run_flat(fname_in, fname_ctr, ofolder):
    cmd = ['sct_flatten_sagittal', '-i', fname_in, '-s', fname_ctr]
    try:
        sct.run(cmd)
    except:
        pass


def create_qc(fname_in, fname_gt, fname_out):
    img, gt = Image(fname_in), Image(fname_gt)
    img.change_orientation('RPI')
    gt.change_orientation('RPI')
    coord_c2c3 = np.where(gt.data == 1)
    y_c2c3, z_c2c3 = coord_c2c3[1][0], coord_c2c3[2][0]
    sag_slice = img.data[0, :, :]
    del img, gt

    ax = plt.gca()
    ax.imshow(sag_slice, interpolation='nearest', cmap='gray', aspect='auto')
    circ = Circle((z_c2c3, y_c2c3), 2, facecolor='chartreuse')
    ax.add_patch(circ)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.savefig(fname_out)
    plt.close()


def run_crop(fname_in, fname_out, nb_slice_average=1.0):
    img = Image(fname_in).change_orientation('RPI')
    if len(list(np.where(img.data == 3)[2])) == 1: # if label file
        x_start, x_end = str(np.where(img.data == 3)[0][0]), str(np.where(img.data == 3)[0][0])
        nb_slice_average_each_side = 0
        img.data[np.where(img.data != 3)] = 0
        img.data[np.where(img.data == 3)] = 1
        img.change_orientation('PIR')
        img.save(fname_out)
        del img
    else: # if grayscale image file
        x_med = int(np.rint(img.dim[0] * 1.0 / 2))
        nb_slice_average_each_side = int(nb_slice_average / 2 / img.dim[4])
        x_start, x_end = str(x_med-nb_slice_average_each_side), str(x_med+nb_slice_average_each_side)
        del img
        cmd_orient = ['sct_image', '-i', fname_in, '-setorient', 'PIR', '-o', fname_out]
        sct.run(cmd_orient)

    cmd_crop = ['sct_crop_image', '-i', fname_out, '-zmin', x_start, '-zmax', x_end, '-o', fname_out]
    sct.run(cmd_crop)

    if nb_slice_average_each_side:
        cmd_mean = ['sct_maths', '-i', fname_out, '-mean', 'z', '-o', fname_out]
        sct.run(cmd_mean)


def preprocessing(df, folder_out, contrast_centerline):
    sct.printv("Preprocessing...")

    qc_fold = os.path.join(folder_out, 'qc')
    if not os.path.isdir(qc_fold):
        os.makedirs(qc_fold)

    for idx, row in df.iterrows():
        if row.contrast.startswith(contrast_centerline):
            sct.printv("\t" + row.subject)
            img = row['img']
            labels = row['labels']
            img_head, img_tail = os.path.split(img)
            img_basename = img_tail.split('.nii')[0]

            folder_out_cur = os.path.join(folder_out, row.subject)
            if not os.path.isdir(folder_out_cur):
                os.makedirs(folder_out_cur)

            ctr = os.path.join(folder_out_cur, img_basename + '_centerline_optic.nii.gz')
            if not os.path.isfile(ctr):
                run_optic(img, contrast_centerline, folder_out_cur)

            flat_in = os.path.join(img_head, img_basename + '_flatten.nii.gz')
            flat = os.path.join(folder_out_cur, img_basename + '_flatten.nii.gz')
            if not os.path.isfile(flat) and os.path.isfile(ctr):
                run_flat(img, ctr, folder_out_cur)
                sct.mv(flat_in, flat)

            oneslice = os.path.join(folder_out_cur, img_basename + '_oneslice.nii')
            oneslice_gt = os.path.join(folder_out_cur, img_basename + '_oneslice_gt.nii')
            if os.path.isfile(flat) and not os.path.isfile(oneslice):
                run_crop(flat, oneslice, 7.0)
            if os.path.isfile(labels) and not os.path.isfile(oneslice_gt):
                run_crop(labels, oneslice_gt, 1.0)

            if os.path.isfile(oneslice) and os.path.isfile(oneslice_gt):
                df.loc[idx, 'train'] = os.path.abspath(oneslice)
                df.loc[idx, 'gt'] = os.path.abspath(oneslice_gt)

                qc_file = os.path.join(qc_fold, '_'.join([row.subject, row.contrast]) + '.png')
                if not os.path.isfile(qc_file):
                    create_qc(oneslice, oneslice_gt, qc_file)

    return df


# def train_model(df, model_name):
    sct.printv("Training...")
    train_txt = 'train_lst.txt'
    train_gt_txt = 'train_gt_lst.txt'
    if os.path.isfile(train_txt) or os.path.isfile(train_gt_txt):
        sct.rm(train_txt)
        sct.rm(train_gt_txt)

    stg_train = '\n'.join([os.path.abspath(f).split('.nii')[0] for f in df['train'].values if str(f) != 'nan'])
    stg_gt_train = '\n'.join([os.path.abspath(f).split('.nii')[0] for f in df['gt'].values if str(f) != 'nan'])
    with open(train_txt, 'w') as text_file:
        text_file.write(stg_train)
        text_file.close()
    with open(train_gt_txt, 'w') as text_file:
        text_file.write(stg_gt_train)
        text_file.close()

    model_path = os.getcwd() + '/trained_model_t1.yml'
    if os.path.isfile(model_path):
        sct.rm(model_path)
    cmd_train = 'isct_train_svm -hogsg -incr=20 ' + model_name + ' ' + train_txt + ' ' + train_gt_txt + ' --list True'
    sct.run(cmd_train, verbose=0, raise_exception=False)


def main():
    df = pd.read_pickle(config['dataframe_database'])
    folder_out = config['folder_out']
    model_name = config['model_name']
    contrast_centerline = config['contrast_centerline']

    df = preprocessing(df, folder_out, contrast_centerline)

    # train_model(df, model_name)

if __name__ == '__main__':
    main()
