import os
import numpy as np
import pandas as pd
import sct_utils as sct
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


def run_crop(fname_in, fname_out):
    img = Image(fname_in).change_orientation('RPI')
    if len(list(np.where(img.data == 3)[2])) == 1: # if label file
        x_med = str(np.where(img.data == 3)[0][0])
        img.data[np.where(img.data != 3)] = 0
        img.data[np.where(img.data == 3)] = 1
        img.change_orientation('PIR')
        img.save(fname_out)
        del img
    else: # if grayscale image file
        x_med = str(int(np.rint(img.dim[0] * 1.0 / 2)))
        del img
        cmd_orient = ['sct_image', '-i', fname_in, '-setorient', 'PIR', '-o', fname_out]
        sct.run(cmd_orient)

    cmd_orient = ['sct_crop_image', '-i', fname_out, '-start', x_med, '-end', x_med, '-dim', '2', '-o', fname_out]
    sct.run(cmd_orient)


def preprocessing(df, folder_out, contrast_centerline):
    sct.printv("Preprocessing...")
    for idx, row in df.iterrows():
        sct.printv("\t" + row.subject)
        img = row['img']
        labels = row['label']
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
            run_crop(flat, oneslice)
        if os.path.isfile(labels) and not os.path.isfile(oneslice_gt):
            run_crop(labels, oneslice_gt)

        if os.path.isfile(oneslice) and os.path.isfile(oneslice_gt):
            df.loc[idx, 'train'] = os.path.abspath(oneslice)
            df.loc[idx, 'gt'] = os.path.abspath(oneslice_gt)

    return df


def train_model(df, model_name):
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

    train_model(df, model_name)

if __name__ == '__main__':
    main()
