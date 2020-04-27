import sys
import os
import argparse


from spinalcordtoolbox.cropping import ImageCropper, BoundingBox
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils import Metavar, SmartFormatter
import sct_utils as sct
print('detect1')
import torch
print('detect2')
from spinalcordtoolbox.vertebrae.models import *
print('detect3')
from spinalcordtoolbox.vertebrae.Predict_utils import *
import numpy as np

import nibabel as nib
print('detect1')

def get_parser():
    # Mandatory arguments
    parser = argparse.ArgumentParser(
        description="tools to detect and label vertebrae with countception deep learning network ",
        epilog="EXAMPLES:\n",
        add_help=None,
        formatter_class=SmartFormatter,
        prog=os.path.basename(__file__).strip('.py'))

    mandatoryArguments = parser.add_argument_group("\nMANDATORY ARGUMENTS")
    mandatoryArguments.add_argument(
        '-i',
        required=True,
        help="Input image. Example: t2.nii.gz",
        metavar=Metavar.file,
    )
    mandatoryArguments.add_argument(
        '-c',
        required=True,
        help="contrast",
    )
    optional = parser.add_argument_group("\nOPTIONAL ARGUMENTS")
    optional.add_argument(
        '-h',
        '--help',
        action='help',
        help="Show this help message and exit")

    optional.add_argument(
        '-t',
        help="threshold",
        metavar=Metavar.str,
    )
    optional.add_argument(
        '-o',
        help="name",
        metavar=Metavar.file,
    )
    optional.add_argument(
        '-m',
        help="save nii heatmap if 1, label if 0",
        default=0,
    )
    optional.add_argument(
        '-net',
        help="Network to use",
        default='CC',
    )


    return parser


def main(args=None):
    """
    Main function
    :param args:
    :return:
    """
    # get parser args
    if args is None:
        args = None if sys.argv[1:] else ['--help']
    parser = get_parser()
    arguments = parser.parse_args(args=args)
    Im_input = Image(arguments.i)
    contrast = arguments.c
    heatmap = int(arguments.m)
    global cuda_available
    cuda_available = torch.cuda.is_available()
    if arguments.net == 'CC':

        model = ModelCountception_v2(inplanes=1, outplanes=1)

        if contrast == 't1':
            model.load_state_dict(torch.load('./checkpoints/Countception_L2T1.model', map_location='cpu')['model_weights'])

        elif contrast == 't2':
            model.load_state_dict(torch.load('./checkpoints/Countception_floatL2T2.model', map_location='cpu')['model_weights'])

        else:
            sct.printv('Error...unknown contrast. please select between t2 and t1.')
            return 100
    if arguments.net == 'AttU':
        model = AttU_Net()

        if contrast == 't1':
            model.load_state_dict(torch.load('/home/GRAMES.POLYMTL.CA/luroub/luroub_local/lurou_local/deep_VL_2019/ivado_med/scripts_vertebral_labeling/checkpoints/Countception_L2T1.model', map_location='cpu')['model_weights'])

        elif contrast == 't2':
            model.load_state_dict(torch.load('../checkpoints/attunet_fullT2.model', map_location='cpu')['model_weights'])

        else:
            sct.printv('Error...unknown contrast. please select between t2 and t1.')
            return 100

    if cuda_available:
        model = model.cuda()
    model = model.float()

    sct.printv('retrieving input...')
    Im_input.change_orientation('RPI')
    arr = np.array(Im_input.data)
    # debugging

    ind = int(np.round(arr.shape[0] / 2))
    inp = np.mean(arr[ind - 2:ind + 2, :, :], 0)
    pad = int(np.ceil(arr.shape[2] / 32)) * 32
    img_tmp = np.zeros((160, pad), dtype=np.float64)
    img_tmp[0:inp.shape[0], 0:inp.shape[1]] = inp
    inp = np.expand_dims(img_tmp, -1)

    sct.printv('Predicting coordinate')

    coord = prediction_coordinates(inp, model, aim='full', heatmap=heatmap, cud_test=cuda_available)

    if heatmap == 1:
        sct.printv('saving heatmap')
        imsh = arr.shape
        to_save = Image(param=[imsh[0], imsh[1], imsh[2]], hdr=Im_input.header)
        to_save.data[ind, :, :] = coord[:imsh[1], :imsh[2]]
        if arguments.o is not None:
            to_save.save(arguments.o)
        else:
            to_save.save('labels_first_try.nii')
    else:
        imsh = arr.shape
        mask_out = np.zeros(arr.shape)
        if len(coord) < 2:
            sct.printv('Error did not work at all, you can try with a different threshold')
        for x in coord:
            if int(x[1]) < imsh[1] and int(x[0]) < imsh[2]:
                mask_out[ind, int(x[1]), int(x[0])] = 10
        sct.printv('saving image')
        imsh = arr.shape
        to_save = Image(param=[imsh[0], imsh[1], imsh[2]], hdr=Im_input.header)
        to_save.data = mask_out
        if arguments.o is not None:
            to_save.save(arguments.o)
        else:
            to_save.save('labels_first_try.nii')


if __name__ == "__main__":
    main()
