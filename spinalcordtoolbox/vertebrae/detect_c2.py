# Author: Lucas
# Copyright (c) 2018 Polytechnique Montreal <www.neuro.polymtl.ca>
# About the license: see the file LICENSE.TXT
import argparse
import nibabel as nib
import numpy as np
import os
import scripts.sct_utils as sct
import sys
import torch
from spinalcordtoolbox.cropping import ImageCropper, BoundingBox
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.utils import Metavar, SmartFormatter
<<<<<<< Updated upstream
from spinalcordtoolbox.vertebrae.models import *
=======
import scripts.sct_utils as sct

import torch
from spinalcordtoolbox.vertebrae.models_c2 import *
>>>>>>> Stashed changes
from spinalcordtoolbox.vertebrae.predict_utils import *


def get_parser():
    # Mandatory arguments
    parser = argparse.ArgumentParser(
        description="tools to detect C2/C3 intervertebral disc with countception deep learning network ",
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
        '-o',
        help="output name",
        metavar=Metavar.str,
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
    im_input = Image(arguments.i)
    contrast = arguments.c

    global cuda_available
    cuda_available = torch.cuda.is_available()

    if arguments.net == 'CC':
        model = ModelCountception_v3(inplanes=1, outplanes=1)

        if contrast == 't1':
            model.load_state_dict(torch.load(
                os.path.join(sct.__sct_dir__, 'spinalcordtoolbox/vertebrae/checkpoints/Countception_C2T1v3.model'),
                map_location='cpu')['model_weights'])

        elif contrast == 't2':
            model.load_state_dict(torch.load(
                os.path.join(sct.__sct_dir__,'spinalcordtoolbox/vertebrae/checkpoints/Countception_C2T2v3.model'),
                map_location='cpu')['model_weights'])

    elif arguments.net == 'AttU':
        model = AttU_Net()
        if contrast == 't1':
            model.load_state_dict(torch.load(
                '~/luroub_local/lurou_local/deep_VL_2019/ivado_med/scripts_vertebral_labeling/checkpoints/attunet_c2T1.model',
                map_location='cpu')['model_weights'])

        elif contrast == 't2':
            model.load_state_dict(torch.load(
                os.path.join(sct.__sct_dir__, 'spinalcordtoolbox/vertebrae/checkpoints/AttU_curveC2T2.model'),
                map_location='cpu')['model_weights'])

    else:
        sct.printv('Error...unknown contrast. please select between t2 and t1.')
        return 100

    if cuda_available:
        model = model.cuda()
    model = model.float()

    sct.printv('retrieving input...')
    im_input.change_orientation('RPI')
    arr = np.array(im_input.data)
    # debugging
    im_shape = arr.shape
    ind = int(np.round(arr.shape[0] / 2))
    inp = np.mean(arr[ind - 3:ind + 3, :, :], 0)
   # pad = int(np.ceil(arr.shape[2] / 32)) * 32
   # xpad = int(np.ceil(arr.shape[1] / 32)) * 32 
   # img_tmp = np.zeros((xpad, pad), dtype=np.float64)
   # img_tmp[0:inp.shape[0], 0:inp.shape[1]] = inp
    inp = np.expand_dims(inp, -1)
    sct.printv('Predicting coordinate')

    coord = prediction_coordinates(inp, model, aim='c2', heatmap=0, cud_test=cuda_available)
    mask_out = np.zeros(arr.shape)
    if len(coord) < 1 or coord == [0, 0]:
        sct.printv('C2/C3 detection failed. Please provide manual initialisation')
        return (100)

    x = coord
    print(x)
    if len(x)==1:
        print(imsh)
        if int(x[0][1]) < imsh[1] and int(x[0][0]) < imsh[2]:
           mask_out[ind, x[0][1], x[0][0]] = 3
    sct.printv('saving image')
    im_shape = arr.shape
    to_save = Image(param=[im_shape[0], im_shape[1], im_shape[2]], hdr=im_input.header)
    to_save.data = mask_out
    if arguments.o is not None:
        to_save.save(arguments.o)
    else:
        to_save.save('labels_detect.nii')
    return (0)


if __name__ == "__main__":
    main()
