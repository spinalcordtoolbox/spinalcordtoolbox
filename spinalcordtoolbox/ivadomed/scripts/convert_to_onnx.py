import argparse
import torch
from ivadomed import utils as imed_utils


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest="model", required=True, type=str,
                        help="Path to .pt model.", metavar=imed_utils.Metavar.file)
    parser.add_argument("-d", "--dimension", dest="dimension", required=True,
                        type=int, help="Input dimension (2 for 2D inputs, 3 for 3D inputs).",
                        metavar=imed_utils.Metavar.int)
    parser.add_argument("-n", "--n_channels", dest="n_channels", default=1, type=int,
                        help="Number of input channels of the model.",
                        metavar=imed_utils.Metavar.int)
    parser.add_argument("-g", "--gpu_id", dest="gpu_id", default=0, type=str,
                        help="GPU number if available.", metavar=imed_utils.Metavar.int)
    return parser


def convert_pytorch_to_onnx(model, dimension, n_channels, gpu_id=0):
    """Convert PyTorch model to ONNX.

    The integration of Deep Learning models into the clinical routine requires cpu optimized models. To export the
    PyTorch models to `ONNX <https://github.com/onnx/onnx>`_ format and to run the inference using
    `ONNX Runtime <https://github.com/microsoft/onnxruntime>`_ is a time and memory efficient way to answer this need.

    This function converts a model from PyTorch to ONNX format, with information of whether it is a 2D or 3D model
    (``-d``).

    Args:
        model (string): Model filename. Flag: ``--model``, ``-m``.
        dimension (int): Indicates whether the model is 2D or 3D. Choice between 2 or 3. Flag: ``--dimension``, ``-d``
        gpu_id (string): GPU ID, if available. Flag: ``--gpu_id``, ``-g``
    """
    if torch.cuda.is_available():
        device = "cuda:" + str(gpu_id)
    else:
        device = "cpu"

    model_net = torch.load(model, map_location=device)
    dummy_input = torch.randn(1, n_channels, 96, 96, device=device) if dimension == 2 \
                  else torch.randn(1, n_channels, 96, 96, 96, device=device)
    imed_utils.save_onnx_model(model_net, dummy_input, model.replace("pt", "onnx"))


def main(args=None):
    imed_utils.init_ivadomed()
    parser = get_parser()
    args = imed_utils.get_arguments(parser, args)
    fname_model = args.model
    dimension = int(args.dimension)
    gpu_id = str(args.gpu_id)
    n_channels = args.n_channels

    convert_pytorch_to_onnx(fname_model, dimension, n_channels, gpu_id)


if __name__ == '__main__':
    main()
