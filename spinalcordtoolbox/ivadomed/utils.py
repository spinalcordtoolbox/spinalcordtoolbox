import os
import sys
import subprocess
import hashlib
import datetime
import platform

import numpy as np
import wandb
from enum import Enum
from loguru import logger
from pathlib import Path
from ivadomed.keywords import ConfigKW, LoaderParamsKW, WandbKW
from typing import List
from difflib import SequenceMatcher

AXIS_DCT = {'sagittal': 0, 'coronal': 1, 'axial': 2}

# List of classification models (ie not segmentation output)
CLASSIFIER_LIST = ['resnet18', 'densenet121']


class Metavar(Enum):
    """This class is used to display intuitive input types via the metavar field of argparse."""

    file = "<file>"
    str = "<str>"
    folder = "<folder>"
    int = "<int>"
    list = "<list>"
    float = "<float>"

    def __str__(self):
        return self.value


def initialize_wandb(wandb_params):
    """Initializes WandB and based upon the parameters sets it up or disables it for experimental tracking
    
    Args:
        wandb_params (dict): wandb parameters
        
    Returns:
        bool, wandb_tracking: True if wandb tracking is enabled        
    """
    try:
        # raise an error if the key is empty
        if not bool(wandb_params[WandbKW.WANDB_API_KEY].strip()):
            raise ValueError()

        # Log on to WandB (assuming that the API Key is correct)
        # if not, login would raise an exception for the cases invalid API key and not found
        wandb.login(key=wandb_params[WandbKW.WANDB_API_KEY], anonymous='allow', timeout=60)

    except Exception as e:
        # log error mssg for unsuccessful wandb authentication
        if wandb_params is not None:
            logger.info("Incorrect WandB API Key! Please re-check the entered API key.")
            logger.info("Disabling WandB Tracking, continuing with Tensorboard Logging")
        else:
            logger.info("No WandB parameters found! Continuing with Tensorboard Logging")

        # set flag
        wandb_tracking = False

    else:
        # setting flag after successful authentication
        logger.info("WandB API Authentication Successful!")
        wandb_tracking = True

    return wandb_tracking


def get_task(model_name):
    return "classification" if model_name in CLASSIFIER_LIST else "segmentation"


def cuda(input_var, cuda_available=True, non_blocking=False):
    """Passes input_var to GPU.

    Args:
        input_var (Tensor): either a tensor or a list of tensors.
        cuda_available (bool): If False, then return identity
        non_blocking (bool):

    Returns:
        Tensor
    """
    if cuda_available:
        if isinstance(input_var, list):
            return [t.cuda(non_blocking=non_blocking) for t in input_var]
        else:
            return input_var.cuda(non_blocking=non_blocking)
    else:
        return input_var


def unstack_tensors(sample):
    """Unstack tensors.

    Args:
        sample (Tensor):

    Returns:
        list: list of Tensors.
    """
    list_tensor = []
    for i in range(sample.shape[1]):
        list_tensor.append(sample[:, i, ].unsqueeze(1))
    return list_tensor


def generate_sha_256(context: dict, df, file_lst: List[str]) -> None:
    """generate sha256 for a training file

    Args:
        context (dict): configuration context.
        df (pd.DataFrame): Dataframe containing all BIDS image files indexed and their metadata.
        file_lst (List[str]): list of strings containing training files
    """
    from pandas import DataFrame
    assert isinstance(df, DataFrame)

    # generating sha256 for list of data
    context[ConfigKW.TRAINING_SHA256] = {}
    # file_list is a list of filename strings
    for file in file_lst:
        # bids_df is a dataframe with column values path...filename...
        # so df_sub is the row with matching filename=file
        df_sub = df.loc[df['filename'] == file]
        file_path = df_sub['path'].values[0]
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
            context[ConfigKW.TRAINING_SHA256][file] = sha256_hash.hexdigest()


def save_onnx_model(model, inputs, model_path):
    """Convert PyTorch model to ONNX model and save it as `model_path`.

    Args:
        model (nn.Module): PyTorch model.
        inputs (Tensor): Tensor, used to inform shape and axes.
        model_path (str): Output filename for the ONNX model.
    """
    import torch
    model.eval()
    dynamic_axes = {0: 'batch', 1: 'num_channels', 2: 'height', 3: 'width', 4: 'depth'}
    if len(inputs.shape) == 4:
        del dynamic_axes[4]
    torch.onnx.export(model, inputs, model_path,
                      opset_version=11,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': dynamic_axes, 'output': dynamic_axes})


def define_device(gpu_id):
    """Define the device used for the process of interest.

    Args:
        gpu_id (int): GPU ID.

    Returns:
        Bool, device: True if cuda is available.
    """
    import torch
    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    if not cuda_available:
        logger.info("Cuda is not available.")
        logger.info("Working on {}.".format(device))
    if cuda_available:
        # Set the GPU
        gpu_id = int(gpu_id)
        torch.cuda.set_device(gpu_id)
        logger.info(f"Using GPU ID {gpu_id}")
    return cuda_available, device


def display_selected_model_spec(params):
    """Display in terminal the selected model and its parameters.

    Args:
        params (dict): Keys are param names and values are param values.
    """
    logger.info('Selected architecture: {}, with the following parameters:'.format(params["name"]))
    for k in list(params.keys()):
        if k != "name":
            logger.info('\t{}: {}'.format(k, params[k]))


def display_selected_transfoms(params, dataset_type):
    """Display in terminal the selected transforms for a given dataset.

    Args:
        params (dict):
        dataset_type (list): e.g. ['testing'] or ['training', 'validation']
    """
    logger.info('Selected transformations for the {} dataset:'.format(dataset_type))
    for k in list(params.keys()):
        logger.info('\t{}: {}'.format(k, params[k]))


def plot_transformed_sample(before, after, list_title=None, fname_out="", cmap="jet"):
    """Utils tool to plot sample before and after transform, for debugging.

    Args:
        before (ndarray): Sample before transform.
        after (ndarray): Sample after transform.
        list_title (list of str): Sub titles of before and after, resp.
        fname_out (str): Output filename where the plot is saved if provided.
        cmap (str): Matplotlib colour map.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    if list_title is None:
        list_title = []
    if len(list_title) == 0:
        list_title = ['Sample before transform', 'Sample after transform']

    plt.interactive(False)
    plt.rcParams.update({'figure.max_open_warning': 0})
    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.imshow(before, interpolation='nearest', cmap=cmap)
    plt.title(list_title[0], fontsize=20)

    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.imshow(after, interpolation='nearest', cmap=cmap)
    plt.title(list_title[1], fontsize=20)

    if fname_out:
        plt.savefig(fname_out)
    else:
        plt.show()


def _git_info(commit_env='IVADOMED_COMMIT', branch_env='IVADOMED_BRANCH'):
    """Get ivadomed version info from GIT.

    This functions retrieves the ivadomed version, commit, branch and installation type.

    Args:
        commit_env (str):
        branch_env (str):
    Returns:
        str, str, str, str: installation type, commit, branch, version.
    """
    ivadomed_commit = os.getenv(commit_env, "unknown")
    ivadomed_branch = os.getenv(branch_env, "unknown")
    if check_exe("git") and Path(__ivadomed_dir__, ".git").is_dir():
        ivadomed_commit = __get_commit() or ivadomed_commit
        ivadomed_branch = __get_branch() or ivadomed_branch

    if ivadomed_commit != 'unknown':
        install_type = 'git'
    else:
        install_type = 'package'

    path_version = Path(__ivadomed_dir__, 'ivadomed', 'version.txt')
    with path_version.open() as f:
        version_ivadomed = f.read().strip()

    return install_type, ivadomed_commit, ivadomed_branch, version_ivadomed


def check_exe(name):
    """Ensure that a program exists.

    Args:
        name (str): Name or path to program.
    Returns:
        str or None: path of the program or None
    """

    def is_exe(fpath):
        return Path(fpath).is_file() and os.access(fpath, os.X_OK)

    fpath = Path(name).parent
    if fpath and is_exe(name):
        return fpath
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = str(Path(path, name))
            if is_exe(exe_file):
                return exe_file

    return None


class ArgParseException(Exception):
    pass


def get_arguments(parser, args):
    """Get arguments from function input or command line.

    Arguments:
        parser (argparse.ArgumentParser): ArgumentParser object
        args (list): either a list of arguments or None. The list
            should be formatted like this:
            ["-d", "SOME_ARG", "--model", "SOME_ARG"]
    """
    try:
        args = parser.parse_args(args)
    except SystemExit as e:
        if e.code != 0:  # Calling `--help` raises SystemExit with 0 exit code (i.e. not an ArgParseException)
            raise ArgParseException('Error parsing args')
        else:
            sys.exit(0)

    return args


def __get_commit(path_to_git_folder=None):
    """Get GIT ivadomed commit.

    Args:
        path_to_git_folder (str): Path to GIT folder.
    Returns:
        str: git commit ID, with trailing '*' if modified.
    """
    if path_to_git_folder is None:
        path_to_git_folder = __ivadomed_dir__
    else:
        path_to_git_folder = Path(path_to_git_folder).expanduser().absolute()

    p = subprocess.Popen(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=path_to_git_folder)
    output, _ = p.communicate()
    status = p.returncode
    if status == 0:
        commit = output.decode().strip()
    else:
        commit = "?!?"

    p = subprocess.Popen(["git", "status", "--porcelain"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         cwd=path_to_git_folder)
    output, _ = p.communicate()
    status = p.returncode
    if status == 0:
        unclean = True
        for line in output.decode().strip().splitlines():
            line = line.rstrip()
            if line.startswith("??"):  # ignore ignored files, they can't hurt
                continue
            break
        else:
            unclean = False
        if unclean:
            commit += "*"

    return commit


def __get_branch():
    """Get ivadomed branch.

    Args:

    Returns:
        str: ivadomed branch.
    """
    p = subprocess.Popen(["git", "rev-parse", "--abbrev-ref", "HEAD"], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, cwd=__ivadomed_dir__)
    output, _ = p.communicate()
    status = p.returncode

    if status == 0:
        return output.decode().strip()


def _version_string():
    install_type, ivadomed_commit, ivadomed_branch, version_ivadomed = _git_info()
    if install_type == "package":
        return version_ivadomed
    else:
        return "{install_type}-{ivadomed_branch}-{ivadomed_commit}".format(**locals())


__ivadomed_dir__ = Path(__file__).resolve().parent.parent
__version__ = _version_string()


def get_command(args, context):
    if args.train:
        return "train"
    elif args.test:
        return "test"
    elif args.segment:
        return "segment"
    else:
        logger.info(
            "No CLI argument given for command: ( --train | --test | --segment ). Will check config file for command...")

        try:
            if context[ConfigKW.COMMAND] == "train" or context[ConfigKW.COMMAND] == "test" or context[
                ConfigKW.COMMAND] == "segment":
                return context[ConfigKW.COMMAND]
            else:
                logger.error("Specified invalid command argument in config file.")
        except AttributeError:
            logger.error("Have not specified a command argument via CLI nor config file.")


def get_path_output(args, context):
    if args.path_output:
        return args.path_output
    else:
        logger.info(
            "CLI flag --path-output not used to specify output directory. Will check config file for directory...")
        try:
            if context[ConfigKW.PATH_OUTPUT]:
                return context[ConfigKW.PATH_OUTPUT]
        except AttributeError:
            logger.error("Have not specified a path-output argument via CLI nor config file.")


def get_path_data(args, context):
    if args.path_data:
        return args.path_data
    else:
        logger.info(
            "CLI flag --path-data not used to specify BIDS data directory. Will check config file for directory...")
        try:
            if context[ConfigKW.LOADER_PARAMETERS][LoaderParamsKW.PATH_DATA]:
                return context[ConfigKW.LOADER_PARAMETERS][LoaderParamsKW.PATH_DATA]
        except AttributeError:
            logger.error("Have not specified a path-data argument via CLI nor config file.")


def format_path_data(path_data):
    """
    Args:
        path_data (list or str): Either a list of paths, or just one path.

    Returns:
        list: A list of paths
    """
    assert isinstance(path_data, str) or isinstance(path_data, list)
    if isinstance(path_data, str):
        path_data = [path_data]
    return path_data


def similarity_score(a: str, b: str) -> float:
    """
    use DiffLIb SequenceMatcher to resolve the similarity between text. Help make better choice in terms of derivatives.
    Args:
        a: a string
        b: another string
    Returns: a score indicative of the similarity between the sequence.
    """
    return SequenceMatcher(None, a, b).ratio()


def init_ivadomed():
    """Initialize the ivadomed for typical terminal usage."""
    # Display ivadomed version
    logger.info('\nivadomed ({})\n'.format(__version__))


def print_stats(arr):
    logger.info(f"\tMean: {np.mean(arr)} %")
    logger.info(f"\tMedian: {np.median(arr)} %")
    logger.info(f"\tInter-quartile range: [{np.percentile(arr, 25)}, {np.percentile(arr, 75)}] %")


def get_timestamp() -> str:
    """
    Return a datetime string in the format YYYY-MM-DDTHHMMSS.(sub-precision)
    Returns:
    """
    timestamp = datetime.datetime.now().isoformat().replace(":", "")
    return timestamp


def get_system_memory() -> float:
    """
    Return the system memory in GB.
    Returns:
    """
    current_platform = platform.system()
    if current_platform == "Linux":
        return get_linux_system_memory()
    elif current_platform == "Windows":
        return get_win_system_memory()
    elif current_platform == "Darwin":
        return get_mac_system_memory()


def get_win_system_memory() -> float:
    """
    Obtain the amount of memory available on Windows system.
    Returns: memory in GB
    Source: https://stackoverflow.com/a/21589439
    """
    process = os.popen('wmic memorychip get capacity')
    result = process.read()
    process.close()
    totalMem = 0
    for m in result.split("  \n\n")[1:-1]:
        totalMem += int(m)
    return totalMem / (1024 ** 3)


def get_linux_system_memory() -> float:
    """
    Obtain the amount of memory available on Linux system.
    Returns: memory in GB
    Source: https://stackoverflow.com/a/28161352
    """
    import os
    mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')  # e.g. 4015976448
    mem_gib = mem_bytes / (1024. ** 3)  # e.g. 3.74
    return mem_gib


def get_mac_system_memory() -> float:
    """
    Obtain the amount of memory available on MacOS system.
    Returns: memory in GB
    Source: https://apple.stackexchange.com/a/4296
    """

    import subprocess
    import re

    # Get process info
    ps = subprocess.Popen(['ps', '-caxm', '-orss,comm'], stdout=subprocess.PIPE).communicate()[0].decode()

    # Iterate processes
    processLines = ps.split('\n')
    sep = re.compile('[\s]+')
    rssTotal = 0  # kB
    for row in range(1, len(processLines)):
        rowText = processLines[row].strip()
        rowElements = sep.split(rowText)
        try:
            rss = float(rowElements[0]) * 1024
        except:
            rss = 0  # ignore...
        rssTotal += rss

    return rssTotal / 1024 ** 3
