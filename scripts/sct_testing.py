#!/usr/bin/env python
#
# Test major functions.
#
# In The following fields should be defined under the init() function of each test script:
#   param_test.list_fname_gt     list containing the relative file name for ground truth data. See test_sct_propseg
#
# Authors: Julien Cohen-Adad, Benjamin De Leener, Augustin Roux

# TODO: list functions to test in help (do a search in testing folder)

from __future__ import print_function, absolute_import

import sys, os, time, copy, shlex, importlib, multiprocessing, tempfile, shutil
import traceback
import signal

import numpy as np
from pandas import DataFrame

import sct_utils as sct

sys.path.append(os.path.join(sct.__sct_dir__, 'testing'))


def fs_signature(root):
    ret = dict()
    root = os.path.abspath(root)
    for cwd, dirs, files in os.walk(root):
        if cwd == os.path.abspath(tempfile.gettempdir()):
            continue
        if cwd == os.path.join(root, "testing-qc"):
            files[:] = []
            dirs[:] = []
            continue
        dirs.sort()
        files.sort()
        for file in files:
            if cwd == root:
                continue
            path = os.path.relpath(os.path.join(cwd, file), root)
            data = os.stat(path)
            ret[path] = data
    return ret


def fs_ok(sig_a, sig_b, exclude=()):
    errors = list()
    for path, data in sig_b.items():
        if path not in sig_a:
            errors.append((path, "added: {}".format(path)))
            continue
        if sig_a[path] != data:
            errors.append((path, "modified: {}".format(path)))
    errors = [ (x,y) for (x,y) in errors if not x.startswith(exclude) ]
    if errors:
        for error in errors:
            sct.printv("Error: %s", 1, type='error')
        raise RuntimeError()

# Parameters
class Param:
    def __init__(self):
        self.download = False
        self.path_data = 'sct_testing_data'  # path to the testing data
        self.path_output = None
        self.function_to_test = None
        self.remove_tmp_file = False
        self.verbose = False
        self.args = []  # list of input arguments to the function
        self.args_with_path = ''  # input arguments to the function, with path
        # self.list_fname_gt = []  # list of fname for ground truth data
        self.contrast = ''  # folder containing the data and corresponding to the contrast. Could be t2, t1, t2s, etc.
        self.output = ''  # output string
        self.results = ''  # results in Panda DataFrame
        self.redirect_stdout = True  # for debugging, set to 0. Otherwise set to 1.
        self.fname_log = None


# define nice colors
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


# PARSER
# ==========================================================================================
def get_parser():
    import argparse

    param_default = Param()

    parser = argparse.ArgumentParser(
     description="Crash and integrity testing for functions of the Spinal Cord Toolbox. Internet connection is required for downloading testing data.",
    )

    parser.add_argument("--function", "-f",
     help="Test this specific script (eg. 'sct_propseg').",
     nargs="+",
    )

    def arg_jobs(s):
        jobs = int(s)
        if jobs > 0:
            pass
        elif jobs == 0:
            jobs = multiprocessing.cpu_count()
        else:
            raise ValueError()
        return jobs

    parser.add_argument("--download", "-d",
     action="store_true",
     default=param_default.download,
    )
    parser.add_argument("--path", "-p",
     help='Path to testing data. NB: no need to set if using "-d 1"',
     default=param_default.path_data,
    )
    parser.add_argument("--remove-temps", "-r",
     help='Remove temporary files.',
     action="store_true",
     default=param_default.remove_tmp_file,
    )
    parser.add_argument("--jobs", "-j",
     type=arg_jobs,
     help="# of simultaneous tests to run (jobs). 0 or unspecified means # of available CPU threads ({})".format(multiprocessing.cpu_count()),
     default=arg_jobs(0),
    )
    parser.add_argument("--verbose", "-v",
     action="store_true",
     default=param_default.verbose,
    )
    parser.add_argument("--abort-on-failure",
     help="Instead of iterating through all tests, abort at the first one that would fail.",
     action="store_true",
    )
    parser.add_argument("--continue-from",
     help="Instead of running all tests (or those specified by --function, start from this one",
    )
    parser.add_argument("--check-filesystem",
     help="Check filesystem for unwanted modifications",
     action="store_true",
    )
    parser.add_argument("--execution-folder",
     help="Folder where to run tests from (default. temporary)",
    )

    return parser


def process_function(fname, param):
    """
    """
    param.function_to_test = fname
    # display script name
    # load modules of function to test
    module_testing = importlib.import_module('test_' + fname)
    # initialize default parameters of function to test
    param.args = []
    # param.list_fname_gt = []
    # param.fname_groundtruth = ''
    param = module_testing.init(param)
    # loop over parameters to test
    list_status_function = []
    list_output = []
    for i in range(0, len(param.args)):
        param_test = copy.deepcopy(param)
        param_test.default_args = param.args
        param_test.args = param.args[i]
        param_test.test_integrity = True
        # if list_fname_gt is not empty, assign it
        # if param_test.list_fname_gt:
        #     param_test.fname_gt = param_test.list_fname_gt[i]
        # test function
        try:
            param_test = test_function(param_test)
        except sct.RunError as e:
            list_status_function.append(1)
            list_output.append("Got SCT exception:")
            list_output.append(e.args[0])
        except Exception as e:
            list_status_function.append(1)
            list_output.append("Got exception: %s" % e)
            list_output += traceback.format_exc().splitlines()
        else:
            list_status_function.append(param_test.status)
            list_output.append(param_test.output)

    return list_output, list_status_function


def process_function_multiproc(fname, param):
    """ Wrapper that makes ^C work in multiprocessing code """
    # Ignore SIGINT, parent will take care of the clean-up
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    return process_function(fname, param)


# Main
# ==========================================================================================
def main(args=None):

    # initializations
    param = Param()

    # check user arguments
    if args is None:
        args = sys.argv[1:]

    # get parser info
    parser = get_parser()

    arguments = parser.parse_args(args)

    param.download = int(arguments.download)
    param.path_data = arguments.path
    functions_to_test = arguments.function
    param.remove_tmp_file = int(arguments.remove_temps)
    jobs = arguments.jobs

    param.verbose = arguments.verbose
    sct.init_sct(log_level=param.verbose, update=True)  # Update log level

    start_time = time.time()

    # get absolute path and add slash at the end
    param.path_data = os.path.abspath(param.path_data)

    # check existence of testing data folder
    if not os.path.isdir(param.path_data) or param.download:
        downloaddata(param)

    # display path to data
    sct.printv('\nPath to testing data: ' + param.path_data, param.verbose)

    # create temp folder that will have all results
    path_tmp = os.path.abspath(arguments.execution_folder or sct.tmp_create(verbose=param.verbose))

    # go in path data (where all scripts will be run)
    curdir = os.getcwd()
    os.chdir(param.path_data)

    functions_parallel = list()
    functions_serial = list()
    if functions_to_test:
        for f in functions_to_test:
            if f in get_functions_parallelizable():
                functions_parallel.append(f)
            elif f in get_functions_nonparallelizable():
                functions_serial.append(f)
            else:
                sct.printv('Command-line usage error: Function "%s" is not part of the list of testing functions' % f, type='error')
        jobs = min(jobs, len(functions_parallel))
    else:
        functions_parallel = get_functions_parallelizable()
        functions_serial = get_functions_nonparallelizable()

    if arguments.continue_from:
        first_func = arguments.continue_from
        if first_func in functions_parallel:
            functions_serial = []
            functions_parallel = functions_parallel[functions_parallel.index(first_func):]
        elif first_func in functions_serial:
            functions_serial = functions_serial[functions_serial.index(first_func):]

    if arguments.check_filesystem and jobs != 1:
        print("Check filesystem used -> jobs forced to 1")
        jobs = 1

    print("Will run through the following tests:")
    if functions_serial:
        print("- sequentially: {}".format(" ".join(functions_serial)))
    if functions_parallel:
        print("- in parallel with {} jobs: {}".format(jobs, " ".join(functions_parallel)))

    list_status = []
    for name, functions in (
      ("serial", functions_serial),
      ("parallel", functions_parallel),
     ):
        if not functions:
            continue

        if any([s for (f, s) in list_status]) and arguments.abort_on_failure:
            break

        try:
            if functions == functions_parallel and jobs != 1:
                pool = multiprocessing.Pool(processes=jobs)

                results = list()
                # loop across functions and run tests
                for f in functions:
                    func_param = copy.deepcopy(param)
                    func_param.path_output = f
                    res = pool.apply_async(process_function_multiproc, (f, func_param,))
                    results.append(res)
            else:
                pool = None

            for idx_function, f in enumerate(functions):
                print_line('Checking ' + f)
                if functions == functions_serial or jobs == 1:
                    if arguments.check_filesystem:
                        if os.path.exists(os.path.join(path_tmp, f)):
                            shutil.rmtree(os.path.join(path_tmp, f))
                        sig_0 = fs_signature(path_tmp)

                    func_param = copy.deepcopy(param)
                    func_param.path_output = f

                    res = process_function(f, func_param)

                    if arguments.check_filesystem:
                        sig_1 = fs_signature(path_tmp)
                        fs_ok(sig_0, sig_1, exclude=(f,))
                else:
                    res = results[idx_function].get()

                list_output, list_status_function = res
                # manage status
                if any(list_status_function):
                    if 1 in list_status_function:
                        print_fail()
                        status = (f, 1)
                    else:
                        print_warning()
                        status = (f, 99)
                    for output in list_output:
                        for line in output.splitlines():
                            print("   %s" % line)
                else:
                    print_ok()
                    if param.verbose:
                        for output in list_output:
                            for line in output.splitlines():
                                print("   %s" % line)
                    status = (f, 0)
                # append status function to global list of status
                list_status.append(status)
                if any([s for (f, s) in list_status]) and arguments.abort_on_failure:
                    break
        except KeyboardInterrupt:
            raise
        finally:
            if pool:
                pool.terminate()
                pool.join()

    print('status: ' + str([s for (f, s) in list_status]))
    if any([s for (f, s) in list_status]):
        print("Failures: {}".format(" ".join([f for (f, s) in list_status if s])))

    # display elapsed time
    elapsed_time = time.time() - start_time
    sct.printv('Finished! Elapsed time: ' + str(int(np.round(elapsed_time))) + 's\n')

    # come back
    os.chdir(curdir)

    # remove temp files
    if param.remove_tmp_file and arguments.execution_folder is None:
        sct.printv('\nRemove temporary files...', 0)
        sct.rmtree(path_tmp)

    e = 0
    if any([s for (f, s) in list_status]):
        e = 1
    # print(e)

    sys.exit(e)


def downloaddata(param):
    """
    Download testing data from internet.
    Parameters
    ----------
    param

    Returns
    -------
    None
    """
    sct.printv('\nDownloading testing data...', param.verbose)
    import sct_download_data
    sct_download_data.main(['-d', 'sct_testing_data'])


def get_functions_nonparallelizable():
    return [
        'sct_deepseg_gm',
        'sct_deepseg_lesion',
        'sct_deepseg_sc',
    ]

def get_functions_parallelizable():
    return [
        'sct_analyze_lesion',
        'sct_analyze_texture',
        'sct_apply_transfo',
        'sct_convert',
        'sct_compute_ernst_angle',
        'sct_compute_hausdorff_distance',
        'sct_compute_mtr',
        'sct_compute_mscc',
        'sct_compute_snr',
        'sct_concat_transfo',
        # 'sct_convert_binary_to_trilinear',  # not useful
        'sct_create_mask',
        'sct_crop_image',
        'sct_dice_coefficient',
        'sct_detect_pmj',
        'sct_dmri_compute_dti',
        'sct_dmri_concat_b0_and_dwi',
        'sct_dmri_concat_bvals',
        'sct_dmri_concat_bvecs',
        'sct_dmri_create_noisemask',
        'sct_dmri_compute_bvalue',
        'sct_dmri_moco',
        'sct_dmri_separate_b0_and_dwi',
        'sct_dmri_transpose_bvecs',
        'sct_extract_metric',
        'sct_flatten_sagittal',
        'sct_fmri_compute_tsnr',
        'sct_fmri_moco',
        'sct_get_centerline',
        'sct_image',
        'sct_label_utils',
        'sct_label_vertebrae',
        'sct_maths',
        'sct_merge_images',
        # 'sct_pipeline',  # not useful-- to remove at some point
        'sct_process_segmentation',
        'sct_propseg',
        'sct_qc',
        'sct_register_multimodal',
        'sct_register_to_template',
        'sct_resample',
        'sct_smooth_spinalcord',
        'sct_straighten_spinalcord', # deps: sct_apply_transfo
        # 'sct_segment_graymatter',
        'sct_warp_template',
    ]


# print without carriage return
# ==========================================================================================
def print_line(string):
    sys.stdout.write(string + make_dot_lines(string))
    sys.stdout.flush()


# fill line with dots
# ==========================================================================================
def make_dot_lines(string):
    if len(string) < 52:
        dot_lines = '.' * (52 - len(string))
        return dot_lines
    else:
        return ''


# print in color
# ==========================================================================================
def print_ok():
    sct.printv("[" + bcolors.OKGREEN + "OK" + bcolors.ENDC + "]")


def print_warning():
    sct.printv("[" + bcolors.WARNING + "WARNING" + bcolors.ENDC + "]")


def print_fail():
    sct.printv("[" + bcolors.FAIL + "FAIL" + bcolors.ENDC + "]")


# init_testing
# ==========================================================================================
def test_function(param_test):
    """

    Parameters
    ----------
    file_testing

    Returns
    -------
    path_output str: path where to output testing data
    """

    # load modules of function to test
    module_function_to_test = importlib.import_module(param_test.function_to_test)
    module_testing = importlib.import_module('test_' + param_test.function_to_test)

    # retrieve subject name
    subject_folder = os.path.basename(param_test.path_data)

    # build path_output variable
    path_testing = os.getcwd()

    # if not param_test.path_output:
    #     param_test.path_output = sct.tmp_create(basename=(param_test.function_to_test + '_' + subject_folder), verbose=0)
    # elif not os.path.isdir(param_test.path_output):
    #     os.makedirs(param_test.path_output)

    # # get parser information
    # parser = module_function_to_test.get_parser()
    # if '-ofolder' in parser.options and '-ofolder' not in param_test.args:
    #     param_test.args += " -ofolder " + param_test.path_output
    #
    # dict_args = parser.parse(shlex.split(param_test.args), check_file_exist=False)
    # # TODO: if file in list does not exist, raise exception and assign status=200
    # # add data path to each input argument
    # dict_args_with_path = parser.add_path_to_file(copy.deepcopy(dict_args), param_test.path_data, input_file=True)
    # # add data path to each output argument
    # dict_args_with_path = parser.add_path_to_file(copy.deepcopy(dict_args_with_path), param_test.path_output, input_file=False, output_file=True)
    # # save into class
    # param_test.dict_args_with_path = dict_args_with_path
    # param_test.args_with_path = parser.dictionary_to_string(dict_args_with_path)
    #
    # initialize panda dataframe
    param_test.results = DataFrame(index=[subject_folder],
                                   data={'status': 0,
                                         'duration': 0,
                                         'output': '',
                                         'path_data': param_test.path_data,
                                         'path_output': param_test.path_output})
    #
    # # retrieve input file (will be used later for integrity testing)00
    # if '-i' in dict_args:
    #     # check if list in case of multiple input files
    #     if not isinstance(dict_args_with_path['-i'], list):
    #         list_file_to_check = [dict_args_with_path['-i']]
    #         # assign field file_input for integrity testing
    #         param_test.file_input = dict_args['-i'].split('/')[-1]
    #         # update index of dataframe by appending file name for more clarity
    #         param_test.results = param_test.results.rename({subject_folder: os.path.join(subject_folder, dict_args['-i'])})
    #     else:
    #         list_file_to_check = dict_args_with_path['-i']
    #         # TODO: assign field file_input for integrity testing
    #     for file_to_check in list_file_to_check:
    #         # file_input = file_to_check.split('/')[1]
    #         # Check if input files exist
    #         if not (os.path.isfile(file_to_check)):
    #             param_test.status = 200
    #             param_test.output += '\nERROR: This input file does not exist: ' + file_to_check
    #             return update_param(param_test)
    #
    # # retrieve ground truth (will be used later for integrity testing)
    # if '-igt' in dict_args:
    #     param_test.fname_gt = dict_args_with_path['-igt']
    #     # Check if ground truth files exist
    #     if not os.path.isfile(param_test.fname_gt):
    #         param_test.status = 201
    #         param_test.output += '\nERROR: The following file used for ground truth does not exist: ' + param_test.fname_gt
    #         return update_param(param_test)

    # run command
    cmd = ' '.join([param_test.function_to_test, param_test.args])
    # param_test.output += '\nWill run in %s:' % (os.path.join(path_testing, param_test.path_output))
    param_test.output += '\n====================================================================================================\n' + cmd + '\n====================================================================================================\n\n'  # copy command
    time_start = time.time()
    try:
        # os.chdir(param_test.path_output)
        # if not os.path.exists(param_test.path_output):
        #     # in case of relative path, we want a subfolder too
        #     os.makedirs(param_test.path_output)
        # os.chdir(path_testing)
        param_test.status, o = sct.run(cmd, verbose=0)
        if param_test.status:
            raise Exception
    except Exception as err:
        param_test.status = 1
        param_test.output += str(err)
        return update_param(param_test)

    param_test.output += o
    param_test.results['duration'] = time.time() - time_start

    # test integrity
    if param_test.test_integrity:
        param_test.output += '\n\n====================================================================================================\n' + 'INTEGRITY TESTING' + '\n====================================================================================================\n\n'  # copy command
        try:
            # os.chdir(param_test.path_output)
            param_test = module_testing.test_integrity(param_test)
            # os.chdir(path_testing)
        except Exception as err:
            # os.chdir(path_testing)
            param_test.status = 2
            param_test.output += str(err)
            return update_param(param_test)

    return update_param(param_test)


def update_param(param):
    """
    Update field "results" in param class
    """
    for results_attr in param.results.columns:
        if hasattr(param, results_attr):
            param.results[results_attr] = getattr(param, results_attr)
    return param


# START PROGRAM
# ==========================================================================================
if __name__ == "__main__":
    sct.init_sct()
    # initialize parameters
    param = Param()
    # call main function
    main()
