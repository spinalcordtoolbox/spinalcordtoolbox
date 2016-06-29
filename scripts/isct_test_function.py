#!/usr/bin/env python
#########################################################################################
#
# This function allows to run a function on a large dataset with a set of parameters.
# Results are extracted and saved in a way that they can easily be compared with another set.
#
# Data should be organized as the following:
# (names of images can be changed but must be passed as parameters to this function)
#
# data/
# ......subject_name_01/
# ......subject_name_02/
# .................t1/
# .........................subject_02_anything_t1.nii.gz
# .........................some_landmarks_of_vertebral_levels.nii.gz
# .........................subject_02_manual_segmentation_t1.nii.gz
# .................t2/
# .........................subject_02_anything_t2.nii.gz
# .........................some_landmarks_of_vertebral_levels.nii.gz
# .........................subject_02_manual_segmentation_t2.nii.gz
# .................t2star/
# .........................subject_02_anything_t2star.nii.gz
# .........................subject_02_manual_segmentation_t2star.nii.gz
# ......subject_name_03/
#          .
#          .
#          .
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Sara Dupont, Benjamin De Leener
# Modified: 2015-09-30
#
# About the license: see the file LICENSE.TXT
#########################################################################################
import sys
import platform
import signal
from time import time, strftime

from msct_parser import Parser
import sct_utils as sct
import os
import copy_reg
import types
import pandas as pd




# get path of the toolbox
# TODO: put it back below when working again (julien 2016-04-04)
# <<<
# OLD
# status, path_sct = commands.getstatusoutput('echo $SCT_DIR')
# NEW
path_script = os.path.dirname(__file__)
path_sct = os.path.dirname(path_script)
# >>>
# append path that contains scripts, to be able to load modules
sys.path.append(path_sct + '/scripts')
sys.path.append(path_sct + '/testing')


def _pickle_method(method):
    """
    Author: Steven Bethard (author of argparse)
    http://bytes.com/topic/python/answers/552476-why-cant-you-pickle-instancemethods
    """
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    cls_name = ''
    if func_name.startswith('__') and not func_name.endswith('__'):
        cls_name = cls.__name__.lstrip('_')
    if cls_name:
        func_name = '_' + cls_name + func_name
    return _unpickle_method, (func_name, obj, cls)


def _unpickle_method(func_name, obj, cls):
    """
    Author: Steven Bethard
    http://bytes.com/topic/python/answers/552476-why-cant-you-pickle-instancemethods
    """
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)


def generate_data_list(folder_dataset, verbose=1):
    """
    Construction of the data list from the data set
    This function return a list of directory (in folder_dataset) in which the contrast is present.
    :return data:
    """
    data_subjects, subjects_dir = [], []

    # each directory in folder_dataset should be a directory of a subject
    for subject_dir in os.listdir(folder_dataset):
        if not subject_dir.startswith('.') and os.path.isdir(folder_dataset + subject_dir):
            data_subjects.append(folder_dataset + subject_dir + '/')
            subjects_dir.append(subject_dir)

    if not data_subjects:
        sct.printv('ERROR: No subject data were found in ' + folder_dataset + '. '
                   'Please organize your data correctly or provide a correct dataset.',
                   verbose=verbose, type='error')

    return data_subjects, subjects_dir


def process_results(results, subjects_name, function, folder_dataset, parameters):
    try:
        results_dataframe = pd.concat([result[2] for result in results])
        results_dataframe.loc[:, 'subject'] = pd.Series(subjects_name, index=results_dataframe.index)
        results_dataframe.loc[:, 'script'] = pd.Series([function]*len(subjects_name), index=results_dataframe.index)
        results_dataframe.loc[:, 'dataset'] = pd.Series([folder_dataset]*len(subjects_name), index=results_dataframe.index)
        results_dataframe.loc[:, 'parameters'] = pd.Series([parameters] * len(subjects_name), index=results_dataframe.index)
        return results_dataframe
    except KeyboardInterrupt:
        return 'KeyboardException'
    except Exception as e:
        sct.printv('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), 1, 'warning')
        sct.printv(str(e), 1, 'warning')
        sys.exit(2)


def function_launcher(args):
    import importlib
    # import traceback
    script_to_be_run = importlib.import_module('test_' + args[0])  # import function as a module
    # try:
    #     output = script_to_be_run.test(*args[1:])
    # except:
    #     print('%s: %s' % ('test_' + args[0], traceback.format_exc()))
    #     output = (1, 'ERROR: Function crashed', 'No result')
    # return output
    return script_to_be_run.test(*args[1:])


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def test_function(function, folder_dataset, parameters='', nb_cpu=None, verbose=1):
    """
    Run a test function on the dataset using multiprocessing and save the results
    :return: results
    # results are organized as the following: tuple of (status, output, DataFrame with results)
    """

    # generate data list from folder containing
    data_subjects, subjects_name = generate_data_list(folder_dataset)

    # All scripts that are using multithreading with ITK must not use it when using multiprocessing on several subjects
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"

    from multiprocessing import Pool

    # create datasets with parameters
    import itertools
    data_and_params = itertools.izip(itertools.repeat(function), data_subjects, itertools.repeat(parameters))

    pool = Pool(processes=nb_cpu, initializer=init_worker)

    try:
        async_results = pool.map_async(function_launcher, data_and_params).get(9999999)
        # results = process_results(async_results.get(9999999), subjects_name, function, folder_dataset, parameters)  # get the sorted results once all jobs are finished
        pool.close()
        pool.join()  # waiting for all the jobs to be done
        results = process_results(async_results, subjects_name, function, folder_dataset, parameters)  # get the sorted results once all jobs are finished
    except KeyboardInterrupt:
        print "\nWarning: Caught KeyboardInterrupt, terminating workers"
        pool.terminate()
        pool.join()
        sys.exit(2)
    except Exception as e:
        sct.printv('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), 1, 'warning')
        sct.printv(str(e), 1, 'warning')
        pool.terminate()
        pool.join()
        sys.exit(2)

    return results


def get_parser():
    # Initialize parser
    parser = Parser(__file__)

    # Mandatory arguments
    parser.usage.set_description("")
    parser.add_option(name="-f",
                      type_value="str",
                      description="Function to test.",
                      mandatory=True,
                      example="sct_propseg")

    parser.add_option(name="-d",
                      type_value="folder",
                      description="Dataset directory.",
                      mandatory=True,
                      example="dataset_full/")

    parser.add_option(name="-p",
                      type_value="str",
                      description="Arguments to pass to the function that is tested. Please put double-quotes if there are spaces in the list of parameters.\n"
                                  "Image paths must be contains in the arguments list.",
                      mandatory=False)

    parser.add_option(name="-cpu-nb",
                      type_value="int",
                      description="Number of CPU used for testing. 0: no multiprocessing. If not provided, "
                                  "it uses all the available cores.",
                      mandatory=False,
                      default_value=0,
                      example='42')

    parser.add_option(name="-log",
                      type_value='multiple_choice',
                      description="Redirects Terminal verbose to log file.",
                      mandatory=False,
                      example=['0', '1'],
                      default_value='0')

    parser.add_option(name="-v",
                      type_value="multiple_choice",
                      description="Verbose. 0: nothing, 1: basic, 2: extended.",
                      mandatory=False,
                      example=['0', '1', '2'],
                      default_value='1')

    return parser


# ====================================================================================================
# Start program
# ====================================================================================================
if __name__ == "__main__":
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])

    function_to_test = arguments["-f"]
    dataset = arguments["-d"]
    dataset = sct.slash_at_the_end(dataset, slash=1)

    parameters = ''
    if "-p" in arguments:
        parameters = arguments["-p"]
    nb_cpu = None
    if "-cpu-nb" in arguments:
        nb_cpu = arguments["-cpu-nb"]
    create_log_file = arguments['-log']


    verbose = arguments["-v"]

    # redirect to log file
    if create_log_file:
        orig_stdout = sys.stdout
        handle_log = file('results_testing_'+strftime("%y%m%d%H%M%S")+'.txt', 'w')
        sys.stdout = handle_log

    # get path of the toolbox
    path_sct = os.getenv("SCT_DIR")
    if path_sct is None :
        raise EnvironmentError("SCT_DIR, which is the path to the "
                               "Spinalcordtoolbox install needs to be set")
    # fetch version of the toolbox
    with open (path_sct+"/version.txt", "r") as myfile:
        version_sct = myfile.read().replace('\n', '')
    with open (path_sct+"/commit.txt", "r") as myfile:
        commit_sct = myfile.read().replace('\n', '')
    print "SCT version: "+version_sct+'-'+commit_sct

    # check OS
    platform_running = sys.platform
    if (platform_running.find('darwin') != -1):
        os_running = 'osx'
    elif (platform_running.find('linux') != -1):
        os_running = 'linux'
    print 'OS: '+os_running+' ('+platform.platform()+')'

    # check hostname
    print 'Hostname:', platform.node()

    # Check number of CPU cores
    from multiprocessing import cpu_count
    status, output = sct.run('echo $ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', 0)
    print 'CPU cores: Available: ' + str(cpu_count()) + ', Used by SCT: '+output

    # check RAM
    print 'RAM:'
    sct.checkRAM(os_running)

    # start timer
    start_time = time()

    print 'Testing... (started on: '+strftime("%Y-%m-%d %H:%M:%S")+')'
    results = test_function(function_to_test, dataset, parameters, nb_cpu, verbose)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    results_subset = results.drop('script', 1).drop('dataset', 1).drop('parameters', 1).drop('output', 1)
    results_display = results_subset

    # save panda structure
    results_subset.to_pickle('results_testing_'+strftime("%y%m%d%H%M%S"))
    results_subset.to_csv('results_testing_'+strftime("%y%m%d%H%M%S")+'.csv')

    # mean
    results_mean = results_subset[results_subset.status != 200].mean(numeric_only=True)
    results_mean['subject'] = 'Mean'
    results_mean.set_value('status', float('NaN'))  # set status to NaN
    results_display = results_display.append(results_mean, ignore_index=True)

    # std
    results_std = results_subset[results_subset.status != 200].std(numeric_only=True)
    results_std['subject'] = 'STD'
    results_std.set_value('status', float('NaN'))  # set status to NaN
    results_display = results_display.append(results_std, ignore_index=True)

    # count tests that passed
    count_passed = results_subset.status[results_subset.status == 0].count()

    # results_display = results_display.set_index('subject')
    # jcohenadad, 2015-10-27: added .reset_index() for better visual clarity
    results_display = results_display.set_index('subject').reset_index()

    # printing results
    print 'Results for "' + function_to_test + ' ' + parameters + '":'
    print 'Dataset: ' + dataset
    print results_display.to_string()
    print 'Passed: ' + str(count_passed) + '/' + str(len(results_subset))

    # display elapsed time
    elapsed_time = time() - start_time
    print 'Total duration: ' + str(int(round(elapsed_time)))+'s'
    print 'Status legend: 0: Passed, 1: Crashed, 99: Failed, 200: File(s) missing'
