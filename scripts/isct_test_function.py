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
import commands
import platform
import signal
from time import time, strftime
from msct_parser import Parser
import sct_utils as sct
import os
import copy_reg
import types
import pandas as pd
import json


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


def generate_data_list(folder_dataset, json_requirements=None, verbose=1):
    """
    Construction of the data list from the data set
    This function return a list of directory (in folder_dataset) in which the contrast is present.
    :return data:
    """
    data_subjects, subjects_dir = [], []

    # each directory in folder_dataset should be a directory of a subject
    for subject_dir in os.listdir(folder_dataset):
        if not subject_dir.startswith('.') and os.path.isdir(folder_dataset + subject_dir):
            if read_json(folder_dataset + subject_dir, json_requirements=json_requirements):
                data_subjects.append(folder_dataset + subject_dir + '/')
                subjects_dir.append(subject_dir)

    if not data_subjects:
        sct.printv('ERROR: No subject data were found in ' + folder_dataset + '. '
                   'Please organize your data correctly or provide a correct dataset.',
                   verbose=verbose, type='error')

    return data_subjects, subjects_dir


def read_json(path_dir, json_requirements=None, fname_json='dataset_description.json'):
    path_dir = sct.slash_at_the_end(path_dir, slash=1)
    if fname_json not in os.listdir(path_dir) and json_requirements is not None:
        accept_subject = False
    elif json_requirements is None:
        accept_subject = True
    else:
        json_file = open(path_dir + fname_json)
        dic_info = json.load(json_file)
        json_file.close()
        # pass keys and items to lower case
        dic_info = dict((k.lower(), v.lower()) for k, v in dic_info.iteritems())
        # if no condition is not verified, accept subject
        accept_subject = True
        # read requirements:
        list_conditions = json_requirements.split(',')
        for condition in list_conditions:
            key, val = condition.split('=')
            key, val = key.lower(), val.lower()
            # if key do not exist, do not accept subject
            if key not in dic_info.keys():
                accept_subject = False
            # if value for this key is not the one required, do not accept subject
            elif dic_info[key] != val:
                accept_subject = False

    return accept_subject


def process_results(results, subjects_name, function, folder_dataset, parameters):
    try:
        results_dataframe = pd.concat([result[2] for result in results])
        results_dataframe.loc[:, 'subject'] = pd.Series(subjects_name, index=results_dataframe.index)
        results_dataframe.loc[:, 'script'] = pd.Series([function] * len(subjects_name), index=results_dataframe.index)
        results_dataframe.loc[:, 'dataset'] = pd.Series([folder_dataset] * len(subjects_name), index=results_dataframe.index)
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
    # append local script to PYTHONPATH for import
    sys.path.append(os.path.abspath(os.curdir))
    script_to_be_run = importlib.import_module('test_' + args[0])  # import function as a module
    try:
        output = script_to_be_run.test(*args[1:])
    except:
        import traceback
        print('%s: %s' % ('test_' + args[0], traceback.format_exc()))
        # output = (1, 'ERROR: Function crashed', 'No result')
        from pandas import DataFrame
        status_script = 1
        output_script = 'ERROR: Function crashed.'
        output = (status_script, output_script, DataFrame(data={'status': int(status_script), 'output': output_script}, index=['']))
    return output
    # return script_to_be_run.test(*args[1:])


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def test_function(function, folder_dataset, parameters='', nb_cpu=None, json_requirements=None, verbose=1):
    """
    Run a test function on the dataset using multiprocessing and save the results
    :return: results
    # results are organized as the following: tuple of (status, output, DataFrame with results)
    """

    # generate data list from folder containing
    data_subjects, subjects_name = generate_data_list(folder_dataset, json_requirements=json_requirements)

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
        # return
        # raise KeyboardInterrupt
        # sys.exit(2)
    except Exception as e:
        sct.printv('Error on line {}'.format(sys.exc_info()[-1].tb_lineno), 1, 'warning')
        sct.printv(str(e), 1, 'warning')
        pool.terminate()
        pool.join()
        # raise Exception
        # sys.exit(2)

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

    parser.add_option(name="-json",
                      type_value="str",
                      description="Requirements on center, study, ... that must be satisfied by the json file of each tested subjects\n"
                                  "Syntax:  center=unf,study=errsm,gm_model=0",
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
                      default_value='1')

    parser.add_option(name='-email',
                      type_value='str',
                      description='Email address to send results followed by SMTP passwd (separate with comma).',
                      mandatory=False,
                      default_value='')

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

    # get parameters
    print_if_error = False  # print error message if function crashes (could be messy)
    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])
    function_to_test = arguments["-f"]
    dataset = arguments["-d"]
    dataset = sct.slash_at_the_end(dataset, slash=1)
    parameters = ''
    message = ''  # terminal printout and email message
    if "-p" in arguments:
        parameters = arguments["-p"]
    json_requirements = None
    if "-json" in arguments:
        json_requirements = arguments["-json"]
    nb_cpu = None
    if "-cpu-nb" in arguments:
        nb_cpu = arguments["-cpu-nb"]
    create_log = int(arguments['-log'])
    if '-email' in arguments:
        email, passwd = arguments['-email'].split(',')
    else:
        email = ''
    verbose = int(arguments["-v"])

    # start timer
    start_time = time()
    # create single time variable for output names
    output_time = strftime("%y%m%d%H%M%S")
    print 'Testing started on: ' + strftime("%Y-%m-%d %H:%M:%S")

    # build log file name
    if create_log:
        file_log = 'results_test_' + function_to_test + '_' + output_time
        orig_stdout = sys.stdout
        fname_log = file_log + '.log'
        handle_log = file(fname_log, 'w')
        # redirect to log file
        sys.stdout = handle_log
        print 'Testing started on: ' + strftime("%Y-%m-%d %H:%M:%S")

    # get path of the toolbox
    path_script = os.path.dirname(__file__)
    path_sct = os.path.dirname(path_script)

    # fetch true commit number and branch (do not use commit.txt which is wrong)
    path_curr = os.path.abspath(os.curdir)
    os.chdir(path_sct)
    sct_commit = commands.getoutput('git rev-parse HEAD')
    if not sct_commit.isalnum():
        print 'WARNING: Cannot retrieve SCT commit'
        sct_commit = 'unknown'
        sct_branch = 'unknown'
    else:
        sct_branch = commands.getoutput('git branch --contains ' + sct_commit).strip('* ')
    # with open (path_sct+"/version.txt", "r") as myfile:
    #     version_sct = myfile.read().replace('\n', '')
    # with open (path_sct+"/commit.txt", "r") as myfile:
    #     commit_sct = myfile.read().replace('\n', '')
    print 'SCT commit/branch: ' + sct_commit + '/' + sct_branch
    os.chdir(path_curr)

    # check OS
    platform_running = sys.platform
    if (platform_running.find('darwin') != -1):
        os_running = 'osx'
    elif (platform_running.find('linux') != -1):
        os_running = 'linux'
    print 'OS: ' + os_running + ' (' + platform.platform() + ')'

    # check hostname
    print 'Hostname:', platform.node()

    # Check number of CPU cores
    from multiprocessing import cpu_count
    # status, output = sct.run('echo $ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', 0)
    print 'CPU cores: ' + str(cpu_count())  # + ', Used by SCT: '+output

    # check RAM
    sct.checkRAM(os_running, 0)

    # test function
    try:
        # during testing, redirect to standard output to avoid stacking error messages in the general log
        if create_log:
            sys.stdout = orig_stdout

        results = test_function(function_to_test, dataset, parameters, nb_cpu, json_requirements, verbose)

        # after testing, redirect to log file
        if create_log:
            sys.stdout = handle_log

        # build results
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        results_subset = results.drop('script', 1).drop('dataset', 1).drop('parameters', 1).drop('output', 1)
        results_display = results_subset

        # save panda structure
        if create_log:
            results_subset.to_pickle(file_log + '.pickle')

        # mean
        results_mean = results_subset.query('status != 200 & status != 201').mean(numeric_only=True)
        results_mean['subject'] = 'Mean'
        results_mean.set_value('status', float('NaN'))  # set status to NaN
        # results_display = results_display.append(results_mean, ignore_index=True)

        # std
        results_std = results_subset.query('status != 200 & status != 201').std(numeric_only=True)
        results_std['subject'] = 'STD'
        results_std.set_value('status', float('NaN'))  # set status to NaN
        # results_display = results_display.append(results_std, ignore_index=True)

        # count tests that passed
        count_passed = results_subset.status[results_subset.status == 0].count()
        count_crashed = results_subset.status[results_subset.status == 1].count()
        # count tests that ran
        count_ran = results_subset.query('status != 200 & status != 201').count()['status']

        # results_display = results_display.set_index('subject')
        # jcohenadad, 2015-10-27: added .reset_index() for better visual clarity
        results_display = results_display.set_index('subject').reset_index()

        print '\nCommand: "' + function_to_test + ' ' + parameters
        print 'Dataset: ' + dataset
        # display general results
        print '\nGLOBAL RESULTS:'
        elapsed_time = time() - start_time
        print 'Duration: ' + str(int(round(elapsed_time))) + 's'
        # display results
        print 'Passed: ' + str(count_passed) + '/' + str(count_ran)
        print 'Crashed: ' + str(count_crashed) + '/' + str(count_ran)
        # build mean/std entries
        dict_mean = results_mean.to_dict()
        dict_mean.pop('status')
        dict_mean.pop('subject')
        print 'Mean: ' + str(dict_mean)
        dict_std = results_std.to_dict()
        dict_std.pop('status')
        dict_std.pop('subject')
        print 'STD: ' + str(dict_std)

        # print detailed results
        print '\nDETAILED RESULTS:'
        print results_display.to_string()
        print 'Status: 0: Passed | 1: Crashed | 99: Failed | 200: Input file(s) missing | 201: Ground-truth file(s) missing'

        if verbose == 2:
            import seaborn as sns
            import matplotlib.pyplot as plt
            from numpy import asarray

            n_plots = len(results_display.keys()) - 2
            sns.set_style("whitegrid")
            fig, ax = plt.subplots(1, n_plots, gridspec_kw={'wspace': 1}, figsize=(n_plots * 4, 15))
            i = 0
            ax_array = asarray(ax)

            for key in results_display.keys():
                if key not in ['status', 'subject']:
                    if ax_array.size == 1:
                        a = ax
                    else:
                        a = ax[i]
                    data_passed = results_display[results_display['status'] == 0]
                    sns.violinplot(x='status', y=key, data=data_passed, ax=a, inner="quartile", cut=0,
                                   scale="count", color='lightgray')
                    sns.swarmplot(x='status', y=key, data=data_passed, ax=a, color='0.3', size=4)
                    i += 1
            if ax_array.size == 1:
                ax.set_xlabel(ax.get_ylabel())
                ax.set_ylabel('')
            else:
                for a in ax:
                    a.set_xlabel(a.get_ylabel())
                    a.set_ylabel('')
            plt.savefig('fig_' + file_log + '.png', bbox_inches='tight', pad_inches=0.5)
            plt.close()

    except Exception as err:
        if print_if_error:
            print err

    # stop file redirection
    if create_log:
        sys.stdout.close()
        sys.stdout = orig_stdout
        # display log file to Terminal
        handle_log = file(fname_log, 'r')
        message = handle_log.read()
        print message

    # send email
    if email:
        print 'Sending email...'
        sct.send_email(email, passwd_from=passwd, subject=file_log, message=message, filename=file_log + '.log')
        print 'done!'
