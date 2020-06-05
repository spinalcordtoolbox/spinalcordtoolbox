#!/usr/bin/env python
"""
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
usage:

    sct_pipeline  -f sct_a_tool -d /path/to/data/  -p  \" sct_a_tool option \" -cpu-nb 8
"""

# TODO: remove compute duration which is now replaced with results.duration
# TODO: create a dictionnary for param, such that results can display reduced param instead of full. Example: -param t1="blablabla",t2="blablabla"
# TODO: read_database: hard coded fields to put somewhere else (e.g. config file)

from __future__ import print_function, absolute_import

import sys, io, os, types, copy, copy_reg, time, itertools, glob, importlib, pickle
import platform
import signal

path_script = os.path.dirname(__file__)
sys.path.append(os.path.join(sct.__sct_dir__, 'testing'))

import concurrent.futures
if "SCT_MPI_MODE" in os.environ:
    from mpi4py.futures import MPIPoolExecutor as PoolExecutor
    __MPI__ = True
    sys.path.insert(0, path_script)
else:
    from concurrent.futures import ProcessPoolExecutor as PoolExecutor
    __MPI__ = False

from multiprocessing import cpu_count

import numpy as np
import h5py
import pandas as pd

import sct_utils as sct
import spinalcordtoolbox.utils as utils
import msct_parser
import sct_testing

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
    list_subj = []

    # each directory in folder_dataset should be a directory of a subject
    for subject_dir in os.listdir(folder_dataset):
        if not subject_dir.startswith('.') and os.path.isdir(os.path.join(folder_dataset, subject_dir)):
            # data_subjects.append(os.path.join(folder_dataset, subject_dir))
            list_subj.append(subject_dir)

    if not list_subj:
        logger.error('ERROR: No subject data were found in ' + folder_dataset + '. '
                   'Please organize your data correctly or provide a correct dataset.',
                   verbose=verbose, type='error')

    return list_subj


def read_database(folder_dataset, specifications=None, fname_database='', verbose=1):
    """
    Read subject database from xls file.
    Parameters
    ----------
    folder_dataset: path to database
    specifications: field-based specifications for subject selection
    fname_database: fname of XLS file that contains database
    verbose:

    Returns
    -------
    subj_selected: list of subjects selected
    """
    # initialization
    subj_selected = []

    # if fname_database is empty, check if xls or xlsx file exist in the database directory.
    if fname_database == '':
        logger.info('  Looking for an XLS file describing the database...')
        list_fname_database = glob.glob(os.path.join(folder_dataset, '*.xls*'))
        if list_fname_database == []:
            logger.warning('WARNING: No XLS file found. Returning empty list.')
            return subj_selected
        elif len(list_fname_database) > 1:
            logger.warning('WARNING: More than one XLS file found. Returning empty list.')
            return subj_selected
        else:
            fname_database = list_fname_database[0]
            # sct.printv('    XLS file found: ' + fname_database, verbose)

    # read data base file and import to panda data frame
    logger.info('  Reading XLS: ' + fname_database, verbose, 'normal')
    try:
        data_base = pd.read_excel(fname_database)
    except:
        logger.error('ERROR: File '+fname_database+' cannot be read. Please check format or get help from SCT forum.')
    #
    # correct some values and clean panda data base
    # convert columns to int
    to_int = ['gm_model', 'PAM50', 'MS_mapping']
    for key in to_int:
        data_base[key].fillna(0.0).astype(int)
    #
    for key in data_base.keys():
        # remove 'unnamed' columns
        if 'Unnamed' in key:
            data_base = data_base.drop(key, axis=1)
        # duplicate columns with lower case names and with space in names
        else:
            data_base[key.lower()] = data_base[key]
            data_base['_'.join(key.split(' '))] = data_base[key]
    #
    ## parse specifications
    ## specification format: "center=unf,twh:pathology=hc:sc_seg=t2"
    list_fields = specifications.split(':')
    dict_spec = {}
    for f in list_fields:
        field, value = f.split('=')
        dict_spec[field] = value.split(',')
    #
    ## select subjects from specification
    # type of field for which the subject should be selected if the field CONTAINS the requested value (as opposed to the field is equal to the requested value)
    list_field_multiple_choice = ['contrasts', 'sc seg', 'gm seg', 'lesion seg']
    list_field_multiple_choice_tmp = copy.deepcopy(list_field_multiple_choice)
    for field in list_field_multiple_choice_tmp:
        list_field_multiple_choice.append('_'.join(field.split(' ')))
    #
    data_selected = copy.deepcopy(data_base)
    for field, list_val in dict_spec.items():
        if field.lower() not in list_field_multiple_choice:
            # select subject if field is equal to the requested value
            data_selected = data_selected[data_selected[field].isin(list_val)]
        else:
            # select subject if field contains the requested value
            data_selected = data_selected[data_selected[field].str.contains('|'.join(list_val)).fillna(False)]
    #
    ## retrieve list of subjects from database
    database_subj = ['_'.join([str(center), str(study), str(subj)]) for center, study, subj in zip(data_base['Center'], data_base['Study'], data_base['Subject'])]
    ## retrieve list of subjects from database selected
    database_subj_selected = ['_'.join([str(center), str(study), str(subj)]) for center, study, subj in zip(data_selected['Center'], data_selected['Study'], data_selected['Subject'])]

    # retrieve folders from folder_database
    list_folder_dataset = [i for i in os.listdir(folder_dataset) if os.path.isdir(os.path.join(folder_dataset, i))]

    # loop across folders
    for ifolder in list_folder_dataset:
        # check if folder is listed in database
        if ifolder in database_subj:
            # check if subject is selected
            if ifolder in database_subj_selected:
                subj_selected.append(ifolder)
        # if not, report to user
        else:
            logger.warning('WARNING: Subject '+ifolder+' is not listed in the database.', verbose, 'warning')

    return subj_selected


# Julien Cohen-Adad 2017-10-21
# def process_results(results, subjects_name, function, folder_dataset):
#     try:
#         results_dataframe = pd.concat([result for result in results])
#         results_dataframe.loc[:, 'subject'] = pd.Series(subjects_name, index=results_dataframe.index)
#         results_dataframe.loc[:, 'script'] = pd.Series([function] * len(subjects_name), index=results_dataframe.index)
#         results_dataframe.loc[:, 'dataset'] = pd.Series([folder_dataset] * len(subjects_name), index=results_dataframe.index)
#         # results_dataframe.loc[:, 'parameters'] = pd.Series([parameters] * len(subjects_name), index=results_dataframe.index)
#         return results_dataframe
#     except KeyboardInterrupt:
#         return 'KeyboardException'
#     except Exception as e:
#         logger.error('Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
#         logger.exception(e)
#         raise

def function_launcher(args):
    # append local script to PYTHONPATH for import
    sys.path.append(os.path.join(sct.__sct_dir__, "testing"))
    # retrieve param class from sct_testing
    param_testing = sct_testing.Param()
    param_testing.function_to_test = args[0]
    param_testing.path_data = args[1]
    param_testing.args = args[2]
    param_testing.test_integrity = args[3]
    param_testing.redirect_stdout = True  # create individual logs for each subject.
    # load modules of function to test
    module_testing = importlib.import_module('test_' + param_testing.function_to_test)
    # initialize parameters specific to the test
    param_testing = module_testing.init(param_testing)
    try:
        param_testing = sct_testing.test_function(param_testing)
    except:
        import traceback
        logger.error('%s: %s' % ('test_' + args[0], traceback.format_exc()))
        # output = (1, 'ERROR: Function crashed', 'No result')
        from pandas import DataFrame
        # TODO: CHECK IF ASSIGNING INDEX WITH SUBJECT IS NECESSARY
        param_testing.results = DataFrame(index=[''], data={'status': int(1), 'output': 'ERROR: Function crashed.'})
        # status_script = 1
        # output_script = 'ERROR: Function crashed.'
        # output = (status_script, output_script, DataFrame(data={'status': int(status_script), 'output': output_script}, index=['']))

    # TODO: THE THING BELOW: IMPLEMENT INSIDE SCT_TESTING SUB-FUNCTION
    # sys.stdout.close()
    # sys.stdout = stdout_orig
    # # write log file
    # write_to_log_file(fname_log, output, mode='r+', prepend=True)

    return param_testing.results
    # return param_testing.results
    # return script_to_be_run.test(*args[1:])


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def get_list_subj(folder_dataset, data_specifications=None, fname_database=''):
    """
    Generate list of eligible subjects from folder and file containing database
    Parameters
    ----------
    folder_dataset: path to database
    data_specifications: field-based specifications for subject selection
    fname_database: fname of XLS file that contains database

    Returns
    -------
    list_subj: list of subjects
    """
    if data_specifications is None:
        list_subj = generate_data_list(folder_dataset)
    else:
        logger.info('Selecting subjects using the following specifications: ' + data_specifications)
        list_subj = read_database(folder_dataset, specifications=data_specifications, fname_database=fname_database)
    # logger.info('  Total number of subjects: ' + str(len(list_subj)))

    # if no subject to process, raise exception
    if len(list_subj) == 0:
        raise Exception('No subject to process. Exit function.')

    return list_subj


def run_function(function, folder_dataset, list_subj, list_args=[], nb_cpu=None, verbose=1, test_integrity=0):
    """
    Run a test function on the dataset using multiprocessing and save the results
    :return: results
    # results are organized as the following: tuple of (status, output, DataFrame with results)
    """

    # add full path to each subject
    list_subj_path = [os.path.join(folder_dataset, subject) for subject in list_subj]

    # All scripts that are using multithreading with ITK must not use it when using multiprocessing
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"

    # create list that finds all the combinations for function + subject path + arguments. Example of one list element:
    # ('sct_propseg', os.path.join(path_sct, 'data', 'sct_test_function', '200_005_s2''), '-i ' + os.path.join("t2", "t2.nii.gz") + ' -c t2', 1)
    list_func_subj_args = list(itertools.product(*[[function], list_subj_path, list_args, [test_integrity]]))
        # data_and_params = itertools.izip(itertools.repeat(function), data_subjects, itertools.repeat(parameters))

    logger.debug("stating pool with {} thread(s)".format(nb_cpu))
    pool = PoolExecutor(nb_cpu)
    compute_time = None
    try:
        compute_time = time.time()
        count = 0
        all_results = []

        # logger.info('Waiting for results, be patient')
        future_dirs = {pool.submit(function_launcher, subject_arg): subject_arg
                         for subject_arg in list_func_subj_args}

        for future in concurrent.futures.as_completed(future_dirs):
            count += 1
            subject = os.path.basename(future_dirs[future][1])
            arguments = future_dirs[future][2]
            try:
                result = future.result()
                sct.no_new_line_log('Processing subjects... {}/{}'.format(count, len(list_func_subj_args)))
                all_results.append(result)
            except Exception as exc:
                logger.error('{} {} generated an exception: {}'.format(subject, arguments, exc))

        compute_time = time.time() - compute_time

        # concatenate all_results into single Panda structure
        results_dataframe = pd.concat(all_results)

    except KeyboardInterrupt:
        logger.warning("\nCaught KeyboardInterrupt, terminating workers")
        for job in future_dirs:
            job.cancel()
    except Exception as e:
        logger.error('Error on line {}'.format(sys.exc_info()[-1].tb_lineno))
        logger.exception(e)
        for job in future_dirs:
            job.cancel()
        raise
    finally:
        pool.shutdown()

    return {'results': results_dataframe, "compute_time": compute_time}


def get_parser():
    # Initialize parser
    parser = msct_parser.Parser(__file__)

    # Mandatory arguments
    parser.usage.set_description("Run a specific SCT function in a list of subjects contained within a given folder. "
                                 "Multiple parameters can be selected by repeating the flag -p as shown in the example below:\n"
                                 "sct_pipeline -f sct_propseg -d PATH_TO_DATA -p \\\"-i t1/t1.nii.gz -c t1\\\" -p \\\"-i t2/t2.nii.gz -c t2\\\"")
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
                      description="Arguments to pass to the function that is tested. Put double-quotes if there are "
                                  "spaces in the list of parameters. Path to images are relative to the subject's folder. "
                                  "Use multiple '-p' flags if you would like to test different parameters on the same"
                                  "subjects.",
                      mandatory=False)

    parser.add_option(name="-subj",
                      type_value="str",
                      description="Choose the subjects to process based on center, study, [...] to select the testing dataset\n"
                                  "Syntax:  field_1=val1,val2:field_2=val3:field_3=val4,val5",
                      example="center=unf,twh:gm_model=0:contrasts=t2,t2s",
                      mandatory=False)

    parser.add_option(name="-subj-file",
                      type_value="file",
                      description="Excel spreadsheet containing database information (center, study, subject, demographics, ...). If this field is empty, it will search for an xls file located in the database folder. If no xls file is present, all subjects will be selected.",
                      default_value='',
                      mandatory=False)

    parser.add_option(name="-j",
                      type_value="int",
                      description="Number of threads for parallel computing (one subject per thread)."
                                  " By default, all available CPU cores will be used. Set to 0 for"
                                  " no multiprocessing.",
                      mandatory=False,
                      example='42')

    parser.add_option(name="-test-integrity",
                      type_value="multiple_choice",
                      description="Run (=1) or not (=0) integrity testing which is defined in test_integrity() function of the test_ script. See example here: https://github.com/neuropoly/spinalcordtoolbox/blob/master/testing/test_sct_propseg.py",
                      mandatory=False,
                      example=['0', '1'],
                      default_value='0')  # TODO: this should have values True/False as defined in sct_testing, not 0/1

    parser.usage.addSection("\nOUTPUT")

    parser.add_option(name="-log",
                      type_value='multiple_choice',
                      description="Redirects Terminal verbose to log file.",
                      mandatory=False,
                      example=['0', '1'],
                      default_value='1')

    parser.add_option(name="-pickle",
                      type_value='multiple_choice',
                      description="Output Pickle file.",
                      mandatory=False,
                      example=['0', '1'],
                      default_value='1')

    parser.add_option(name='-email',
                      type_value=[[','], 'str'],
                      description="Email information to send results." \
                       " Fields are assigned with '=' and are separated with ',':\n" \
                       "  - addr_to: address to send email to\n" \
                       "  - addr_from: address to send email from (default: spinalcordtoolbox@gmail.com)\n" \
                       "  - login: SMTP login (use if different from email_from)\n"
                       "  - passwd: SMTP password\n"
                       "  - smtp_host: SMTP server (default: 'smtp.gmail.com')\n"
                       "  - smtp_port: port for SMTP server (default: 587)\n"
                       "Note: will always use TLS",
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
    sct.init_sct()


    parser = get_parser()
    arguments = parser.parse(sys.argv[1:])
    function_to_test = arguments["-f"]
    path_data = os.path.abspath(arguments["-d"])
    if "-p" in arguments:
        # in case users used more than one '-p' flag, the output will be a list of all arguments (for each -p)
        if isinstance(arguments['-p'], list):
            list_args = arguments['-p']
        else:
            list_args = [arguments['-p']]
    else:
        list_args = []
    data_specifications = None
    if "-subj" in arguments:
        data_specifications = arguments["-subj"]
    if "-subj-file" in arguments:
        fname_database = arguments["-subj-file"]
    else:
        fname_database = ''  # if empty, it will look for xls file automatically in database folder
    if "-j" in arguments:
        jobs = arguments["-j"]
    else:
        jobs = cpu_count()  # uses maximum number of available CPUs
    test_integrity = int(arguments['-test-integrity'])
    create_log = int(arguments['-log'])
    output_pickle = int(arguments['-pickle'])

    send_email = False
    if '-email' in arguments:
        create_log = True
        send_email = True
        # loop across fields
        for i in arguments['-email']:
            k, v = i.split("=")
            if k == 'addr_to':
                addr_to = v
            if k == 'addr_from':
                addr_from = v
            if k == 'login':
                login = v
            if k == 'passwd':
                passwd_from = v
            if k == 'smtp_host':
                smtp_host = v
            if k == 'smtp_port':
                smtp_port = int(v)

    verbose = int(arguments["-v"])

    # start timer
    time_start = time.time()
    # create single time variable for output names
    output_time = time.strftime("%y%m%d%H%M%S")

    # build log file name
    if create_log:
        # global log:
        file_log = "_".join([output_time, function_to_test, sct.__get_branch().replace("/", "~")]).replace("sct_", "")
        fname_log = file_log + '.log'
        # handle_log = sct.ForkStdoutToFile(fname_log)
        file_handler = sct.add_file_handler_to_logger(fname_log)

    logger.info('Testing started on: ' + time.strftime("%Y-%m-%d %H:%M:%S"))

    # fetch SCT version
    logger.info('SCT version: {}'.format(sct.__version__))

    # check OS
    platform_running = sys.platform
    if (platform_running.find('darwin') != -1):
        os_running = 'osx'
    elif (platform_running.find('linux') != -1):
        os_running = 'linux'
    logger.info('OS: ' + os_running + ' (' + platform.platform() + ')')

    # check hostname
    logger.info('Hostname: {}'.format(platform.node()))

    # Check number of CPU cores
    logger.info('CPU Thread on local machine: {} '.format(cpu_count()))

    logger.info('    Requested threads:       {} '.format(jobs))

    if __MPI__:
        logger.info("Running in MPI mode with mpi4py.futures's MPIPoolExecutor")
    else:
        logger.info("Running with python concurrent.futures's ProcessPoolExecutor")

    # check RAM
    sct.checkRAM(os_running, 0)

    # display command
    logger.info('\nCommand(s):')
    for args in list_args:
        logger.info('  ' + function_to_test + ' ' + args)
    logger.info('Dataset: ' + path_data)
    logger.info('Test integrity: ' + str(test_integrity))

    # test function
    try:
        # retrieve subjects list
        list_subj = get_list_subj(path_data, data_specifications=data_specifications, fname_database=fname_database)
        # during testing, redirect to standard output to avoid stacking error messages in the general log
        if create_log:
            # handle_log.pause()
            sct.remove_handler(file_handler)
        # run function
        logger.debug("enter test fct")
        tests_ret = run_function(function_to_test, path_data, list_subj, list_args=list_args, nb_cpu=jobs, verbose=1, test_integrity=test_integrity)
        logger.debug("exit test fct")
        results = tests_ret['results']
        compute_time = tests_ret['compute_time']
        # after testing, redirect to log file
        if create_log:
            logger.addHandler(file_handler)
        # build results
        pd.set_option('display.max_rows', 500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.max_colwidth', -1)  # to avoid truncation of long string
        pd.set_option('display.width', 1000)
        # drop entries for visibility
        results_subset = results.drop(labels=['status', 'duration', 'path_output', 'path_data', 'output'], axis=1)
        # build new dataframe with nice order
        results_subset = pd.concat([results[['status', 'duration']], results_subset, results[['path_output']]], axis=1)
        # save panda structure
        if output_pickle:
            results.to_pickle(file_log + '.pickle')
            with io.open(file_log + '.pickle', "ab") as f:
                metadata = {
                 "sct_version": sct.__version__,
                 "command-line": sys.argv,
                }
                pickle.dump(metadata, f)

        # compute mean
        results_mean = results.query('status != 200 & status != 201').mean(numeric_only=True)
        results_mean['subject'] = 'Mean'
        results_mean.set_value('status', float('NaN'))  # set status to NaN
        # compute std
        results_std = results_subset.query('status != 200 & status != 201').std(numeric_only=True)
        results_std['subject'] = 'STD'
        results_std.set_value('status', float('NaN'))  # set status to NaN
        # count tests that passed
        count_passed = results_subset.status[results_subset.status == 0].count()
        count_crashed = results_subset.status[results_subset.status == 1].count()
        # count tests that ran
        count_ran = results_subset.query('status != 200 & status != 201').count()['status']
        # display general results
        logger.info('\nGLOBAL RESULTS:')
        logger.info('Duration: ' + str(int(np.round(compute_time))) + 's')
        # display results
        logger.info('Passed: ' + str(count_passed) + '/' + str(count_ran))
        logger.info('Crashed: ' + str(count_crashed) + '/' + str(count_ran))
        # build mean/std entries
        dict_mean = results_mean.to_dict()
        dict_mean.pop('status')
        dict_mean.pop('subject')
        logger.info('Mean: ' + str(dict_mean))
        dict_std = results_std.to_dict()
        dict_std.pop('status')
        dict_std.pop('subject')
        logger.info('STD: ' + str(dict_std))
        # logger.info(detailed results)
        logger.info('\nDETAILED RESULTS:')
        logger.info(results_subset.to_string())
        logger.info('\nLegend status:\n0: Passed | 1: Function crashed | 2: Integrity testing crashed | 99: Failed | 200: Input file(s) missing | 201: Ground-truth file(s) missing')

        if verbose == 2:
            import seaborn as sns
            import matplotlib.pyplot as plt
            from numpy import asarray

            n_plots = len(results_subset.keys()) - 2
            sns.set_style("whitegrid")
            fig, ax = plt.subplots(1, n_plots, gridspec_kw={'wspace': 1}, figsize=(n_plots * 4, 15))
            i = 0
            ax_array = asarray(ax)

            for key in results_subset.keys():
                if key not in ['status', 'subject']:
                    if ax_array.size == 1:
                        a = ax
                    else:
                        a = ax[i]
                    data_passed = results_subset[results_subset['status'] == 0]
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
    finally:
        if create_log:
            file_handler.flush()
            sct.remove_handler(file_handler)
        # send email
        if send_email:
            logger.info('\nSending email...')
            # open log file and read content
            with io.open(fname_log, "r") as fp:
                message = fp.read()
            # send email
            utils.send_email(addr_to=addr_to, addr_from=addr_from,
             subject=file_log, message=message, filename=fname_log,
             login=login, passwd=passwd_from, smtp_host=smtp_host, smtp_port=smtp_port,
             html=True)
            # handle_log.send_email(email=email, passwd_from=passwd, subject=file_log, attachment=True)
            logger.info('Email sent!\n')
