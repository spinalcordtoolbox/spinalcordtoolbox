# pytest unit tests for sct_run_batch

import glob
import os
import json
import yaml

import pytest
from stat import S_IEXEC
from textwrap import dedent

from spinalcordtoolbox.scripts import sct_run_batch


@pytest.fixture
def dummy_script(tmp_path):
    """Dummy executable script that displays subject."""
    path_out = str(tmp_path / "dummy_script.sh")
    script_text = """
    #!/bin/bash
    SUBJECT=$1
    echo $SUBJECT
    """
    with open(path_out, 'w') as script:
        # indexing removes beginning newline
        script.write(dedent(script_text)[1:])
    # Set "Owner has execute permission" bit to 1 to ensure script is executable
    script_stat = os.stat(path_out)
    os.chmod(path_out, script_stat.st_mode | S_IEXEC)
    # NB: This does nothing on Windows. (https://docs.python.org/3/library/os.html#os.chmod)
    # > Although Windows supports chmod(), you can only set the file’s read-only flag with it
    # > (via the stat.S_IWRITE and stat.S_IREAD constants or a corresponding integer value).
    # > All other bits are ignored.
    return path_out


@pytest.fixture
def dummy_script_with_file_exclusion(tmp_path):
    """Dummy executable script that displays file only for non-excluded files."""
    path_out = str(tmp_path / "dummy_script_with_file_exclusion.sh")
    script_text = """
    #!/bin/bash
    SUBJECT=$1
    # check if FILE_T2 is in the list of excluded files
    FILE_T2="$SUBJECT"_T2w.nii.gz
    if [[ ! " $EXCLUDE_FILES " =~ " $FILE_T2 " ]]; then
        # process file only if it is not excluded
        echo "$SUBJECT"
    fi
    """
    with open(path_out, 'w') as script:
        script.write(dedent(script_text)[1:])
    script_stat = os.stat(path_out)
    os.chmod(path_out, script_stat.st_mode | S_IEXEC)
    return path_out


def test_config_with_args_warning(tmp_path, dummy_script):
    """
    Test that an error is thrown when trying to pass an argument ('-include') alongside a config file.
    """
    data = tmp_path / 'data'
    data.mkdir()
    out = tmp_path / 'out'
    config_path = tmp_path / 'config.json'

    with open(config_path, 'w') as f:
        config = {"jobs": 1, "path_data": str(data), "path_output": str(out)}
        json.dump(config, f)

    with pytest.warns(UserWarning, match=r'-config.*discouraged'):
        sct_run_batch.main(['-c', str(config_path), '-include', 'something', '-script', dummy_script])


def test_config_extra_value_warning(tmp_path, dummy_script):
    """
    Test that an error is thrown for passing a non-existent argument within a config file.
    """
    data = tmp_path / 'data'
    data.mkdir()
    out = tmp_path / 'out'
    config_path = str(tmp_path / 'config.yml')

    with open(config_path, 'w') as config:
        cnf_text = """
        jobs: 1
        path_data: {}
        path_output: {}
        unknowable: unknown
        """.format(data, out)
        config.write(dedent(cnf_text))
        config.flush()

    with pytest.warns(UserWarning, match='unknowable'):
        sct_run_batch.main(['-c', config_path, '-script', dummy_script])


def test_only_one_include_exclude(tmp_path, dummy_script):
    """
    Test that an error is thrown for trying to pass both '-include'/'-include-list', or '-exclude'/'-exclude-list'.
    """
    data = tmp_path / 'data'
    data.mkdir()
    out = tmp_path / 'out'
    with pytest.raises(SystemExit) as e:
        sct_run_batch.main(['-include', 'arg', '-include-list', 'arg2',
                            '-path-data', str(data), '-path-out', str(out), '-script', dummy_script])
    assert e.value.code == 2  # Default error code given by `parser.error` within an ArgumentParser
    with pytest.raises(SystemExit) as e:
        sct_run_batch.main(['-exclude', 'arg', '-exclude-list', 'arg2',
                            '-path-data', str(data), '-path-out', str(out), '-script', dummy_script])
    assert e.value.code == 2  # Default error code given by `parser.error` within an ArgumentParser


@pytest.mark.parametrize("exclusion_type", ['subject', 'file'])
def test_exclude_file(tmp_path, dummy_script, dummy_script_with_file_exclusion, exclusion_type):
    """
    Test that `-exclude-file` properly filters subjects or files depending on the contents of the YAML file.
    """
    # generate I/O folders
    sub_dir_list = ['sub-001', 'sub-002', 'sub-003', 'sub-010', 'sub-011', 'sub-012']
    data = tmp_path / 'data'
    for sub in sub_dir_list:
        (data / sub / 'anat').mkdir(parents=True)
    out = tmp_path / 'out'

    # generate `exclude.yml`
    sub_exclude_list = sub_dir_list[::2]  # exclude every other subject
    exclude_list = (sub_exclude_list if exclusion_type == 'subject'
                    else [f"{sub}_T2w.nii.gz" for sub in sub_exclude_list])
    fname_exclude = tmp_path / 'exclude.yml'
    with open(fname_exclude, 'w') as fp:
        yaml.dump(exclude_list, fp)

    # run script
    script_to_run = dummy_script if exclusion_type == 'subject' else dummy_script_with_file_exclusion
    sct_run_batch.main(argv=['-path-data', str(data), '-path-out', str(out), '-script', script_to_run,
                             '-exclude-file', str(fname_exclude)])

    # test log contents to see which subjects were processed
    script_name = os.path.splitext(os.path.basename(script_to_run))[0]
    for sub in sub_dir_list:
        # excluded subjects
        log_file = out / 'log' / f'{script_name}_{sub}.log'
        if sub in sub_exclude_list:
            # subject exclusion: no log file (subject was not processed)
            if exclusion_type == 'subject':
                assert not log_file.exists()
            # file exclusion: empty log file (`echo` should be skipped for that subject)
            else:
                assert exclusion_type == 'file'
                assert log_file.exists() and os.path.getsize(log_file) == 0

        # included subjects: non-empty log file (`echo` should be run for that subject)
        else:
            assert log_file.exists() and os.path.getsize(log_file) > 0


def test_directory_inclusion_exclusion():
    """
    Test that -include, -include-list, -exclude, and -exclude-list properly filter directories.
    """
    filter_dirs = sct_run_batch._filter_directories  # for brevity

    # Test list of subjects by themselves
    sub_dir_list = ['sub001', 'sub002', 'sub003', 'sub010', 'sub011', 'sub012']
    assert filter_dirs(sub_dir_list, include="sub.*2") == ['sub002', 'sub012']
    assert filter_dirs(sub_dir_list, include_list=["sub001", "sub002"]) == ['sub001', 'sub002']
    assert filter_dirs(sub_dir_list, exclude="sub001") == ['sub002', 'sub003', 'sub010', 'sub011', 'sub012']
    assert filter_dirs(sub_dir_list, exclude_list=['sub010', 'sub011', 'sub012']) == ['sub001', 'sub002', 'sub003']

    # Test list of subjects with session subdirectories
    sess_dir_list = ['sub01/ses01', 'sub01/ses02', 'sub02/ses01', 'sub02/ses02', 'sub03/ses01', 'sub03/ses02']
    assert filter_dirs(sess_dir_list, include="sub01") == ['sub01/ses01', 'sub01/ses02']
    # NB: The `include_list` case below is tied to https://github.com/spinalcordtoolbox/spinalcordtoolbox/issues/3598
    assert filter_dirs(sess_dir_list, include_list=["sub01", "sub02"]) == ['sub01/ses01', 'sub01/ses02',
                                                                           'sub02/ses01', 'sub02/ses02']
    assert filter_dirs(sess_dir_list, exclude="sub01") == ['sub02/ses01', 'sub02/ses02', 'sub03/ses01', 'sub03/ses02']
    assert filter_dirs(sess_dir_list, exclude_list=['sub01', 'sub02']) == ['sub03/ses01', 'sub03/ses02']


def test_directory_parsing(tmp_path):
    """
    Test that subject and session subdirectories are parsed correctly within a sample dataset structure.
    """

    parse_dir = sct_run_batch._parse_dataset_directory  # for brevity

    subject_names = ['sub-001', 'sub-002', 'sub-003']
    for sub in subject_names:
        (tmp_path / sub).mkdir()
        (tmp_path / f"{sub}.txt").touch()  # Add false positive files to ensure only directories are returned
    assert parse_dir(tmp_path) == subject_names
    assert parse_dir(tmp_path, subject_prefix="subject-") == []

    session_names = ['ses-001', 'ses-002']
    for sub in subject_names[:-1]:  # Skip creating 'ses' dirs in [-1] to test a mix of 'sub' and 'sub/ses' directories
        for ses in session_names:
            (tmp_path / sub / ses).mkdir()
            (tmp_path / sub / f"{ses}.txt").touch()  # Add false positive files to ensure only directories are returned
    assert parse_dir(tmp_path) == ([os.path.join(sub, ses) for sub in subject_names[:-1] for ses in session_names]
                                   + [subject_names[-1]])  # Last subject shouldn't have any session subdirectories
    assert parse_dir(tmp_path, ignore_ses=True) == subject_names


def test_non_executable_task(tmp_path, dummy_script):
    """
    Test that sct_run_batch can still process a non-executable script. (sct_run_batch will attempt
    to set the execute bit if passed a non-executable script.)
    """
    data = tmp_path / 'data'
    data.mkdir()
    out = tmp_path / 'out'

    # Set "Owner has execute permission" bit to 0 to ensure script is non-executable
    script_stat = os.stat(dummy_script)
    os.chmod(dummy_script, script_stat.st_mode & ~S_IEXEC)
    # NB: This does nothing on Windows. (https://docs.python.org/3/library/os.html#os.chmod)
    # > Although Windows supports chmod(), you can only set the file’s read-only flag with it
    # > (via the stat.S_IWRITE and stat.S_IREAD constants or a corresponding integer value).
    # > All other bits are ignored.

    sct_run_batch.main(['-include', '^t.*',
                        '-subject-prefix', '',
                        '-path-data', str(data), '-path-out', str(out),
                        '-script', dummy_script,
                        '-continue-on-error', 0])


def test_no_sessions(tmp_path, dummy_script):
    """
    Test that individual subjects (i.e. not in 'ses' subfolders) can be processed individually by sct_run_batch.
    """
    data = tmp_path / 'data'
    data.mkdir()
    out = tmp_path / 'out'
    # Create dummy BIDS directory without sessions
    for sub in ['01', '02']:
        (data / f'sub-{sub}' / 'anat').mkdir(parents=True)
    sct_run_batch.main(['-path-data', str(data), '-path-out', str(out), '-script', dummy_script])
    for sub in ['01', '02']:
        file_log = glob.glob(os.path.join(out, 'log', f'*sub-{sub}.log'))[0]
        assert f'sub-{sub}' in open(file_log, "r").read()


def test_separate_sessions(tmp_path, dummy_script):
    """
    Test that sessions ('ses') can be separated so that sct_run_batch can process each session folder separately.
    """
    data = tmp_path / 'data'
    data.mkdir()
    out = tmp_path / 'out'
    # Create dummy BIDS directory with sessions
    sub_ses_pairs = [('01', '01'), ('01', '02'), ('01', '03'), ('02', '01'), ('02', '02')]
    for sub, ses in sub_ses_pairs:
        (data / f'sub-{sub}' / f'ses-{ses}').mkdir(parents=True)
    sct_run_batch.main(['-path-data', str(data), '-path-out', str(out), '-script', dummy_script])
    for sub, ses in sub_ses_pairs:
        file_log = glob.glob(os.path.join(out, 'log', f'*sub-{sub}_ses-{ses}.log'))[0]
        assert os.path.join(f'sub-{sub}', f'ses-{ses}') in open(file_log, "r").read()
