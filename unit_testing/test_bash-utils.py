#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for bash-utils
# Only happy path is included as the inverse will be found with their failure

import pytest

# To check if a list of commands can be run on the current system, use ./bash_utils.sh commands_file="<path to file>"
# To check if a file exists on the current system, use ./bash_utils.sh isExisting_File="<path to file>"


# Check if a list of commands can be run on the current system
# Happy Path Tests
def test_command_exists_and_found():
    sh touch test_file
    # Allow empty files
    assert sh ../util/bash-utils.sh commands_file="./test_file" is True
    sh echo "pytest" > test_file
    assert sh ../util/bash-utils.sh commands_file="./test_file" is True
    sh rm test_file
    
def test_command_missing_and_complains():
    sh touch test_file
    sh echo "notACommand" > test_file
    assert sh ../util/bash-utils.sh ./test_file is False
    sh rm test_file
   
   
# Check if a file exists on the current system
# Happy Path Tests
def test_file_found_not_found():
    sh touch test_file
    assert sh ../util/bash_utils.sh isExisting_File="test_file" is True
    # Test a path not in current directory
    sh mkdir test_dir
    sh touch test_dir/test_file_2
    assert sh ../util/bash_utils.sh isExisting_File="test_dir/test_file" is True
    sh rm -fr test_dir
    sh rm test_file
    
    # If not done after passing the above in this test, 
    # this could present a false positive test, so they must be tested as one
    assert ../util/bash_utils.sh isExisting_File="notAFile" is False
    







