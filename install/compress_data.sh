#!/bin/bash

# ZIP DATA
# 1: remove OSX ugly files
find . -name '._*' -type f -delete
find . -name '.DS_Store' -type f -delete
# 2: put files in folder sct_example_data
# 3: go outside of folder and type: zip -r DATE_sct_example_data.zip sct_example_data
# 4: upload on OSF and neuropoly wiki
# 5: update links on sct_download_data
