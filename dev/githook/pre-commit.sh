#!/bin/bash
export SCT_DIR=/Users/olcoma/code/spinalcordtoolbox
echo $SCT_DIR
python $SCT_DIR/dev/githook/pre_commit_hook.py
