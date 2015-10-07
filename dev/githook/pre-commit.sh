#!/bin/bash
export SCT_DIR=$PWD
echo $SCT_DIR
python $SCT_DIR/dev/githook/pre_commit_hook.py
