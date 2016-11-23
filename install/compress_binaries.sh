#!/bin/bash
# to uncompress, type:
tar -zxvf sct_binaries.tar.gz
# make sure everything is executable
chmod 755 *
# compress data: To be run within a folder that contains all binaries (e.g. tmpbin/)
tar -czf sct_binaries.tar.gz *
# change permission
chmod 775 sct_binaries.tar.gz
# remove binaries
rm isct*
