#!/bin/bash
# to uncompress, type:
tar -zxvf sct_binaries.tar.gz
# make sure everything is executable
chmod 775 *
# compress data
tar -czf sct_binaries.tar.gz *
# change permission
chmod 775 sct_binaries.tar.gz
# remove binaries
rm isct*
