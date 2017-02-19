#!/bin/bash
# compress SCT binaries
# run this script inside folder that contains all binaries. E.g.:
# ~/code/sct/install/compress_binaries.sh
# once tar.gz is generated, rename as: "DATE"_sct_binaries_"OS".tar.gz
# to uncompress, type:
#tar -zxvf sct_binaries.tar.gz

# remove Apple crap
rm .DS_Store
# make sure everything is executable
chmod 755 *
# compress data
tar -czf sct_binaries.tar.gz *
# change permission
chmod 775 sct_binaries.tar.gz
