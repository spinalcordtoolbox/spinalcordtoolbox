#!/usr/bin/env python
# -*- coding: utf-8
# Dealing with saving data structure as csv, xls, pickle, etc.

from __future__ import absolute_import

def save_as_csv(data, fname):
    """
    Write data structure as csv
    :param data:
    :param fname:
    :return:
    """
# Create output csv file
fname_out = file_out + '.csv'
file_results = open(fname_out, 'w')
file_results.write(','.join(["Slice [z]", "Vertebral level"] + header) + '\n')
# build csv file
file_results.write(','.join([slicegroup, vertgroup] + [str(np.mean(i[ind_slicegroup])) for i in metrics])
                   + '\n')

file_results.close()
