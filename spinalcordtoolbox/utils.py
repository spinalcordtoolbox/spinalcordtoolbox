#!/usr/bin/env python
# -*- coding: utf-8
# Collection of useful functions


def num_parser(str_num):
    """
    Parse numbers in string based on delimiter: , or :
    Examples:
      '1,2,3' gives [1, 2, 3]
      '1:3' gives [1, 2, 3]
    :param str_num: string
    :return: list of int
    """
    # check if user selected specific slices using delimitor ','
    if not str_num.find(',') == -1:
        list_num = [int(x) for x in str_num.split(',')]  # n-element list
    else:
        num_range = [int(x) for x in str_num.split(':')]  # 2-element list
        # if only one slice (z) was selected, consider as z:z
        if len(num_range) == 1:
            slices_range = [num_range[0], num_range[0]]
        list_num = [i for i in range(num_range[0], num_range[1] + 1)]
    return list_num
