#!/usr/bin/env python
# -*- coding: utf-8
# Collection of useful functions

from __future__ import absolute_import

import re

# TODO: add test

def parse_num_list(str_num):
    """
    Parse numbers in string based on delimiter: , or :
    Examples:
      '' -> []
      '1,2,3' -> [1, 2, 3]
      '1:3,4' -> [1, 2, 3, 4]
      '1,1:4' -> [1, 2, 3, 4]
    :param str_num: string
    :return: list of ints
    """
    list_num = list()

    if not str_num:
        return list_num

    elements = str_num.split(",")
    for element in elements:
        m = re.match(r"^\d+$", element)
        if m is not None:
            val = int(element)
            if val not in list_num:
                list_num.append(val)
            continue
        m = re.match(r"^(?P<first>\d+):(?P<last>\d+)$", element)
        if m is not None:
            a = int(m.group("first"))
            b = int(m.group("last"))
            list_num += [ x for x in range(a, b+1) if x not in list_num ]
            continue
        raise ValueError("unexpected group element {} group spec {}".format(element, str_num))

    return list_num
