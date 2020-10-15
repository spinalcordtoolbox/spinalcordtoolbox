#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for utils

from spinalcordtoolbox import utils

def test_parse_num_list_inv():
    assert utils.parse_num_list_inv([1, 2, 3, 5, 6, 9]) == '1:3;5:6;9'
    assert utils.parse_num_list_inv([3, 2, 1, 5]) == '1:3;5'
    assert utils.parse_num_list_inv([]) == ''

