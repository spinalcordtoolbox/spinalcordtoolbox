#!/usr/bin/env python
# -*- coding: utf-8
# Convenience/shell related utilites

import os
import sys
import re
import shutil
import logging
import argparse

from enum import Enum

from .sys import check_exe, printv

logger = logging.getLogger(__name__)


def display_open(file):
    """Print the syntax to open a file based on the platform."""
    if sys.platform == 'linux':
        printv('\nDone! To view results, type:')
        printv('xdg-open ' + file + '\n', verbose=1, type='info')
    elif sys.platform == 'darwin':
        printv('\nDone! To view results, type:')
        printv('open ' + file + '\n', verbose=1, type='info')
    else:
        printv('\nDone! To view results, open the following file:')
        printv(file + '\n', verbose=1, type='info')


def display_viewer_syntax(files, colormaps=[], minmax=[], opacities=[], mode='', verbose=1):
    """
    Print the syntax to open a viewer and display images for QC. To use default values, enter empty string: ''
    Parameters
    ----------
    files [list:string]: list of NIFTI file names
    colormaps [list:string]: list of colormaps associated with each file. Available colour maps: see dict_fsleyes
    minmax [list:string]: list of min,max brightness scale associated with each file. Separate with comma.
    opacities [list:string]: list of opacity associated with each file. Between 0 and 1.

    Returns
    -------
    None

    Example
    -------
    display_viewer_syntax([file1, file2, file3])
    display_viewer_syntax([file1, file2], colormaps=['gray', 'red'], minmax=['', '0,1'], opacities=['', '0.7'])
    """
    list_viewer = ['fsleyes', 'fslview_deprecated', 'fslview']  # list of known viewers. Can add more.
    dict_fslview = {'gray': 'Greyscale', 'red-yellow': 'Red-Yellow', 'blue-lightblue': 'Blue-Lightblue', 'red': 'Red',
                    'green': 'Green', 'random': 'Random-Rainbow', 'hsv': 'hsv', 'subcortical': 'MGH-Subcortical'}
    dict_fsleyes = {'gray': 'greyscale', 'red-yellow': 'red-yellow', 'blue-lightblue': 'blue-lightblue', 'red': 'red',
                    'green': 'green', 'random': 'random', 'hsv': 'hsv', 'subcortical': 'subcortical'}
    selected_viewer = None

    # find viewer
    exe_viewers = [viewer for viewer in list_viewer if check_exe(viewer)]
    if exe_viewers:
        selected_viewer = exe_viewers[0]
    else:
        return

    # loop across files and build syntax
    cmd = selected_viewer
    # add mode (only supported by fslview for the moment)
    if mode and selected_viewer in ['fslview', 'fslview_deprecated']:
        cmd += ' -m ' + mode
    for i in range(len(files)):
        # add viewer-specific options
        if selected_viewer in ['fslview', 'fslview_deprecated']:
            cmd += ' ' + files[i]
            if colormaps:
                if colormaps[i]:
                    cmd += ' -l ' + dict_fslview[colormaps[i]]
            if minmax:
                if minmax[i]:
                    cmd += ' -b ' + minmax[i]  # a,b
            if opacities:
                if opacities[i]:
                    cmd += ' -t ' + opacities[i]
        if selected_viewer in ['fsleyes']:
            cmd += ' ' + files[i]
            if colormaps:
                if colormaps[i]:
                    cmd += ' -cm ' + dict_fsleyes[colormaps[i]]
            if minmax:
                if minmax[i]:
                    cmd += ' -dr ' + ' '.join(minmax[i].split(','))  # a b
            if opacities:
                if opacities[i]:
                    cmd += ' -a ' + str(float(opacities[i]) * 100)  # in percentage
    cmd += ' &'

    if verbose:
        printv('\nDone! To view results, type:')
        printv(cmd + '\n', verbose=1, type='info')


class ActionCreateFolder(argparse.Action):
    """
    Custom action: creates a new folder if it does not exist. If the folder
    already exists, do nothing.

    The action will strip off trailing slashes from the folder's name.
    Source: https://argparse-actions.readthedocs.io/en/latest/
    """
    @staticmethod
    def create_folder(folder_name):
        """
        Create a new directory if not exist. The action might throw
        OSError, along with other kinds of exception
        """
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        folder_name = os.path.normpath(folder_name)
        return folder_name

    def __call__(self, parser, namespace, values, option_string=None):
        if type(values) == list:
            folders = list(map(self.create_folder, values))
        else:
            folders = self.create_folder(values)

        # Add the attribute
        setattr(namespace, self.dest, folders)


# TODO: Use for argparse wherever type_value=[['delim'], 'type'] was used
def list_type(delimiter, subtype):
    """
        Factory function that returns a list parsing function, which can be
        used with argparse's `type` option. This allows for more complex type
        parsing, and preserves the behavior of the old msct_parser.
    """
    def list_typecast_func(string):
        return [subtype(v) for v in string.split(delimiter)]

    return list_typecast_func


class Metavar(Enum):
    """
    This class is used to display intuitive input types via the metavar field of argparse
    """
    file = "<file>"
    str = "<str>"
    folder = "<folder>"
    int = "<int>"
    list = "<list>"
    float = "<float>"

    def __str__(self):
        return self.value


class SmartFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """
    Custom formatter that inherits from HelpFormatter, which adjusts the default width to the current Terminal size,
    and that gives the possibility to bypass argparse's default formatting by adding "R|" at the beginning of the text.
    Inspired from: https://pythonhosted.org/skaff/_modules/skaff/cli.html
    """

    def __init__(self, *args, **kw):
        self._add_defaults = None
        super(SmartFormatter, self).__init__(*args, **kw)
        # Update _width to match Terminal width
        try:
            self._width = shutil.get_terminal_size()[0]
        except (KeyError, ValueError):
            logger.warning('Not able to fetch Terminal width. Using default: %s', self._width)

    # this is the RawTextHelpFormatter._fill_text
    def _fill_text(self, text, width, indent):
        # print("splot",text)
        if text.startswith('R|'):
            paragraphs = text[2:].splitlines()
            rebroken = [argparse._textwrap.wrap(tpar, width) for tpar in paragraphs]
            rebrokenstr = []
            for tlinearr in rebroken:
                if (len(tlinearr) == 0):
                    rebrokenstr.append("")
                else:
                    for tlinepiece in tlinearr:
                        rebrokenstr.append(tlinepiece)
            return '\n'.join(rebrokenstr)  # (argparse._textwrap.wrap(text[2:], width))
        return argparse.RawDescriptionHelpFormatter._fill_text(self, text, width, indent)

    # this is the RawTextHelpFormatter._split_lines
    def _split_lines(self, text, width):
        if text.startswith('R|'):
            lines = text[2:].splitlines()
            while lines[0] == '':  # Discard empty start lines
                lines = lines[1:]
            offsets = [re.match("^[ \t]*", l).group(0) for l in lines]
            wrapped = []
            for i, li in enumerate(lines):
                if len(li) > 0:
                    o = offsets[i]
                    ol = len(o)
                    init_wrap = argparse._textwrap.fill(li, width).splitlines()
                    first = init_wrap[0]
                    rest = "\n".join(init_wrap[1:])
                    rest_wrap = argparse._textwrap.fill(rest, width - ol).splitlines()
                    offset_lines = [o + wl for wl in rest_wrap]
                    wrapped = wrapped + [first] + offset_lines
                else:
                    wrapped = wrapped + [li]
            return wrapped
        return argparse.HelpFormatter._split_lines(self, text, width)

    def _get_help_string(self, action):
        if action.default not in [None, "", [], (), {}]:
            return super()._get_help_string(action)
        else:
            return action.help


def parse_num_list(str_num):
    """
    Parse numbers in string based on delimiter ',' or ':'

    .. note::
        Examples:
        '' -> []
        '1,2,3' -> [1, 2, 3]
        '1:3,4' -> [1, 2, 3, 4]
        '1,1:4' -> [1, 2, 3, 4]

    :param str_num: str
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
            list_num += [x for x in range(a, b + 1) if x not in list_num]
            continue
        raise ValueError("unexpected group element {} group spec {}".format(element, str_num))

    return list_num


def get_interpolation(program, interp):
    """
    Get syntax on interpolation field depending on program. Supported programs: ants, flirt, WarpImageMultiTransform
    :param program:
    :param interp:
    :return:
    """
    # TODO: check if field and program exists
    interp_program = ''
    # FLIRT
    if program == 'flirt':
        if interp == 'nn':
            interp_program = ' -interp nearestneighbour'
        elif interp == 'linear':
            interp_program = ' -interp trilinear'
        elif interp == 'spline':
            interp_program = ' -interp spline'
    # ANTs
    elif program == 'ants' or program == 'ants_affine' or program == 'isct_antsApplyTransforms' \
            or program == 'isct_antsSliceRegularizedRegistration' or program == 'isct_antsRegistration':
        if interp == 'nn':
            interp_program = ' -n NearestNeighbor'
        elif interp == 'linear':
            interp_program = ' -n Linear'
        elif interp == 'spline':
            interp_program = ' -n BSpline[3]'
    # check if not assigned
    if interp_program == '':
        logger.warning('%s: interp_program not assigned. Using linear for ants_affine.', os.path.basename(__file__))
        interp_program = ' -n Linear'
    # return
    return interp_program.strip().split()


def parse_num_list_inv(list_int):
    """
    Take a list of numbers and output a string that reduce this list based on delimiter ';' or ':'

    .. note::
        Note: we use ; instead of , for compatibility with csv format.
        Examples:
        [] -> ''
        [1, 2, 3] --> '1:3'
        [1, 2, 3, 5] -> '1:3;5'

    :param list_int: list: list of ints
    :return: str_num: string
    """
    # deal with empty list
    if not list_int or list_int is None:
        return ''
    # Sort list in increasing number
    list_int = sorted(list_int)
    # initialize string
    str_num = str(list_int[0])
    colon_is_present = False
    # Loop across list elements and build string iteratively
    for i in range(1, len(list_int)):
        # if previous element is the previous integer: I(i-1) = I(i)-1
        if list_int[i] == list_int[i - 1] + 1:
            # if ":" already there, update the last chars (based on the number of digits)
            if colon_is_present:
                str_num = str_num[:-len(str(list_int[i - 1]))] + str(list_int[i])
            # if not, add it along with the new int value
            else:
                str_num += ':' + str(list_int[i])
                colon_is_present = True
        # I(i-1) != I(i)-1
        else:
            str_num += ';' + str(list_int[i])
            colon_is_present = False

    return str_num
