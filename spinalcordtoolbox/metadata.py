#!/usr/bin/env python
# -*- coding: utf-8
# Deal with SCT dataset metadata

import sys, io, os, re

class InfoLabel(object):
    def __init__(self):
        """
        """
        raise NotImplementedError()

    def load(self, file):
        """
        Load self from file
        :param file: input filename or file-like object
        """
        raise NotImplementedError()

    def save(self, file):
        """
        Save self to file
        :param file: output filename or file-like object
        """
        raise NotImplementedError()


def read_label_file(path_info_label, file_info_label):
    """Reads file_info_label (located inside label folder) and returns the information needed."""

    indiv_labels_ids, indiv_labels_names, indiv_labels_files, combined_labels_ids, combined_labels_names, combined_labels_id_groups, clusters_apriori = [], [], [], [], [], [], []

    # file name of info_label.txt
    fname_label = os.path.join(path_info_label, file_info_label)

    # Read file
    try:
        f = io.open(fname_label, "rb")
    except IOError:
        sct.printv('\nWARNING: Cannot open ' + fname_label, 1, 'warning')
        # raise
    else:
        # Extract all lines in file.txt
        lines = [line.decode("utf-8") for line in f.readlines() if line.rstrip()]
        lines[-1] += ' '  # To fix an error that could occur at the last line (deletion of the last character of the .txt file)

        # Check if the White matter atlas was provided by the user
        # look at first line
        # header_lines = [lines[i] for i in range(0, len(lines)) if lines[i][0] == '#']
        # info_label_title = header_lines[0].split('-')[0].strip()
        # if '# White matter atlas' not in info_label_title:
        #     sct.printv("ERROR: Please provide the White matter atlas. According to the file "+fname_label+", you provided the: "+info_label_title, type='error')

        # remove header lines (every line starting with "#")
        section = ''
        for line in lines:
            # update section index
            if '# Keyword=' in line:
                section = line.split('Keyword=')[1].split(' ')[0]
            # record the label according to its section
            if (section == 'IndivLabels') and (line[0] != '#'):
                parsed_line = line.split(', ')
                indiv_labels_ids.append(int(parsed_line[0]))
                indiv_labels_names.append(parsed_line[1].strip())
                indiv_labels_files.append(parsed_line[2].strip())

            elif (section == 'CombinedLabels') and (line[0] != '#'):
                parsed_line = line.split(', ')
                combined_labels_ids.append(int(parsed_line[0]))
                combined_labels_names.append(parsed_line[1].strip())
                combined_labels_id_groups.append(','.join(parsed_line[2:]).strip())

            elif (section == 'MAPLabels') and (line[0] != '#'):
                parsed_line = line.split(', ')
                clusters_apriori.append(parsed_line[-1].strip())

        # check if all files listed are present in folder. If not, ERROR.
        for file in indiv_labels_files:
            sct.check_file_exist(os.path.join(path_info_label, file))

        # Close file.txt
        f.close()

        return indiv_labels_ids, indiv_labels_names, indiv_labels_files, combined_labels_ids, combined_labels_names, combined_labels_id_groups, clusters_apriori


def read_label_file_atlas(path_info_label):

    # file name of info_label.txt
    fname_label = path_info_label + param.file_info_label

    # Check info_label.txt existence
    sct.check_file_exist(fname_label)

    # Read file
    f = open(fname_label)

    # Extract all lines in file.txt
    lines = [lines for lines in f.readlines() if lines.strip()]

    # separate header from (every line starting with "#")
    lines = [lines[i] for i in range(0, len(lines)) if lines[i][0] != '#']

    # read each line
    label_id = []
    label_name = []
    label_file = []
    for i in range(0, len(lines) - 1):
        line = lines[i].split(',')
        label_id.append(int(line[0]))
        label_name.append(line[1])
        label_file.append(line[2][:-1].strip())
    # An error could occur at the last line (deletion of the last character of the .txt file), the 5 following code
    # lines enable to avoid this error:
    line = lines[-1].split(',')
    label_id.append(int(line[0]))
    label_name.append(line[1])
    line[2] = line[2] + ' '
    label_file.append(line[2].strip())

    # check if all files listed are present in folder. If not, WARNING.
    sct.printv('\nCheck existence of all files listed in ' + param.file_info_label + ' ...')
    for fname in label_file:
        if os.path.isfile(os.path.join(path_info_label, fname)) or os.path.isfile(os.path.join(path_info_label, fname + '.nii')) or \
                os.path.isfile(os.path.join(path_info_label, fname + '.nii.gz')):
            sct.printv('  OK: ' + path_info_label + fname)
        else:
            sct.printv('  WARNING: ' + path_info_label + fname + ' does not exist but is listed in '
                       + param.file_info_label + '.\n')

    # Close file.txt
    f.close()

    return [label_id, label_name, label_file]


def get_file_label(path_label='', label='', output='file'):
    """
    Get label file name given based on info_label.txt file.
    Label needs to be a substring of the "name" field. E.g.: T1-weighted, spinal cord, white matter, etc.
    :param path_label:
    :param label:
    :param output: {file, filewithpath}
    :return:
    """
    # init
    file_info_label = 'info_label.txt'
    file_label = ''
    # Open file
    fname_label = os.path.join(path_label, file_info_label)
    try:
        f = io.open(fname_label)
    except IOError:
        sct.printv('\nWARNING: Cannot open ' + fname_label, 1, 'warning')
        # raise
    else:
        # Extract lines from file
        lines = [line for line in f.readlines() if line.strip()]
        # find line corresponding to label
        for line in lines:
            # ignore comment
            if not line[0] == '#':
                # check "name" field
                if label in line.split(',')[1].strip():
                    file_label = line.split(',')[2].strip()
                    # sct.printv('Found Label ' + label + ' in file: ' + file_label)
                    break
        if file_label == '':
            sct.printv('\nWARNING: Label ' + label + ' not found.', 1, 'warning')
        # output
        if output == 'file':
            return file_label
        elif output == 'filewithpath':
            return os.path.join(path_label, file_label)
