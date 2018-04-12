#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for spinalcordtoolbox.metadata

import sys, io, os

import pytest

import spinalcordtoolbox.metadata

@pytest.fixture(scope="session")
def info_labels():
    sct_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(sct_dir, "data")
    info_labels = list()
    for cwd, dirs, files in os.walk(data_dir):
        for file in files:
            if file == "info_label.txt":
                path = os.path.join(cwd, file)
                info_labels.append(path)
    return info_labels

def test_load_save_load(info_labels):
    for info_label in info_labels:
        il = spinalcordtoolbox.metadata.InfoLabel()
        il.load(info_label)

        b = io.BytesIO()
        il.save(b)
        b.seek(0)
        il2 = spinalcordtoolbox.metadata.InfoLabel()
        il2.load(b)

        assert il2._indiv_labels == il._indiv_labels
        assert il2._combined_labels == il._combined_labels
        assert il2._clusters_apriori == il._clusters_apriori

def test_atlas(info_labels):
    for info_label in info_labels:
        spinalcordtoolbox.metadata.read_label_file_atlas(os.path.dirname(info_label), os.path.basename(info_label))

def test_file(info_labels):
    for info_label in info_labels:
        spinalcordtoolbox.metadata.read_label_file(os.path.dirname(info_label), os.path.basename(info_label))

