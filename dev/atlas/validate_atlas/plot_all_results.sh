#!/bin/bash

mkdir result_plots

python plot_abs_error_vs_csf_values.py
python plot_abs_error_vs_fractional_volume.py
python plot_auto_vs_manual.py
python plot_map.py
python plot_snr_and_tracts_std.py
