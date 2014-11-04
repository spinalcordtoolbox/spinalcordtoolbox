#!/bin/bash
# Julien Cohen-Adad, Simon Levy
# 2014-10-29

epi_1xangle=$1
epi_2xangle=$2

fname_epi_list="${epi_1xangle} ${epi_2xangle}"
echo $fname_epi_list

for epi in $fname_epi_list; do

    echo ${epi}
    echo $(basename ${epi} .${epi##*.})

    # split
    fslsplit ${epi} -z

    # 2d median filter
    fslmaths vol0000 -kernel boxv 7x7x1 -fmedian vol0000_median
    fslmaths vol0001 -kernel boxv 7x7x1 -fmedian vol0001_median
    fslmaths vol0002 -kernel boxv 7x7x1 -fmedian vol0002_median
    fslmaths vol0003 -kernel boxv 7x7x1 -fmedian vol0003_median

    # merge
    fslmerge -z ${epi}_smoothed vol0000_median vol0001_median vol0002_median vol0003_median

    # remove files
    rm vol*.*

done

fslmaths -dt double ${epi_1xangle}_smoothed -div 2 -div ${epi_2xangle}_smoothed ${epi_1xangle}_half_ratio_smoothed