#!/bin/bash
# binarize image
# author: jcohen@polymtl.ca
# created: 2014-12-07
# dependences: ImageMagick
convert atlas_grays_cerv_sym_correc_r4.png -threshold 0.2% mask_grays_cerv_sym_correc_r4.png

