#!/usr/bin/env python
#########################################################################################
#
# This script generates multiple images from a warping field.
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Benjamin De Leener
# Modified: 2015-04-27
#
# About the license: see the file LICENSE.TXT
#########################################################################################

from msct_image import Image
from sct_utils import run, printv


class WarpingField(Image):
    def __init__(self, param=None, hdr=None, orientation=None, absolutepath="", verbose=1):
        super(WarpingField, self).__init__(param, hdr, orientation, absolutepath, verbose)
        self.num_of_frames = 5
        self.iteration = 0

    def __iter__(self):
        return self

    def next(self):
        if self.iteration <= self.num_of_frames:
            result = Image(self)
            print "Iteration #" + str(self.iteration)
            result.data *= float(self.iteration) / float(self.num_of_frames)
            result.file_name = "tmp."+result.file_name+"_" + str(self.iteration)
            self.iteration += 1
            return result, self.iteration
        else:
            raise StopIteration()

if __name__ == "__main__":
    from msct_parser import Parser
    import sys

    parser = Parser(__file__)
    parser.usage.set_description('This script generates multiple images from a warping field.')
    parser.add_option(name="-i",
                      type_value="file",
                      description="source image (moving). Can be 3D or 4D.",
                      mandatory=True,
                      example="t2.nii.gz")
    parser.add_option(name="-o",
                      type_value="file_output",
                      description="output file.",
                      mandatory=True,
                      example="movie.nii.gz")
    parser.add_option(name="-d",
                      type_value="file",
                      description="destination image (fixed). Must be 3D.",
                      mandatory=True,
                      example="template.nii.gz")
    parser.add_option(name="-w",
                      type_value=[[','], 'file'],
                      description="warping field. If more than one, separate with ','.\nThe movie generator currently supports only warping field and not affine transformation.",
                      mandatory=True,
                      example="warping_field.nii.gz")
    parser.add_option(name="-n",
                      type_value='int',
                      description="number of frames.",
                      mandatory=False,
                      example="20",
                      default_value=5)
    arguments = parser.parse(sys.argv[1:])

    input_file = arguments["-i"]
    output_file = arguments["-o"]
    reference_image = arguments["-d"]
    warping_fields_filename = arguments["-w"]
    number_of_frames = arguments["-n"]

    warping_fields = [WarpingField(filename) for filename in warping_fields_filename]

    filenames_output = []
    while True:
        try:
            warping_fields[0].num_of_frames = number_of_frames
            image_output_iter, iteration = warping_fields[0].next()
            image_output_iter.save()
            filename_warp = image_output_iter.path + image_output_iter.file_name + image_output_iter.ext
            filename_output = "tmp.warped_image_" + str(iteration - 1) + image_output_iter.ext
            run("sct_apply_transfo -i " + input_file + " -d " + reference_image + " -w " + filename_warp +
                " -o " + filename_output)
            filenames_output.append(filename_output)
        except ValueError:
            printv('\nError during warping field generation...', 1, 'error')
        except StopIteration:
            printv('\nFinished iterations.')
            break

    run("fslmerge -t " + output_file + " " + " ".join(filenames_output))
    run("rm -rf tmp.*")

    printv('fslview ' + output_file + " &", 1, 'info')
