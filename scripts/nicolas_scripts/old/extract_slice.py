from nicolas_scripts.old.algo_rot_test import *


def get_parser():

    parser = Parser(__file__)
    parser.usage.set_description('Blablablabla')
    parser.add_option(name="-i",
                      type_value="file",
                      description="Input file to extract slice from",
                      mandatory=True,
                      example="/home/data")
    parser.add_option(name="-slice",
                      type_value="str",
                      description="slice number",
                      mandatory=True,
                      example="3")
    parser.add_option(name="-o",
                      type_value="folder",
                      description="output folder ",
                      mandatory=True,
                      example="path/to/output/folder")

    return parser


def main(args=None):

    if args is None:
        args = sys.argv[1:]

    parser = get_parser()
    arguments = parser.parse(args)
    file_input = arguments['-i']
    path_output = arguments['-o']
    slice_no = int(arguments['-slice'])

    data = load_image(file_input, 3)[:, :, slice_no]
    save_image(data, "slice_" + str(slice_no) + "_of_" + (file_input.split("/")[-1]).split(".nii")[0] + ".nii", file_input, path_output)


if __name__ == "__main__":
    sct.init_sct()
    # call main function
    main()
