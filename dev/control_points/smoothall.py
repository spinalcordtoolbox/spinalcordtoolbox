
import glob
import sct_utils


def main():

    for file in glob.glob('./t250/t*'):
        print file
        path, file_name, ext_fname = sct_utils.extract_fname(file)
        cmd = 'fslmaths '+file+' -s 1 ./t250/smooth'+file_name
        print cmd
        status, output = sct.run(cmd)
        #print status, output
    
    
if __name__ == "__main__":
    main()