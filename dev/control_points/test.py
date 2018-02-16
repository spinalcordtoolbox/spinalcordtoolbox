import splines_approximation_v2 as spline
import glob

import sct_utils

def main():
    i = 25
    b = [5,7,9,11,13,15,17,19,21,23,25]
    
    for file in glob.glob('./t250/smoooooth*'):
    #for file in glob.glob('./t250/t250*'):

    	path, file_name, ext_fname = sct_utils.extract_fname(file)
    	cmd1 = 'mkdir ../curves/'+file_name
    	print cmd1
    	status, output = sct.run(cmd1)
    	print status, output
        for bc in b:
            spline.main(file, bc)        
        
if __name__ == "__main__":
    main()