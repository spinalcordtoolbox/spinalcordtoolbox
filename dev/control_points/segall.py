import commands
import glob


def main():
    print 'ddd'

    for file in glob.glob('./t250/*'):
        print file
        cmd = 'sct_propseg -i '+file+' -o ./t250/seg -t t2'
        print cmd
        status, output = commands.getstatusoutput(cmd)
        #print status, output
    
    
if __name__ == "__main__":
    main()