#!/usr/bin/env bash
# Re-usable bash utilities
# Useage consists of passing named variables as command lined variables
#
# Must be used one at a time :
# To check if a list of commands can be run on the current system, use ./bash_utils.sh commands_file="<path to file>"
# To check if a file exists on the current system, use ./bash_utils.sh isExisting_File="<path to file>"


# Assigns command line arguments
for ARGUMENT in "$@"
do

    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)   

    case "$KEY" in

            commands_file)      commands_file=${VALUE} ;;
            isExisting_File)    isExisting_File=${VALUE} ;; 
    
            *)   ;esac
done


# Checks commands in a file exist on the system
# Takes in variable specifically named commands_file
# Does check for file existance

if [ ./bash-utils.sh commands_file=$commands_file ]:
   if [[ ! -z "$commands_file" ]]:
      while read -r tool
      do
         if [ -z $( command -v "$tool" ) ]; then
            echo "Missing tool: $tool"; exit 1
         fi
      done < $commands_file
   fi
fi


# Checks if a file exists on the system
# Takes in variable specifically named isExisting_File
if [[ ! -z "$isExisting_File" ]]:
   [ -a $isExisting_File] && exit 0 || echo "File $isExisting_File does not exist"; exit 1
fi


