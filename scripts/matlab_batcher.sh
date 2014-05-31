#!/bin/sh

matlab_exec=/Applications/MATLAB_R2012a.app/bin/matlab
X="${1}(${2}); exit"
echo ${X} > matlab_command.m
cat matlab_command.m
${matlab_exec} -nodesktop -nosplash < matlab_command.m
