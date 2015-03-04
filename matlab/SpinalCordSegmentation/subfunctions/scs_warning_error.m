function [] = scs_warning_error(error_code, param)
% scs_warning
% This function is called when an error or a warning situation occurs. It
% first finds the corresponding message in the xls file 'error definition'
% before displaying and crashing the spinal cord segmentation function if
% there is an error. Warning codes correspond to the error_code 101 to 199.
% Error codes correspond to the error_code 201 to 299.
%
% SYNTAX:
% [param] = scs_verif_param(param)
% _________________________________________________________________________
% INPUTS:  
% ERROR_CODE
%   Contains the 
%
% PARAM
%   Contains different parameters that are used to fine-tune the
%   segmentation
% _________________________________________________________________________
% OUTPUTS:
% NONE

%%
global fname_log

% Reads the error_definition file
filename = 'String_definition.xlsx';
try
    [num,txt,~] = xlsread(filename);
catch exception
    error_txt = ['The xlsx file containing the error messages (' filename ') was not found'];
    j_disp(fname_log,error_txt);
    error(error_txt);
end

% Finds the column corresponding to param.language
for i=2:size(txt,2)
    if strcmp(param.language,txt{1,i}) == 1
        language = i;
    end
end

% Finds the line corresponding to the error code
error_line = find(num==error_code) + 1;

% Finds the corresponding text
error_txt = txt{error_line,language};

% Displays the error or warning and write it in the log file
j_disp(fname_log,error_txt);

% If it is an error (and not a warning). Crashes the program
if error_code>=200  
    error(error_txt);
end

end

