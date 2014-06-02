function [fname_log]=scs_log()
% scs_log 
%   creates a folder in the current directory to generate a log file
%
% SYNTAX:
% [FNAME_LOG]=scs_log()
%
% _________________________________________________________________________
% INPUTS:
%
% NONE	
% _________________________________________________________________________
% OUTPUTS:
%
% FNAME_LOG
%	String that defines the path of the log file

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The core of the function starts here %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

warning off MATLAB:MKDIR:DirectoryExists
mkdir(pwd,'logs')
fname_log=strcat('logs',filesep,'scs_log.', datestr(now, 'yyyy.mm.dd.HH.MM'),'.log');

% START FUNCTION
j_disp(fname_log,['\n\n\n==========================================================================================================']);
j_disp(fname_log,['   Running: spinal cord segmentation adapted from Horsfield et al 2010']);
j_disp(fname_log,['   GBM4900']);
j_disp(fname_log,['==========================================================================================================']);
j_disp(fname_log,['.. Started: ',datestr(now)]);
j_disp(fname_log,['\n\n']);

end
