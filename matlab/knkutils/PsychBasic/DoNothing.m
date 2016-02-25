function [empty1,etc]=DoNothing(arg1,etc)
% function [empty1,etc]=DoNothing([arg1],[etc])
%
% DoNothing is a dummy MEX routine used to measure Matlab's overhead
% in calling MEX functions. Input arguments are ignored. DoNothing
% returns an empty matrix for each explicitly provided output argument.

% 8/17/97 dgp Wrote it.
% 4/24/02 awi Fixed function definition. 
%             Added an error message if DoNothing.m is executed
% 7/2/04  awi Replaced the error message with a call to PsychAssertMex.

AssertMex;


