function j_renvar(x,y)
%RENVAR Rename Variable Without Memory Reallocation.
% RENVAR OldVar NewVar
% renames the variable named 'OldVar' to 'NewVar'
% If NewVar already exists, it is overwritten.
%
% Example:
%
% x=pi*eye(5); % create data
% RENVAR x y
% renames the variable x as the variable y
%
% Based on insight provided by Doug Schwarz

% D.C. Hanselman, University of Maine, Orono, ME 04469
% MasteringMatlab@yahoo.com
% Mastering MATLAB 7
% 2005-10-07

if nargin~=2 || ~ischar(x) || ~ischar(y)
   error('Two String Arguments Required.')
end
try
   evalin('caller',[y '=' x '; clear ' x])
catch
   error('Variable Rename Failed.')
end