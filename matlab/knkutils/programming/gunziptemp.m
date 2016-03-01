function file = gunziptemp(file)

% function file = gunziptemp(file)
%
% <file> is a path to a .gz file or just a path
%   to some other type of file.
%
% gunzip <file> into a temporary directory
% and then return the path to the resulting file.
%
% if <file> does not end in .gz, we simply return
% the path to the original file.

if isequal(getextension(file),'.gz')
  temp = gunzip(file,maketempdir);
  file = temp{1};
end
