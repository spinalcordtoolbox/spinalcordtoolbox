function savebinary(file,precision,m,wantappend)

% function savebinary(file,precision,m,wantappend)
%
% <file> is a file location
% <precision> is something like 'int16'
% <m> is a matrix
% <wantappend> (optional) is whether to append to <file>.  default: 0.
%
% write <m> to <file>.  for the machine format, we use IEEE floating 
% point with little-endian byte ordering (see fopen).
%
% see also loadbinary.m.
%
% example:
% savebinary('test','uint8',repmat(0:255,[2 1]));
% isequal(loadbinary('test','uint8',[0 256],[255 256]),repmat([254 255],[2 1]))

% constants
machineformat = 'l';

% input
if ~exist('wantappend','var') || isempty(wantappend)
  wantappend = 0;
end

% open file
fid = fopen(file,choose(wantappend,'a','w'),machineformat);
assert(fid ~= -1,'<file> could not be opened for writing');

% do it
assert(fwrite(fid,m,precision,0,machineformat)==prod(size(m)));

% close file
assert(fclose(fid)==0);
