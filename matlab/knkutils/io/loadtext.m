function s = loadtext(file);

% function s = loadtext(file);
%
% <file> is a file location
%
% read <file> and return a cell vector of strings.
% each string corresponds to one line of <file>.
% see also savetext.m.
%
% note that if <file> consists entirely of numeric text,
% it is much faster to use load.m!
%
% example:
% savetext('temp.txt',{'this' 'is' 'a' 'test'});
% a = loadtext('temp.txt')

% open file
fid = fopen(file,'r');
assert(fid ~= -1,'<file> could not be opened for reading');

% do it
s = {};
while 1
  line = fgetl(fid);
  if ~ischar(line)
    break;
  end
  s{end+1} = line;
end

% close file
assert(fclose(fid)==0);
