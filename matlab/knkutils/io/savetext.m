function savetext(file,s);

% function savetext(file,s);
%
% <file> is a file location
% <s> is a string or a cell vector of strings.
%
% write <s> to <file>.  we write '\n' after each string.
% see also loadtext.m.
%
% example:
% savetext('temp.txt',{'this' 'is' 'a' 'test'});
% a = loadtext('temp.txt')

% input
if ~iscell(s)
  s = {s};
end

% open file
fid = fopen(file,'w');
assert(fid ~= -1,'<file> could not be opened for writing');

% do it
for p=1:length(s)
  fprintf(fid,'%s\n',s{p});
end

% close file
assert(fclose(fid)==0);
