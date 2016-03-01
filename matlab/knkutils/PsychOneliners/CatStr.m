function string=CatStr(stringArray)
% string=CatStr(stringArray)
% The supplied array or cell array of strings is concatenated, to make one string.
% This function is used by Rush.mex.
% 
% Denis Pelli 16 July 1998

c=cellstr(stringArray);
string=[c{:}];
