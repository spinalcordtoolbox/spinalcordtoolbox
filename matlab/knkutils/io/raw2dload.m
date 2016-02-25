function data = raw2dload(files,datatype,sizefirst,lastdimrange,finaldatatype,wanttranspose)

% function data = raw2dload(files,datatype,sizefirst,lastdimrange,finaldatatype,wanttranspose)
%
% <files> is a pattern (see matchfiles.m) that matches one or more binary files.
%   each binary file should contain data from a 2D matrix, and the 2D matrices
%   associated with the binary files should have the same dimensions.
% <datatype> is the datatype like 'int16'
% <sizefirst> is the number of elements along the first dimension of each 2D matrix
% <lastdimrange> (optional) is a vector of indices referring to the second dimension.
%   these indices do not have to be in any particular order and can include repeats.
%   we return only this section of the data.  (the closer that union(<lastdimrange>,[])
%   is to a contiguous chunk of indices, the faster the load will be.)  special 
%   case is [] which means to return all of the data.  default: [].
% <finaldatatype> (optional) is the final datatype that is desired (achieved through cast.m).
%   default is [] which means to use <datatype>.
% <wanttranspose> (optional) is whether to perform a transpose at the end (so you get B x A).
%   default: 0.
%
% if we match only one file, return <data> as A x B.
% if we match multiple files, return <data> as a cell vector of matrices that are A x B.
%
% example:
% savebinary('test','int16',1:10);
% raw2dload('test','int16',1,[4 7 8 9])

% input
if ~exist('lastdimrange','var') || isempty(lastdimrange)
  lastdimrange = [];
end
if ~exist('finaldatatype','var') || isempty(finaldatatype)
  finaldatatype = datatype;
end
if ~exist('wanttranspose','var') || isempty(wanttranspose)
  wanttranspose = 0;
end

% do it
files = matchfiles(files);
if isempty(files)
  error('<files> does not match any files');
end
data = {};
for p=1:length(files)
  fprintf('loading data from %s.\n',files{p});
  
  % load the chunk of data
  if isempty(lastdimrange)
    lastdimrange0 = [];
  else
    lastdimrange0 = [min(lastdimrange) max(lastdimrange)];
  end
  data{p} = loadbinary(files{p},datatype,[sizefirst 0],lastdimrange0);
  
  % subset through data if necessary
  if ~isempty(lastdimrange)
    ix = calcposition(lastdimrange0(1):lastdimrange0(2),lastdimrange);
    if ~isequal(ix,1:size(data{p},2))
      data{p} = data{p}(:,ix);
    end
  end
  
  % convert if necessary
  if ~isequal(datatype,finaldatatype)
    data{p} = cast(data{p},finaldatatype);
  end
  
  % transpose if necessary
  if wanttranspose
    data{p} = data{p}.';
  end

end

% don't embed single cases
if length(data)==1
  data = data{1};
end
