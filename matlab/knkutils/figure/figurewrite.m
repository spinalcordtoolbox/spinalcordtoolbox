function figurewrite(prefix,num,mode,outputdir,omitclose)

% function figurewrite(prefix,num,mode,outputdir,omitclose)
%
% <prefix> (optional) is the filename prefix.  the prefix can include in it
%   '%d' (or a variant thereof) for the figure number.  you can pass in
%   a number and we automatically convert it using num2str.
%   default: '%d'.
% <num> (optional) is a number to use instead of the figure number
% <mode> (optional) is like in printnice.m.  can also be a cell vector,
%   in which we loop over the elements.  default: [1 72].
%   special case is -1 which means {0 [1 72]}.
% <outputdir> (optional) is the directory to write to.  default: pwd.
%   we automatically make the directory if it doesn't exist.
% <omitclose> (optional) is whether to omit the closing of the figure.  default: 0.
%
% print current figure to <prefix>.[png,eps] and then close figure.
% can use in conjunction with figureprep.m.
%
% example:
% figureprep;
% scatter(randn(100,1),randn(100,1));
% figurewrite;

% SEE: printnice.m.

% input
if ~exist('prefix','var') || isempty(prefix)
  prefix = '%d';
end
if ~exist('num','var') || isempty(num)
  num = [];
end
if ~exist('mode','var') || isempty(mode)
  mode = [1 72];
end
if ~exist('outputdir','var') || isempty(outputdir)
  outputdir = pwd;
end
if ~exist('omitclose','var') || isempty(omitclose)
  omitclose = 0;
end
if isequal(mode,-1)
  mode = {0 [1 72]};
end
if ~iscell(mode)
  mode = {mode};
end

% do it
for p=1:length(mode)
  if isempty(num)
    printnice([],mode{p},outputdir,prefix);
  else
    printnice([],mode{p},outputdir,sprintf(prefix,num));
  end
end
if ~omitclose
  close;
end
