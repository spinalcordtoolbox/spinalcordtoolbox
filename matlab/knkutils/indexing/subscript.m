function varargout = subscript(m,range,wantc)

% function varargout = subscript(m,range,wantc)
%
% <m> is a matrix (of numbers) or a string referring to a matrix
%   variable in the base workspace.  cell matrices are okay.
% <range> is:
%   (1) vector index range that does not use
%       the 'end' keyword, e.g., 4, [5 6]
%   (2) a logical indexing matrix
%   (3) the string ':'
%   (4) a cell vector where elements can be of types (1), (2), (3),
%       e.g., {':' 4:5}
% <wantc> (optional) is whether to perform cell de-referencing.  default: 0.
%
% return something like m(range), or m{range} when <wantc>.
%
% this function is useful for cases where you want
% to get a range of elements from something that
% already has parentheses or brackets.
% it is also useful for working with string-variable
% representations of matrices.  it is also useful
% for writing functions that make no assumption about
% the size of the matrices used as input.
%
% example:
% isequal(size(subscript(randn(10,10),{1:2 ':'})),[2 10])
% a = [1 2; 3 4];
% subscript('a',1)
% subscript('a',a<1.5)

% input
if ~exist('wantc','var') || isempty(wantc)
  wantc = 0;
end

% do it
if wantc
  if iscell(range)
    if ischar(m)
      varargout = evalin('base',['subscript(',m,',',cell2str(range),',1);']);
    else
      varargout = m(range{:});
    end
  else
    if ischar(m)
      if ischar(range) && isequal(range,':')
        varargout = evalin('base',[m,'(:);']);
      else
        varargout = evalin('base',[m,'(',mat2str(range),');']);
      end
    else
      if ischar(range) && isequal(range,':')
        varargout = m(:);
      else
        varargout = m(range);
      end
    end
  end
else
  varargout = cell(1,1);
  if iscell(range)
    if ischar(m)
      varargout{1} = evalin('base',['subscript(',m,',',cell2str(range),');']);
    else
      varargout{1} = m(range{:});
    end
  else
    if ischar(m)
      if ischar(range) && isequal(range,':')
        varargout{1} = evalin('base',[m,'(:);']);
      else
        varargout{1} = evalin('base',[m,'(',mat2str(range),');']);
      end
    else
      if ischar(range) && isequal(range,':')
        varargout{1} = m(:);
      else
        varargout{1} = m(range);
      end
    end
  end
end
