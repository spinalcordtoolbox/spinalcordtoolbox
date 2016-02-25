function assignseparatevars(varargin)

% function assignseparatevars(m,dim,varname)
%
% <m> is a matrix
% <dim> is a dimension of <m>
% <varname> is a string that can accept a positive integer
%
% assign, in the caller workspace, variables named sprintf(<varname>,p)
% where p ranges from 1 to size(<m>,<dim>).  the value of each variable
% is the corresponding slice of <m> along <dim>.
%
% example:
% assignseparatevars([1 2 3],2,'test%d');
% isequal(test1,1)
% isequal(test2,2)
% isequal(test3,3)
%
% OR
%
% function assignseparatevars(m,varname)
%
% <m> is a cell vector
% <varname> is a string that can accept a positive integer
%
% assign, in the caller workspace, variables named sprintf(<varname>,p)
% where p ranges from 1 to length(<m>).  the value of each variable
% is the corresponding element of <m>.
%
% example:
% assignseparatevars({1 2 3},'test%d');
% isequal(test1,1)
% isequal(test2,2)
% isequal(test3,3)

if nargin==3
  for p=1:size(varargin{1},varargin{2})
    assignin('caller',sprintf(varargin{3},p),slicematrix(varargin{1},varargin{2},p));
  end
else
  for p=1:length(varargin{1})
    assignin('caller',sprintf(varargin{2},p),subscript(varargin{1},p,1));
  end
end
