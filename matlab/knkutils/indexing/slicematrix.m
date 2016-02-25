function f = slicematrix(m,dim,indices)

% function f = slicematrix(m,dim,indices)
%
% <m> is a matrix
% <dim> is a dimension of <m>
% <indices> is a vector of indices
%
% return a slice from <m>.
%
% example:
% isequal(slicematrix([1 2; 3 4; 5 6],2,2),[2 4 6]')

sub = repmat({':'},1,max(dim,ndims(m)));
sub{dim} = indices;
f = subscript(m,sub);
