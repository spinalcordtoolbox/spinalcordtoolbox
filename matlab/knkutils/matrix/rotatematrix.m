function m = rotatematrix(m,dim1,dim2,k)

% function m = rotatematrix(m,dim1,dim2,k)
%
% <m> is a matrix
% <dim1> is a dimension of <m>
% <dim2> is a dimension of <m>
% <k> is an integer indicating the number of 90 degrees CCW
%   rotations to perform
%
% interpret <m> as 2D slices --- the first dimension is <dim1>,
% and the second dimension is <dim2>.  return the rotated <m>.
%
% a = getsampleimage;
% figure; imagesc(makeimagestack(cat(3,a,rotatematrix(a,1,2,1),rotatematrix(a,1,2,2),rotatematrix(a,1,2,3)),[],[],-1)); axis equal tight;

k = mod(k,4);
switch k
case 0
case {1 3}
  dims = 1:max(max(ndims(m),dim1),dim2);
  dims([dim1 dim2]) = [dim2 dim1];
  m = permute(m,dims);
  switch k
  case 1
    m = flipdim(m,dim1);
  case 3
    m = flipdim(m,dim2);
  end
case 2
  m = flipdim(m,dim1);
  m = flipdim(m,dim2);
end
