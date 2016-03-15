function [f,filters] = filter2subsample(x,y,indices)

% function [f,filters] = filter2subsample(x,y,indices)
%
% <x> is a 2D square matrix with the filter
% <y> is a 2D square matrix with the image.  can be 3D with multiple images.
% <indices> is a vector of indices pertaining to each of the
%   first two dimensions of <y>.  can be outside the range [1,max] 
%   (but not too far outside).
%
% compute filter2(x,y) and then subsample according to <indices>.
% return <f> as N x N x cases where N is length(indices).
% also return <filters> as res x res x N*N with the filters used.
%   do not assign this output unless you actually need it, since
%   in some cases it is not necessary to calculate this output
%   in order to calculate <f>.
%
% we use some trickery to make the computation fast.
%
% example:
% isequal(filter2subsample(ones(3,3),[1 2 3; 4 5 6; 7 8 9],[2]),45)

% calc
xn = size(x,1);
yn = size(y,1);
n = length(indices);

% init
filters = [];

% if there are a lot of subsamples and not too many images, do it the regular way
if n/yn > 1/10 && size(y,3) < 10

  f = zeros(n,n,size(y,3));
  for zz=1:size(y,3)

    % if weird indices, we have to use the 'full' option
    if any(indices < 1 | indices > yn)
      adjustment = choose(mod(xn,2)==0,xn/2,(xn-1)/2);
      temp = filter2(x,y(:,:,zz),'full');
      assert(all(indices+adjustment >= 1 & indices+adjustment <= size(temp,1)),'indices are invalid (out-of-range)');
      f(:,:,zz) = temp(indices+adjustment,indices+adjustment);
    else
      temp = filter2(x,y(:,:,zz));
      f(:,:,zz) = temp(indices,indices);
    end

  end
  
  % construct filters explicitly (if the user wants them)
  if nargout >= 2
    filters = constructfiltersubsample(yn,x,indices);
  end

% otherwise, do the trick
else

  % construct filters explicitly
  filters = constructfiltersubsample(yn,x,indices);
  
  % do the dot-product
  f = permute(reshape(squish(y,2)'*squish(filters,2),[],n,n),[2 3 1]);

end
