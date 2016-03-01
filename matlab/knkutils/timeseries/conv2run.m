function f = conv2run(a,b,c)

% function f = conv2run(a,b,c)
%
% <a> is a 2D matrix with time x cases
% <b> is a column vector with time x 1
% <c> is a column vector with the same number of rows as <a>.
%   elements should be positive integers.
%
% convolve <a> with <b>, returning a matrix the same size as <a>.
% the convolution is performed separately for each group indicated
% by <c>.  for example, the convolution is performed separately
% for elements matching <c>==1, elements matching <c>==2, etc.
% this ensures that there is no convolution bleedage across groups.
%
% this function is useful for performing convolutions for multiple
% runs (where time does not extend across consecutive runs).
%
% example:
% a = [1 0 0 4 0 0 1 0 0 0 0]';
% b = [1 1 1 1 1]';
% c = [1 1 1 2 2 2 3 3 3 3 3]';
% f = conv2run(a,b,c);
% [a f]

% calc
blen = length(b);

% init
f = zeros(size(a),class(a));

% loop over cases
for p=1:max(c)
  temp = conv2(a(c==p,:),b);
  f(c==p,:) = temp(1:end-blen+1,:);
end
