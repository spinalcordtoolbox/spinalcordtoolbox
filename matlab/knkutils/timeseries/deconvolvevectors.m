function f = deconvolvevectors(a,b)

% function f = deconvolvevectors(a,b)
%
% <a> is a vector with the convolved data
% <b> is a vector with the thing to deconvolve out
%
% deconvolve <b> from <a>, returning a vector the same length as <a>.
% note that this overlaps somewhat with the functionality of deconv.m.
%
% example:
% a = randn(1,10);
% b = [1 2 1];
% c = conv(a,b);
% a2 = deconvolvevectors(c,b);
% allzero(a2-[a 0 0])

X = conv2(eye(length(a)),b');
X = X(1:length(a),:);
f = (olsmatrix(X)*a')';
