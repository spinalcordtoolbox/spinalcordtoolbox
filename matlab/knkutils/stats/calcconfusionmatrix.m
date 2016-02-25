function f = calcconfusionmatrix(m1,m2,mode,wantnanpp)

% function f = calcconfusionmatrix(m1,m2,mode,wantnanpp)
%
% <m1> is points x dimensions1
% <m2> (optional) is points x dimensions2
%   default: <m1>.
% <mode> (optional) is
%   0 means use dot
%   1 means use dot after mean-subtract
%   2 means use dot after mean-subtract and unit-length-normalize
%   3 means use calcmutualinformation.m
%   4 means Euclidean distance
%   5 means use dot after unit-length-normalize
%   default: 2.
% <wantnanpp> (optional) is whether to perform NaN pre-processing. default: 1.
%
% return something like <m2>'*<m1>, which has the size dimensions2 x dimensions1.
% if <mode> is 0, the result is exactly that.
% if <mode> is 1, the mean of each column of <m1> and <m2> is subtracted first.
% if <mode> is 2, the mean of each column of <m1> and <m2> is subtracted and
%                 then unit-length-normalized.  note that if columns have zero
%                 variance, they will be treated as NaNs, and so the result matrix 
%                 may have NaNs in it.
% if <mode> is 3, we calculate the mutual information between each dimension
%   of <m2> and each dimension of <m1>.
% if <mode> is 4, we calculate the Euclidean distance.
% if <mode> is 5, this is like 2 except we don't subtract the mean first.
%
% if <wantnanpp>, we perform some pre-processing on <m1> and <m2> to deal with the case 
% where one or more elements of these matrices are NaN.  specifically, we omit all
% rows of <m1> and <m2> for which at least one element is NaN.  after that,
% we proceed with the calculations described above.
%
% this function is sort of like pdist.m i think...
%
% example:
% x = randnmulti(10000,[],[1 .5 .3; .5 1 .5; .3 .5 1],[]);
% calcconfusionmatrix(x)
%
% history:
% 2014/09/16 - add <wantnanpp> input
% 2010/06/05 - implement detection and exclusion of rows with NaNs

% input
if ~exist('m2','var') || isempty(m2)
  m2 = m1;
end
if ~exist('mode','var') || isempty(mode)
  mode = 2;
end
if ~exist('wantnanpp','var') || isempty(wantnanpp)
  wantnanpp = 1;
end

% propagate NaNs
if wantnanpp
  bad = any(isnan(m1),2) | any(isnan(m2),2);
  m1 = m1(~bad,:);
  m2 = m2(~bad,:);
end

% do it
switch mode
case 0
  f = m2'*m1;
case 1
  f = zeromean(m2,1)'*zeromean(m1,1);
case 2
  f = unitlength(zeromean(m2,1),1,[],0)'*unitlength(zeromean(m1,1),1,[],0);  % OOPS, slow if m2=m1
case 3
  f = zeros(size(m2,2),size(m1,2));
  fprintf('calcconfusionmatrix');
  for p=1:size(m2,2), fprintf('.');
    for q=1:size(m1,2)
      f(p,q) = calcmutualinformation(m2(:,p),m1(:,q));
    end
  end
  fprintf('done.\n');
case 4
  f = zeros(size(m2,2),size(m1,2));
  for p=1:size(m2,2)
    f(p,:) = sqrt(sum(bsxfun(@minus,m2(:,p),m1).^2,1));  % is there a faster way?
  end
case 5
  f = unitlength(m2,1,[],0)'*unitlength(m1,1,[],0);  % OOPS, slow if m2=m1
end
