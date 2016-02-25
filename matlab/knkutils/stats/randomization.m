function [pval,val,dist] = randomization(m,dim,fun,num,mode,wantconsolidate)

% function [pval,val,dist] = randomization(m,dim,fun,num,mode,wantconsolidate)
%
% <m> is a matrix
% <dim> is the dimension to randomize
% <fun> is a function that accepts matrices of the same dimensions
%   as <m> and outputs a matrix that is collapsed along <dim>
% <num> (optional) is number of randomizations.  default: 1000.
% <mode> (optional) is
%   0 means two-tailed (i.e. abs(dist) > abs(val))
%   1 means one-tailed (i.e. dist > val).
%   default: 0.
% <wantconsolidate> (optional) is whether to return <pval> as cat(dim,pval,val).
%   default: 0.
% 
% randomize along dimension <dim> and apply <fun>.  (individual cases
% are permuted individually.)  calculate p-values by comparing these
% randomly obtained values to actual values returned by <fun>.
% 
% return p-values in <pval>, actual function values in <val>, and 
% distribution of randomly obtained values in <dist>.
%
% the dimensions of <pval> and <val> are the same as <m> except
% collapsed along <dim>.  the dimension of <dist> is the same as
% <m> except with <num> elements along <dim>.
%
% example:
% a = randnmulti(1000,[],[1 .5; .5 1]);
% isequal(0,randomization(a(:,1),1,@(x) calccorrelation(x,a(:,2))))

% inputs
if ~exist('num','var') || isempty(num)
  num = 1000;
end
if ~exist('mode','var') || isempty(mode)
  mode = 0;
end
if ~exist('wantconsolidate','var') || isempty(wantconsolidate)
  wantconsolidate = 0;
end

% calc actual
val = feval(fun,m);

% calc random
dist = [];  % slow but straightforward code
for p=1:num
  dist = cat(dim,dist,feval(fun,permutedim(m,dim,[],1)));
end

% calc p-value
if mode==0
  pval = sum(bsxfun(@ge,abs(dist),abs(val)),dim)/num;
else
  pval = sum(bsxfun(@ge,dist,val),dim)/num;
end

% consolidate
if wantconsolidate
  pval = cat(dim,pval,val);
end
