function [peak,loc] = calcpeak(m,dim,method,fctr,wantmax)

% function [peak,loc] = calcpeak(m,dim,method,fctr,wantmax)
%
% <m> is a matrix with time-series data along some dimension
% <dim> (optional) is the dimension of <m> with time-series data.
%   default to 2 if <m> is a row vector and to 1 otherwise.
% <method> (optional) is the 'method' input to imresize.m.
%   default: 'lanczos3'.
% <fctr> (optional) is the upsampling factor to use (positive integer).
%   default: 100.
% <wantmax> (optional) is
%   1 means look for the maximum
%   anything else means look for the minimum
%   default: 1.
%
% return <peak> and <loc>, which are matrices with the same dimensions as <m>
% except collapsed along <dim>.  <peak> is the maximum value found.
% <loc> is the location of the maximum value in matrix units.
% we consider only those locations that are within the original valid bounds
% (i.e. 1 <= loc <= size(m,dim)).
%
% to determine peaks, we upsample the data using imresize.m, with an upsampling
% factor of <fctr>.  for example, a vector of length 10 would be upsampled to
% length 1000 if <fctr> is 100.  we then simply find the maximum element (within
% the original valid bounds).
%
% example:
% x = [0 1 2 1 3 4 4.5 2];
% [peak,loc] = calcpeak(x);
% figure; hold on;
% plot(x,'ro-');
% scatter(loc,peak,'bx');
%
% NOTE: this routine is potentially slow and wasteful and can't achieve arbitrary precision!
%       should we revisit?

% input
if ~exist('dim','var') || isempty(dim)
  dim = choose(isrowvector(m),2,1);
end
if ~exist('method','var') || isempty(method)
  method = 'lanczos3';
end
if ~exist('fctr','var') || isempty(fctr)
  fctr = 100;
end
if ~exist('wantmax','var') || isempty(wantmax)
  wantmax = 1;
end

% prep 2D
msize = size(m);
m = reshape2D(m,dim);

% imresize it
m2 = imresize(m,[size(m,1)*fctr size(m,2)],method);
indices = resamplingindices(1,size(m,1),-fctr);

% crop it
ok = indices >= 1 & indices <= size(m,1);
m2 = m2(ok,:);
indices = indices(ok);

% find peak
if wantmax==1
  [peak,ix] = max(m2,[],1);
else
  [peak,ix] = min(m2,[],1);
end
loc = indices(ix);

% undo 2D
dsize = msize;
dsize(dim) = 1;
peak = reshape2D_undo(peak,dim,dsize);
loc = reshape2D_undo(loc,dim,dsize);
