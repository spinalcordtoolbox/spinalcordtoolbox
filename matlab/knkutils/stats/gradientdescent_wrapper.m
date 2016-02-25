function [h,dc,numiters,hse,dcse] = gradientdescent_wrapper(varargin)

% function [h,dc,numiters,hse,dcse] = gradientdescent_wrapper(varargin)
%
% input arguments are the same as for gradientdescent.m, except that we either:
%   expect <randomtype> to be [3 X 0].  this case
%   means to systematically fit using different stopping sets:
%   [3 X 1], [3 X 2], ..., [3 X round(1/splitfrac)] and then
%   average across the results.  this can be thought of as an 
%   "n-fold averaging technique".
% OR
%   expect <randomtype> to be [4 X].  this case means to
%   bootstrap for X times (X >= 1), with <randomtype> internally
%   passed to gradientdescent.m as 1.
% OR
%   expect <randomtype> to be any of the normal cases as described
%   in gradientdescent.m.  in this case there is only one fit result.
%
% return <h> and <dc> as the mean across fit results.
% return <numiters> as the median across fit results.
% return <hse> and <dcse> as the standard deviation across fit results.
% if <fittype> is 1, 2, or 3, we return <h> and <hse> as sparse.
%
% example:
% (see gradientdescent.m)

% NOTE: HIGH VARIANCE ACROSS RESAMPLINGS.  SHOULD WE REVIVE THE FIXEDITER IDEA WHEREIN WE
% FIX THE NUMBER OF ITERATIONS AND USE THAT NUMBER ACROSS RESAMPLINGS?
% SHOULD THIS WRAPPER DO CROSS-VALIDATION ALSO???

% decide case based on <randomtype>
if varargin{5}(1)==3 && varargin{5}(3)==0

  % do it
  h = []; dc = []; numiters = [];
  for p=1:round(1/choose(isempty(varargin{4}),0.2,varargin{4}))  % ACK: we have to know about the default!
    varargin{5}(3) = p;
    [h(:,:,p),dc(:,:,p),numiters(:,:,p)] = gradientdescent(varargin{:});
  end

elseif varargin{5}(1)==4

  % prep
  numboot = varargin{5}(2);
  varargin{5} = 1;  % mangle!
  numdata = size(varargin{1},1);

  % do it
  h = []; dc = []; numiters = [];
  for p=1:numboot
    ix = ceil(rand(1,numdata)*numdata);
    [h(:,:,p),dc(:,:,p),numiters(:,:,p)] = gradientdescent(varargin{1}(ix,:),varargin{2}(ix,:),varargin{3:end});
  end

else

  [h,dc,numiters] = gradientdescent(varargin{:});

end

% deal with output
hse = std(h,[],3);
dcse = std(dc,[],3);
h = mean(h,3);
dc = mean(dc,3);
numiters = median(numiters,3);

% sparsify if necessary
if ismember(varargin{3},[1 2 3])
  h = sparse(h);
  hse = sparse(hse);
end
