function [transform,transformR] = resamplingtransform(resampling)

% function [transform,transformR] = resamplingtransform(resampling)
%
% <resampling> is in the format of the cross-validation case of
%   opt.resampling in fitnonlinearmodel.m.  basically, each row corresponds
%   to a cross-validation iteration and -1s in each row indicate the data
%   points to test on.  all rows must be mutually exclusive and should union
%   to produce the full set of data points.
%
% return <transform> as a vector with the indices of all the data points used
% to test on (different cross-validation iterations are just concatenated together
% in order).  thus, you can use <transform> as an indexing vector on the original
% data points in order to construct a vector of data points that corresponds to
% the 'modelpred' output of fitnonlinearmodel.m.
%
% return <transformR> as an indexing vector that takes the data points in the order
% given by the 'modelpred' output and then returns the data points in the original
% numerical order.
%
% example:
% [transform,transformR] = resamplingtransform([1 1 1 -1 -1; -1 -1 -1 1 1])

transform = [];
for p=1:size(resampling,1)
  transform = [transform find(resampling(p,:)==-1)];
end
transformR = calcposition(transform,1:size(resampling,2));
