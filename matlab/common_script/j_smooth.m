% =========================================================================
% FUNCTION
% j_smooth.m
%
% Smooth vector data.
% 
% INPUT
% data				double
% (window_size)		int			Size of the averaging window (default=3)
%
% OUTPUT
% data_filt			double
% 
% COMMENT
% Julien Cohen-Adad 2009-04-12
% =========================================================================
function data_filt = j_smooth(data,window_size)


% retreive arguments
if (nargin<1), help j_smooth; return; end
if (nargin<2), window_size = 3; end

% filter data
winarray = ones(window_size,1)/window_size; 
data_filt = convn(data,winarray,'same');

% data_filt = j_filter(data,200,'low',0.5);

