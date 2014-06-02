% =========================================================================
% FUNCTION
% j_mricro2matlab.m
%
% Convert MRIcro coordinates to matlab.
% 
% INPUT
% hdr               structure. Header from analyze image.
% coord_mricro      1x3 double.
%
% OUTPUT
% coord_matlab      1x3 double.
% 
% COMMENTS
% Julien Cohen-Adad 2008-01-18
% =========================================================================
function coord_matlab = j_mricro2matlab(hdr,coord_mricro,opt)


% default initialization
round_output    = 1;
flip_xy         = 0;

% user initialization
if nargin<2, help j_mricro2matlab; return; end
if ~exist('opt'), opt = []; end
if isfield(opt,'round_output'), round_output = opt.round_output; end
if isfield(opt,'flip_xy'), flip_xy = opt.flip_xy; end

% convert
coord_matlab = coord_mricro./hdr.private.hdr.dime.pixdim(2:4) + hdr.private.hdr.hist.origin(1:3);

if round_output
    coord_matlab = round(coord_matlab);
end

if flip_xy
    tmp = coord_matlab(1);
    coord_matlab(1) = coord_matlab(2);
    coord_matlab(2) = tmp;
end


