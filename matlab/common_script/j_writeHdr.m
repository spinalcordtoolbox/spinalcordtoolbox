% =========================================================================
% FUNCTION
% j_writeHdr
%
% Write HDR for analyze volumes
%
% INPUTS
% hdrstruct     structure containing optionally the following fields
%   fname       e.g. 'D:\data_irm\10-DIFFUSION\meanB0\s2006-0001.img'
%   dim         e.g. [128 128 52 4]
%   mat         [4x4 double]
%   pinfo       [3x1 double]
%   descrip     e.g. '2.89T 2D SE\EP TR=9500ms/TE=102ms/FA=90deg 12-Apr-06'
%   n           ?
%   private     [1x1 struct]
%
% OUTPUTS
% hdr_out       est une structure qui contient une description du volume.
%
% DEPENDANCES
% (-)
%
% COMMENTS
% Julien Cohen-Adad 2006-08-28
% =========================================================================
function varargout = j_writeHdr(hdrstruct)


% initialization
if (nargin<1), error('Insufficient arguments. Type "help j_writeHdr"'); end

% fill fields
if isfield(hdrstruct,'fname')
    hdr_out.fname = hdrstruct.fname;
else
    hdr_out.fname = '';
end

if isfield(hdrstruct,'dim')
    hdr_out.dim = hdrstruct.dim;
else
    hdr_out.dim = [];
end

if isfield(hdrstruct,'mat')
    hdr_out.mat = hdrstruct.mat;
else
    hdr_out.mat = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1];
end

if isfield(hdrstruct,'pinfo')
    hdr_out.pinfo = hdrstruct.pinfo;
else
    hdr_out.pinfo = '';
end

% output
% if (nargout==0), error('Insufficient output arguments. Type "help j_writeHdr"'); end
varargout{1} = hdr_out;
