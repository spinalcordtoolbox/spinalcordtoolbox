function writespmfiles(data,matrixsize,matrixlength,prefix,params)

% function writespmfiles(data,matrixsize,matrixlength,prefix,params)
%
% <data> is X x Y x Z x T with some data.  (int16 is okay.)
% <matrixsize> is [X Y Z] with the matrix size
% <matrixlength> is [A B C] with the corresponding matrix lengths (in mm)
% <prefix> is a string with %d in it (e.g. 'image%04d').  it is okay to
%   omit the %d, in which case T should be 1 (i.e. you should be dealing
%   with just one volume).
% <params> (optional) is an argument to pass to spm_matrix.m.  if supplied,
%   the 'mat' field is pre-multiplied with spm_matrix(<params>).
%
% write each volume in <data> to a separate SPM/ANALYZE file (1-indexed).
% note that we automatically convert to int16, use a default transformation
% matrix (potentially pre-multiplied according to <params>), 
% and use a specific byte-ordering.
%
% example:
% writespmfiles(10000*rand(64,64,16,10),[64 64 16],[3 3 3],'image%04d');

% input
if ~exist('params','var') || isempty(params)
  params = [];
end

% check range
if min(data(:)) < intmin('int16') || max(data(:)) > intmax('int16')
  warning('some data out of the int16 range');
end

% initialize volume information
V = [];
if ~isempty(params)
  V.mat = spm_matrix(params) * createspmmatrix(matrixsize,matrixlength);
else
  V.mat = createspmmatrix(matrixsize,matrixlength);
end
V.dim = matrixsize;
V.dt = [4 0];  % assume int16 and a specific byte ordering
V.pinfo = [1 0 0]';

% loop
for p=1:size(data,4)
  V.fname = [sprintf(prefix,p) '.img'];
  spm_write_vol(V,int16(data(:,:,:,p)));
end
