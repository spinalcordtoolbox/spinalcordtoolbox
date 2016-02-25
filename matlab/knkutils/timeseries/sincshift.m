function m = sincshift(m,shift,dim,padding)

% function m = sincshift(m,shift,dim,padding)
%
% <m> is a matrix with time-series data along some dimension
% <shift> is the temporal shift to apply (in matrix units).  can be any number.  
%   for example, 1 means exactly one time point in the future; -1 means exactly 
%   one time point in the past.  can be a matrix the same size as <m> except 
%   collapsed along <dim>.
% <dim> (optional) is the dimension of <m> with time-series data.
%   default to 2 if <m> is a row vector and to 1 otherwise.
% <padding> (optional) is a non-negative integer with the number of pads to 
%   put before and after the time-series data.  pads are just replicates of 
%   the first and last data points.  the purpose of padding is to reduce 
%   wraparound effects.  default: 100.
%
% temporally shift the time-series data in <m> using sinc interpolation.
% this is accomplished by applying a linear phase shift in the Fourier domain.
% beware of wraparound effects, especially in cases where the beginning and
% ending values are very different and in cases with large transients (e.g. "spikes").
%
% example:
% x0 = 0:.01:10;
% y0 = sin(x0);
% x = 0:.4:10;
% y = sin(x);
% y2 = sincshift(y,1/4);
% figure; hold on;
% plot(x0,y0,'r-');
% plot(x,y,'ro');
% plot(x+.1,y2,'b.');
% %
% x = randn(1,24);
% y = sincshift(x,4);
% figure; hold on;
% plot(x(5:end),'ro');
% plot(y,'b.');

% NOTE: should we use a lanczos kernel???  would this be a better solution to the problem of wraparound?

% input
if ~exist('dim','var') || isempty(dim)
  dim = choose(isrowvector(m),2,1);
end
if ~exist('padding','var') || isempty(padding)
  padding = 100;
end

% prep 2D
msize = size(m);
m = reshape2D(m,dim);
if numel(shift) > 1
  shift = reshape2D(shift,dim);
end

% do it in chunks
chunks = chunking(1:size(m,2),round(1000000/(size(m,1)+2*padding)));
for p=1:length(chunks)

  % pad data; take fft and fftshift it
  m0 = fftshift(fft(cat(1,repmat(m(1,chunks{p}),[padding 1]),m(:,chunks{p}),repmat(m(end,chunks{p}),[padding 1])),[],1),1);
  
  % what are the frequencies in [-pi,pi)?
  freqs = calccpfov1D(size(m0,1),1);
  
  % apply phase shift
    % for positive <shift>, this increases the phase angle, which is like making the wave travel left.
    % this is like going into the future.
  if numel(shift) > 1
    m0 = bsxfun(@times,m0,exp(bsxfun(@times,j*freqs',shift(chunks{p}))));
  else
    m0 = bsxfun(@times,m0,exp(bsxfun(@times,j*freqs',shift)));
  end
  
  % ifftshift it, invert fft
  m0 = real(ifft(ifftshift(m0,1),[],1));
  
  % unpad data, undo 2D
  m(:,chunks{p}) = m0(padding+1:end-padding,:);

end
clear m0;
m = reshape2D_undo(m,dim,msize);
