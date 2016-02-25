function f = tsfilter(ts,flt,mode)

% function f = tsfilter(ts,flt,mode)
%
% <ts> is a b x t matrix with time-series data oriented along the rows
% <flt> is a row vector with the filter (in either the Fourier domain or space domain)
% <mode> (optional) is
%   0 means interpret <flt> as a magnitude filter in the Fourier domain,
%     and do the filtering in the Fourier domain.
%   [1 sz A] means interpret <flt> as a magnitude filter in the Fourier domain,
%     but do the filtering in the space domain using imfilter.m and 'replicate'.
%     in order to convert the Fourier filter to the space domain, we use
%     fouriertospace1D.m and sz and use A as the <mode> input to fouriertospace1D.m.
%     you can omit A, in which case we default A to 1 (which means to ensure that
%     the space filter sums to 1).
%   2 means interpret <flt> as a space-domain filter and do the filtering in the
%     space domain using imfilter.m and 'replicate'.
%   default: 0.
%
% return the filtered time-series data.  we force the output to be real-valued.
% in general, beware of wraparound and edge issues!
%
% example:
% flt = zeros(1,100);
% flt(40:60) = 1;
% flt = ifftshift(flt);
% figure; plot(tsfilter(randn(1,100),flt));

% SEE ALSO IMAGEFILTER.M

% constants
num = 1000;  % number to do at a time

% input
if ~exist('mode','var') || isempty(mode)
  mode = 0;
end
if length(mode)==2
  mode = [mode 1];
end

% construct space filter if necessary
if mode(1)==1
  flt = fouriertospace1D(flt,mode(2),[],mode(3));
end

% do it
switch mode(1)
case 0
  f = [];
  for p=1:ceil(size(ts,1)/num)
    mn = (p-1)*num+1;
    mx = min(size(ts,1),(p-1)*num+num);
    f = cat(1,f,real(ifft(fft(ts(mn:mx,:),[],2) .* repmat(flt,[mx-mn+1 1]),[],2)));
  end
case {1 2}
  f = processmulti1D(@imfilter,ts,flt,'replicate','same','conv');
end
