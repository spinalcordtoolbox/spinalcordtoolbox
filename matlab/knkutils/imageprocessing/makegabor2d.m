function [f,g,xx,yy,sd] = makegabor2d(res,r,c,cpfov,ang,phase,bandwidth,xx,yy)

% function [f,g,xx,yy,sd] = makegabor2d(res,r,c,cpfov,ang,phase,bandwidth,xx,yy)
%
% <res> is the number of pixels along one side
% <r> is the row associated with the peak of the Gaussian envelope (can be a decimal).
%   if [], default to the exact center of the image along the vertical dimension.
% <c> is the column associated with the peak of the Gaussian envelope (can be a decimal).
%   if [], default to the exact center of the image along the horizontal dimension.
% <cpfov> is the number of cycles per field-of-view
% <ang> is the orientation in [0,2*pi).  0 means a horizontal Gabor.
% <phase> is the phase in [0,2*pi)
% <bandwidth> is
%   +A where A is the number of cycles per 4 std dev of the Gaussian envelope
%   -B where B is the spatial frequency bandwidth in octave units (FWHM of amplitude spectrum)
%   [X Y] where X is like +A or -B and Y is a positive number.  the interpretation is
%     that X determines the std dev of the Gaussian along the minor axis (orthogonal to the orientation)
%     and Y is a scale factor on X that determines the std dev of the Gaussian along the major axis
%     (parallel to the orientation).
%   note that cases +A and -B imply an isotropic Gaussian envelope.
% <xx>,<yy> (optional) are speed-ups (dependent on <res>)
%
% return <f>, an image where values are in [-1,1].
% we don't normalize the matrix for power or anything like that.
% also return <g>, an image where values are in [0,1].  this is the Gaussian
%   envelope used to construct <f>.
% also return <sd>, the standard deviation(s) that we used (in pixel units)
%
% example:
% figure; imagesc(makegabor2d(32,[],[],4,pi/6,0,2),[-1 1]);
%
% here's an example to check the -B bandwidth case:
% a = makegabor2d(101,[],[],10,pi/2,0,-1);
% b = fftshift(abs(fft2(a)));
% figure; plot(log2(1:50),b(51,52:end),'ro-');

% input
if isempty(r)
  r = (1+res)/2;
end
if isempty(c)
  c = (1+res)/2;
end

% construct coordinates
if ~exist('xx','var') || isempty(xx)
  [xx,yy] = calcunitcoordinates(res);
end

% convert to the unit coordinate frame
r = normalizerange(r,.5,-.5,.5,res+.5,0,0,1);  % note the signs
c = normalizerange(c,-.5,.5,.5,res+.5,0,0,1);

% calculate sd based on bandwidth
if bandwidth(1) > 0
  sd = (bandwidth(1)/cpfov)/4;  % first convert bandwidth to an absolute distance, then the sd is simply one-fourth of this
else
  sd = sqrt(1/(4*pi^2*((2^(-bandwidth(1))*cpfov-cpfov)/(sqrt(2*log(2))+2^(-bandwidth(1))*sqrt(2*log(2))))^2));  % derivation is ugly.
end
if length(bandwidth) > 1
  sd = [sd bandwidth(2)*sd];
end

% do it
if length(bandwidth)==1
  g = exp( ((xx-c).^2+(yy-r).^2)/-(2*sd^2) );
  f = g .* cos(2*pi*cpfov*(-sin(ang)*(xx-c) + cos(ang)*(yy-r)) + phase);
    % the idea here is based on what we said in makegrating2d.m.
    % we want to first rotate the grating CCW and then translate to (<c>,<r>).
    % so, to figure out the values for a given grating, we undo the translation,
    % rotate CW, and then sample from the "base" case.
else
    % see evalgabor2d.m
  coord = [cos(ang) sin(ang); -sin(ang) cos(ang)]*[flatten(xx-c); flatten(yy-r)];
  g = exp(-1/2 * (coord(1,:).^2/(sd(2)^2) + coord(2,:).^2/(sd(1)^2)));
  f = g .* cos(2*pi*cpfov*coord(2,:) + phase);
  g = reshape(g,size(xx));
  f = reshape(f,size(xx));
end

% export
sd = sd*res;
