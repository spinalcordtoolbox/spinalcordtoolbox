function drawcolorbarcircular(cmap,circulartype)

% function drawcolorbarcircular(cmap,circulartype)
% 
% <cmap> is a colormap
% <circulartype> (optional) is
%   0 means normal colormap interpretation (i.e. minimum value is the beginning of the 
%     first color and maximum value is the end of the last color)
%   1 means centered colormap interpretation (i.e. minimum value is the center of
%     the first color and maximum value is the center of the first color repeated
%     after the last color)
%   default: 0.
%
% draw circular color bar based on colormap <cmap>.
%
% example:
% figure; drawcolorbarcircular(hsv(20),0);
% figure; drawcolorbarcircular(hsv(20),1);

% inputs
if ~exist('circulartype','var') || isempty(circulartype)
  circulartype = 0;
end

% do it
colormap(cmap);
cmapnum = size(cmap,1);
angstep = 2*pi/cmapnum;
switch circulartype
case 0
  for p=1:cmapnum
    drawsector(0,(p-1)*angstep,p*angstep,0,1,cmap(p,:),[]);
  end
case 1
  for p=1:cmapnum
    drawsector(0,(p-1)*angstep - angstep/2,p*angstep - angstep/2,0,1,cmap(p,:),[]);
  end
end
axis equal off;
