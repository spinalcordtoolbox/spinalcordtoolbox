function imwritemulti(m,file)

% function imwritemulti(m,file)
%
% <m> is a uint8 matrix with multiple images along the third dimension.
%   if not uint8, we automatically scale each image to fit the 
%   range [0,255] and convert to uint8.
% <file> (optional) is a filename with something like %d in it.
%   default: 'images%03d.png'.
%
% write each image to a separate file (1-indexed).
%
% example:
% imwritemulti(uint8(255*rand(100,100,3)),'test%d.png');

% input
if ~exist('file','var') || isempty(file)
  file = 'images%03d.png';
end

% do it
for p=1:size(m,3)
  if ~isa(m,'uint8')
    temp = uint8(normalizerange(m(:,:,p),0,255));
  else
    temp = m(:,:,p);
  end
  imwrite(temp,sprintf(file,p));
end
