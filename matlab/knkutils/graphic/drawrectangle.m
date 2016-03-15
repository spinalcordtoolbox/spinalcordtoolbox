function h = drawrectangle(x,y,szx,szy,linestyle)

% function h = drawrectangle(x,y,sz1,sz2,linestyle)
%
% <x> is x-position of rectangle center
% <y> is y-position of rectangle center
% <szx> is size along the x-direction
% <szy> is size along the y-direction.  if [], default to <szx>.
% <linestyle> (optional) is like 'r-'.  default: 'r-'.
%
% draw a rectangle on the current figure.
% return the handle to the line object that we create.
%
% example:
% figure; drawrectangle(5,2,4,[],'r-'); axis equal;

% input
if ~exist('linestyle','var') || isempty(linestyle)
  linestyle = 'r-';
end
if isempty(szy)
  szy = szx;
end

% prep figure
hold on;

% do it
h = plot([x-szx/2 x+szx/2 x+szx/2 x-szx/2 x-szx/2], ...
         [y+szy/2 y+szy/2 y-szy/2 y-szy/2 y+szy/2],linestyle);
