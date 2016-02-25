function drawcheckerboard(res,phwidth,phheight,ang,width,height,color1,color2)

% function drawcheckerboard(res,phwidth,phheight,ang,width,height,color1,color2)
%
% <res> is
%   0 means standard figure coordinate frame
%  -1 means y-reversed coordinate frame
% <phwidth> is the phase in [0,2*pi) along the width direction (positive means rightward)
% <phheight> is the phase in [0,2*pi) along the height direction (positive means downward)
% <ang> is the orientation in [0,2*pi).  0 means horizontal.
% <width> is the width of a check
% <height> is the height of a check.  0 is a special case which means to use checks that
%   always extend beyond the field-of-view (thus, analogous to bars).
%   in this case, <phheight> should be set to 0.
% <color1> is a 3-element vector with color of center check
% <color2> is a 3-element vector with the other color
%
% draw an oriented checkerboard on the current figure.
% <width> and <length> are interpreted with respect to the x-
% and y-axes being bounded by [-.5,.5].  we automatically set the
% axis bounds and also reverse the y-axis if necessary.
%
% example:
% figure; drawcheckerboard(-1,0,0,pi/6,.3,.1,[.5 0 0],[1 1 1]);

% input
if height==0
  height = sqrt(2)*1.05;
end

% figure out where the center check should be
x0 = phwidth/(2*pi)*(2*width) * cos(ang) + phheight/(2*pi)*(2*height) * sin(ang);
y0 = phwidth/(2*pi)*(2*width) * sin(ang) - phheight/(2*pi)*(2*height) * cos(ang);
y0 = choose(res==0,y0,-y0);

% figure out how many checks to extend in both the + and - directions.
% this is an overestimate, but that's okay.
% (x0+.5 is half, need sqrt(2) because of rotation, ceil to be conservative, and +2 because of translation)
xout = ceil(sqrt(2)*(abs(x0)+.5)/width) + 2;
yout = ceil(sqrt(2)*(abs(y0)+.5)/height) + 2;

% prepare rotation
ang0 = choose(res==0,-ang,ang);
trans = [cos(ang0) sin(ang0); -sin(ang0) cos(ang0)];

% do it
for p=-xout:xout
  for q=-yout:yout
    newcoord = trans * [p*width; q*height];  % want to rotate CCW (see makegrating.m)
    drawbar(res,newcoord(1,:)+x0,newcoord(2,:)+y0,ang,height,width,choose(mod(p+q,2)==0,color1,color2),[]);  % tricky
  end
end

% prep figure
axis([-.5 .5 -.5 .5]);
if res ~= 0
  set(gca,'YDir','reverse');
end
