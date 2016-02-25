function f = renderfigure(res,wantgray)

% function f = renderfigure(res,wantgray)
%
% <res> is number of pixels along one side
% <wantgray> (optional) is
%   0 means to render the old way and do not explicitly convert to grayscale.
%   1 means to render the old way and explicitly convert to grayscale.
%   2 means to render the new way.
%   default: 0.
%
% render the current figure and return the result with values in [0,1].
% rendering is accomplished using ImageMagick/Ghostscript.
% we assume that the field-of-view is supposed to be square.
% we use a (temporary) figure size of max(500,<res>*2) along each dimension;
% this is because the ImageMagick/Ghostscript rendering quality depends on the size of 
% the figure that is embedded in the .eps file.  empirically, it appears
% that max(500,<res>*2) gives good anti-aliasing results.
% note that in the course of processing, we turn off the axis and 
% change the position of the axis.
%
% example:
% figure; drawbar(0,.2,.1,pi/6,.1,.4,[.5 .1 .3],[1 1 1]);
% temp = renderfigure(500); figure; imagesc(temp); axis equal tight;

% input
if ~exist('wantgray','var') || isempty(wantgray)
  wantgray = 0;
end

% make invisible
prev = get(gcf,'Visible');
set(gcf,'Visible','off');

% make nice and big and square
pos = getfigurepos;
setfigurepos([0 0 max(500,res*2) max(500,res*2)]);
if res*2 > 10000
  error('res*2 is probably too big...')
end

% remove axis and prep position of axis (to reduce weird boundingbox issues)
axis off;
set(gca,'Position',[0 0 1 1]);

% write .eps to a temporary file
tmpfile = tempname;
printnice([],[0 1],[],tmpfile);

% convert eps to png
if ismember(wantgray,[0 1])
  opt = choose(wantgray,'-type Grayscale','');
  assert(unix(sprintf('convert %s.eps -scale %dx%d! %s %s.png',tmpfile,res,res,opt,tmpfile))==0);
  fct = 65535;
else
        %%%"-r72x72" -g1199x1199 
  assert(unix(sprintf('gs -q -dQUIET -dPARANOIDSAFER -dEPSCrop -dSAFER -dBATCH -dNOPAUSE -dNOPROMPT -dDOINTERPOLATE -dAlignToPixels=0 -dGridFitTT=0 -dGraphicsAlphaBits=4 -dTextAlphaBits=4 -sDEVICE=pngalpha -sOutputFile=%s.png %s.eps',tmpfile,tmpfile))==0);
  assert(unix(sprintf('convert %s.png -scale %dx%d! %s.png',tmpfile,res,res,tmpfile))==0);
  fct = 255;
end

% read it in
f = double(imread(sprintf('%s.png',tmpfile)))/fct;

% restore position and visibility
setfigurepos(pos);
set(gcf,'Visible',prev);
