function [f,mn,sd] = defineellipse3d(m,autocontrast,wantfit,mn,sd)

% function [f,mn,sd] = defineellipse3d(m,autocontrast,wantfit,mn,sd)
%
% <m> is a 3D volume
% <autocontrast> (optional) is whether to contrast-normalize
%   <m> for display purposes.  default: 1.
% <wantfit> (optional) is whether to obtain an initial position
%   and size by fitting a 3D Gaussian to <m>.  default: 1.
% <mn>,<sd> (optional) are the initial position and size to use.
%   if supplied, then we ignore <wantfit>.  if not supplied
%   and not <wantfit>, then we use a default position and size.
%
% open up a figure window and allow the user to control
% the position and size of a 3D ellipse.  we show in the
% figure window a maximum of 25 slices and we use the gray
% colormap.  the 3D ellipse is created by thresholding a 3D 
% Gaussian that is aligned with the coordinate axes.
% there are six parameters: three parameters controlling 
% the position (mean) along three dimensions and three 
% parameters controlling the size (standard deviation) 
% along three dimensions.
%
% the following are keys that can be pressed:
%   ws,ad,qe control decrementing and incrementing the 
%     mean along each of the three dimensions.
%   ik,jl,uo control decrementing and incrementing the
%     standard deviation along each of the three dimensions.
%   capital versions of any of the above keys achieve the
%     same effect, except that the step size is automatically
%     increased by a factor of five.
%   ' toggles between various visualization modes
%   - decreases the step size
%   = increases the step size
%   0 resets the position and size to a default value
%   ESC quits
%
% return:
%   <f> as a 3D volume with values that are either 0 or 1.
%       the 1 values constitute the final 3D ellipse.
%   <mn> is the final position of the ellipse, and is the same
%     format as in the input to makegaussian3d.m
%   <sd> is the final size of the ellipse, and is the same
%     format as in the input to makegaussian3d.m
%
% to generate the 3D ellipse from just the <mn> and <sd> parameters,
% you can do:
%   f = makegaussian3d(msize,mn,sd) > 0.5;
% where <msize> is the size of the 3D volume.
%
% example:
% vol = getsamplebrain(2);
% [f,mn,sd] = defineellipse3d(vol);
%
% history:
% 2011/03/09 - show max 25 slices
% 2010/10/16 - implement mn and sd inputs
% 2010/10/02 - change initial visualization state
% 2010/10/02 - the fitted Gaussian no longer has a dc fixed to 0 but has an exponent now.
% 2010/09/29 - change it so that when <wantfit>, the 3D Gaussian has an offset fixed to 0.

% internal constants
mul = 5;     % what factor does the shift key apply?
step = .02;  % what is the initial step size?
pct = 1;     % percentile for normalization
state = 2;   % see below
mix = .5;    % amount of ball to mix
ss = 20;     % low-resolution for fitting
mndefault = [.5 .5 .5];
sddefault = [.2 .2 .2];
maxslices = 25;  % maximum number of slices to show

% input
if ~exist('autocontrast','var') || isempty(autocontrast)
  autocontrast = 1;
end
if ~exist('wantfit','var') || isempty(wantfit)
  wantfit = 1;
end
if ~exist('mn','var') || isempty(mn)
  mn = [];
end
if ~exist('sd','var') || isempty(sd)
  sd = [];
end

% autocontrast
if autocontrast
  rng = prctile(m(:),[pct 100-pct]);
  m = normalizerange(m,0,1,rng(1),rng(2));
end

% define initial seed
if ~isempty(mn)
elseif wantfit
  mtemp = permute(processmulti(@(x) imresize(x,[ss ss]),permute(processmulti(@(x) imresize(x,[ss ss]),m),[3 1 2])),[2 3 1]);
  [params,r] = fitgaussian3d(mtemp);
  mn = normalizerange(params(1:3),0,1,1,ss,0);
  sd = params(4:6)/(ss-1);
else
  mn = mndefault;
  sd = sddefault;
end

% prep
xx = []; yy = []; zz = [];
msize = sizefull(m,3);
doupdate = 1;
mmx = max(m(:));
mmn = min(m(:));
mrng = mmx-mmn;
mmx = mmx + mrng/2;
mmn = mmn - mrng/2;

% do it
figure;
while 1

  if doupdate

    % make the ball volume
    [ball,xx,yy,zz] = makegaussian3d(msize,mn,sd,xx,yy,zz);
    ball = ball > .5;
  
    % make the weighted volume
    switch state
    case 0
      wvol = (1-ball) .* m;
    case 1
      wvol = m;
      wvol(ball) = mmn * mix + wvol(ball) * (1-mix);
    case 2
      wvol = m;
      wvol(ball) = mmx * mix + wvol(ball) * (1-mix);
    case 3
      wvol = ball .* m;
    end
  
    % show it
    imagesc(makeimagestack(wvol(:,:,round(linspace(1,size(wvol,3),min(maxslices,size(wvol,3))))))); colormap(gray); axis tight;%%equal tight;
  
  end
  title(sprintf('mn=%s, sd=%s, step=%s',mat2str(mn,5),mat2str(sd,5),mat2str(step,5)));
  
  % wait for the user
  keydown = waitforbuttonpress;
  doupdate = 1;
  if keydown
    switch get(gcf,'CurrentCharacter')

    case 'w'
      mn(1) = mn(1) - step;
    case 's'
      mn(1) = mn(1) + step;
    case 'a'
      mn(2) = mn(2) - step;
    case 'd'
      mn(2) = mn(2) + step;
    case 'q'
      mn(3) = mn(3) - step;
    case 'e'
      mn(3) = mn(3) + step;
    case 'i'
      sd(1) = sd(1) - step;
    case 'k'
      sd(1) = sd(1) + step;
    case 'j'
      sd(2) = sd(2) - step;
    case 'l'
      sd(2) = sd(2) + step;
    case 'u'
      sd(3) = sd(3) - step;
    case 'o'
      sd(3) = sd(3) + step;

    case 'W'
      mn(1) = mn(1) - mul*step;
    case 'S'
      mn(1) = mn(1) + mul*step;
    case 'A'
      mn(2) = mn(2) - mul*step;
    case 'D'
      mn(2) = mn(2) + mul*step;
    case 'Q'
      mn(3) = mn(3) - mul*step;
    case 'E'
      mn(3) = mn(3) + mul*step;
    case 'I'
      sd(1) = sd(1) - mul*step;
    case 'K'
      sd(1) = sd(1) + mul*step;
    case 'J'
      sd(2) = sd(2) - mul*step;
    case 'L'
      sd(2) = sd(2) + mul*step;
    case 'U'
      sd(3) = sd(3) - mul*step;
    case 'O'
      sd(3) = sd(3) + mul*step;
    
    case ''''
      state = mod(state+1,4);
    
    case '-'
      step = step / 2;
      doupdate = 0;
    case '='
      step = step * 2;
      doupdate = 0;
    
    case '0'
      mn = mndefault;
      sd = sddefault;

    case char(27)
      break;

    end
  end

end

% return
f = ball;
