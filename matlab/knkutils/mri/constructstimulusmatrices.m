function f = constructstimulusmatrices(m,prenumlag,postnumlag,wantwrap)

% function f = constructstimulusmatrices(m,prenumlag,postnumlag,wantwrap)
%
% <m> is a 2D matrix, each row of which is a stimulus sequence (i.e.
%   a vector that is all zeros except for ones indicating the onset
%   of a given stimulus (fractional values are also okay))
% <prenumlag> is the number of stimulus points in the past
% <postnumlag> is the number of stimulus points in the future
% <wantwrap> (optional) is whether to wrap around.  default: 0.
%
% return a stimulus matrix of dimensions
% size(m,2) x ((prenumlag+postnumlag+1)*size(m,1)).
% this is a horizontal concatenation of the stimulus
% matrix for the first stimulus sequence, the stimulus
% matrix for the second stimulus sequence, and so on.
% this function is useful for fitting finite impulse response (FIR) models.
%
% history:
% 2013/05/12 - update doc to indicate fractional values are okay.
%
% example:
% imagesc(constructstimulusmatrices([0 1 0 0 0 0 0 0 0; 0 0 1 0 0 0 0 0 0],0,3));

% input
if ~exist('wantwrap','var') || isempty(wantwrap)
  wantwrap = 0;
end

% get out early
if prenumlag==0 && postnumlag==0
  f = m';
  return;
end

% do it
num = prenumlag + postnumlag + 1;
f = zeros([size(m,2) num*size(m,1)]);
for p=1:size(m,1)
  f(:,(p-1)*num+(1:num)) = constructstimulusmatrix(m(p,:),prenumlag,postnumlag,wantwrap);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% HELPER FUNCTION

function f = constructstimulusmatrix(v,prenumlag,postnumlag,wantwrap)

% function f = constructstimulusmatrix(v,prenumlag,postnumlag,wantwrap)
%
% <v> is the stimulus sequence represented as a vector
% <prenumlag> is the number of stimulus points in the past
% <postnumlag> is the number of stimulus points in the future
% <wantwrap> (optional) is whether to wrap around.  default: 0.
%
% return a stimulus matrix of dimensions
% length(v) x (prenumlag+postnumlag+1)
% where each column represents the stimulus at
% a particular time lag.

% input
if ~exist('wantwrap','var') || isempty(wantwrap)
  wantwrap = 0;
end

% do it  
total = prenumlag + postnumlag + 1;
f = zeros([length(v) total]);
for p=1:total
  if wantwrap
    f(:,p) = circshift(v,[0 -prenumlag+(p-1)]).';
  else
    temp = -prenumlag+(p-1);
    if temp < 0
      f(1:end+temp,p) = v(1-temp:end);
    else
      f(temp+1:end,p) = v(1:end-temp);
    end
  end
end
