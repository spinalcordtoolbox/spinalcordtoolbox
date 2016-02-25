function f = straightline(value,direction,linestyle,rng)

% function f = straightline(value,direction,linestyle,rng)
%
% <value> is a vector of values
% <direction> is
%   'h' or 'y' means horizontal lines
%   'v' or 'x' means vertical lines
% <linestyle> is like 'r:'
% <rng> (optional) is [A B] with the range to restrict to.
%   for example, when <direction> is 'h', then <rng> being
%   [1 3] means to restrict the horizontal extent of the line
%   to between 1 and 3.  if [] or not supplied, use the 
%   full range of the current axis.
%
% draw lines on the current figure.
% return a vector of handles.
%
% example:
% figure;
% h = straightline(randn(1,10),'v','r-');
% set(h,'LineWidth',2);

% input
if ~exist('rng','var') || isempty(rng)
  rng = [];
end

% hold on
prev = ishold;
hold on;

% do it
ax = axis;
f = zeros(1,length(value));
switch direction
case {'h' 'y'}
  if isempty(rng)
    rng = ax(1:2);
  end
  for p=1:length(value)
    f(p) = plot(rng,[value(p) value(p)],linestyle);
  end
case {'v' 'x'}
  if isempty(rng)
    rng = ax(3:4);
  end
  for p=1:length(value)
    f(p) = plot([value(p) value(p)],rng,linestyle);
  end
otherwise
  error('invalid <direction>');
end

% hold off
if ~prev
  hold off;
end
