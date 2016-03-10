function statusdots(p,total,num)

% function statusdots(p,total,num)
% 
% <p> is the index (starting from 1)
% <total> is the total number of things to process
% <num> (optional) is the desired number of dots.  default: 20.
%
% depending on the value of <p>, fprintf out an '.'
% such that we write out a total of <num> dots,
% equally spaced as best as possible.  we start
% with a dot when <p> is 1.
%
% example:
% fprintf('starting');
% for p=1:57
%   statusdots(p,57,10);
%   pause(.1);
% end
% fprintf('done.\n');

% input
if ~exist('num','var') || isempty(num)
  num = 20;
end

% do it
points = round(linspacecircular(1,total+1,num));
if ismember(p,points)
  fprintf('.');
end
