function f = circularshiftandsum(v,n)

% function f = circularshiftandsum(v,n)
% 
% <v> is a vector of values
% <n> is a positive integer
% 
% circularly shift <v> for <n> times and sum across these instances.
% the shifts are equally spaced (rounding to nearest position if necessary).
%
% example:
% x = [0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0];
% figure; plot(circularshiftandsum(x,1),'r-');
% figure; plot(circularshiftandsum(x,4),'b-');

% calc
len = length(v);

% do it
shifts = round((0:n-1)/n * len);
f = 0;
for p=1:length(shifts)
  f = f + circshift(v,[0 shifts(p)]);
end
