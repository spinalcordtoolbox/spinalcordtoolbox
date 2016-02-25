function f = evaldoublegamma(params,x)

% function f = evaldoublegamma(params,x)
%
% <params> is [n1 t1 a1 n2 t2 a2 delay d] where
%   <n1> is the power
%   <t1> is the time constant for the exponential
%   <a1> is the amplitude
%   <n2> is the power
%   <t2> is the time constant for the exponential
%   <a2> is the amplitude
%   <delay> is the delay from 0, such that the entire double gamma
%     is shifted forward by <delay> and then any x-values less than
%     <delay> get assigned a y-value of <d>
%   <d> is the offset
% <x> is a matrix of values to evaluate at.
%
% evaluate the double-gamma function at <x>.
%
% example:
% x = -1:.1:20;
% y = evaldoublegamma([1 1 1 1 2 .2 0 0],x);
% yn = y + 0.1*randn(size(y));
% [params,d,d,exitflag,output] = lsqcurvefit(@(a,b) evaldoublegamma([a 0 0],b),ones(1,6),x,yn,[],[],defaultoptimset);
% figure; hold on;
% plot(x,y,'r-');
% plot(x,yn,'m-');
% plot(x,evaldoublegamma([params 0 0],x),'g-');

% input
n1 = params(1);
t1 = params(2);
a1 = params(3);
n2 = params(4);
t2 = params(5);
a2 = params(6);
delay = params(7);
d = params(8);

% do it
x0 = x-delay;
f = a1 * x0.^n1 .* exp(-x0/t1) - a2 * x0.^n2 .* exp(-x0/t2) + d;
f(x <= delay) = d;
