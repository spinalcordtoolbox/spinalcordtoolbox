function f = matchgain(data1,data2)

% function f = matchgain(data1,data2)
%
% <data1> is cases x amplitudes
% <data2> is cases x amplitudes
%
% return <f>, a vector with dimensions cases x 1.  these are
% scalar factors such that <f>.*<data2> is closest in the 
% squared-error sense to <data1>.
%
% example:
% x = randn(1,100);
% y = x*4 + .3*randn(1,100);
% f = matchgain(x,y);
% figure; hold on;
% plot(x,'r-'); plot(y,'g-');
% figure; hold on;
% plot(x,'r-'); plot(f*y,'g-');

f = 1./dot(data2,data2,2) .* dot(data2,data1,2);
