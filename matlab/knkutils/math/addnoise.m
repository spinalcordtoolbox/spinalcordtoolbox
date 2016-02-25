function f = addnoise(m,amt)

% function f = addnoise(m,amt)
%
% <m> is a matrix
% <amt> is [A B] where A and B are positive values with A < B
%
% add between A% and B% of noise to <m> (each element treated independently).
% the noise is randomly positive or negative.
% the noise amount is chosen from a flat distribution between A and B.
%
% example:
% addnoise([1 2 3],[1 2])

ramt = m .* normalizerange(rand(size(m)),amt(1)/100,amt(2)/100,0,1);
rsign = (2*(rand(size(m))>.5))-1;  % -1 or 1
f = m + rsign .* ramt;
