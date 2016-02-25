function f = GPUok

% function f = GPUok
%
% based on the hostname and possibly a random flip, return whether we should try to use the GPU.
%
% example:
% GPUok

temp = gethostname;
   % azure ~ 3, chroma ~ 1.5
f = (isequal(temp,'azure.stanford.edu') & rand < 3/32) | (isequal(temp,'chroma.stanford.edu') & rand < 1.5/4);
