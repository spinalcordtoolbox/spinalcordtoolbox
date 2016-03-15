function psychrethrow(msg)
% psychrethrow(msg) - Replacement for Matlab 6.5s builtin rethrow().
% This is hopefully useful for older Matlab installations and
% for the Octave port:
%
% If the rethrow-function is supported as builtin function on
% your Matlab installation, this function will call the builtin
% "real" rethrow function.
%
% If your Matlab lacks a rethrow-function, this function
% will try to emulate the real rethrow function.

if exist('rethrow', 'builtin')==5
  % Call builtin implementation:
  builtin('rethrow', msg);
else
  % Use our simple fallback-implementation:
  error(msg.message);
end

return;
