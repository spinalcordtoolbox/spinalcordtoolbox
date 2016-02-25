function f = inputmulti

% function f = inputmulti
%
% capture input from stdin until '.' is entered on a line by 
% itself.  return a string vector that represents the input received.
%
% note that newlines (\n) are embedded in the string vector, and
% the last newline is ignored.

% init
isfirst = 1;
f = '';

% do it
in = input('(''.'' by itself terminates the input)\n','s');
while ~isequal(in,'.')
  if isfirst==1
    f = in;
    isfirst = 0;
  else
    f = [f sprintf('\n') in];
  end
  in = input('','s');
end
