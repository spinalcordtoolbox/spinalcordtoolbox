function str=hexstr(n)
% str=hexstr(n)
% Convert any number to a hex string representing the lower 32 bits,
% emulating the C format %lx for a long. This works with both positive and
% negative numbers, unlike the MATLAB format %x (and %tx and %bx) which
% can't deal with negative numbers.
% 
% fprintf('Error 0x%s.\n',hexstr(err));
% 
% See also DEC2HEX, FPRINTF.

% 4/24/05 dgp Wrote it.

h=dec2hex(n+2^33);
str=h(end-7:end);
