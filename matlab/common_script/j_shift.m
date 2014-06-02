% =========================================================================
% FUNCTION
% j_shift
%
% Shift 1-d signal. A positive shift means that the signal is shifted to
% the right.
%
% INPUT
% data          1xn float.
% shift         integer.
%
% OUTPUT
% datas         1xn float. Shifted data.
%
% COMMENTS
% Julien Cohen-Adad 2007-12-07
% =========================================================================
function [datas] = j_shift(data,shift)


if shift>0
    datas = cat(2,data(end-shift+1:end),data(1:end-shift));
elseif shift<0
    datas = cat(2,data(1-shift:end),data(1:-shift));
else
    datas = data;
end
