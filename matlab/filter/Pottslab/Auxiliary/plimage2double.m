function img = plimage2double( plimg )
%PLIMAGE2DOUBLE Summary of this function goes here
%   Detailed explanation goes here
img = reshape(plimg.toDouble(), [plimg.mRow, plimg.mCol, plimg.mLen]);

end

