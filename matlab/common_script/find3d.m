function [X,Y,Z]=find3d(input)
dimY=size(input,2);
[X,Y] = find(input);
Z = floor(Y/dimY);
Y = Y - dimY*Z;
Z = Z + 1;