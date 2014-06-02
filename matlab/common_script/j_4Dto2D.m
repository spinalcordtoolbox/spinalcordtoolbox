function [data2D] = j_4Dto2D(data4D)

% [data2D] = j_4Dto2D(data4D)
%
% Convert 4D-volume into 2D-matrix, one dimension is space, other dimension
% is time
%
%
% INPUT
% data4D    4-D data
%
% OUTPUT
% data2D    2-D data
%
% DEPENDANCES
% (-)
%
% COMMENTS
% Julien Cohen-Adad 13/11/2005


data2D = [];

% fout la taille de la premiere dimension dans nx, etc.
[nx,ny,nz,nt] = size(data4D);

% fout la matrice 4D ds une matrice 2D
data2D = reshape(data4D,nx*ny*nz,nt);
