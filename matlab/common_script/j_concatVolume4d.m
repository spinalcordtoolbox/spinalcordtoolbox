function [volume4d_concat] = j_concatVolume4d(volume4d,nb_volumes)

% [volume4d_concat] = j_concatVolume4d(volume4d,nb_volumes)
%
% This function concatenates extrem values of a volume4d to avoid edge
% artefacts in temporal filtering. This concatenation occurs for the
% TEMPORAL DIMENSION ONLY
%
% INPUTS
% volume4d          4d-volume
% nb_volumes        number of first volumes to concatenate (the same number
%of last volumes will be added)
%
% OUTPUTS
% volume4d_concat   new 4d-volume
%
% DEPENDENCES
% (-)
%
% COMMENTS
% Julien Cohen-Adad 13/11/2005


size_volume4d = size(volume4d,4);
volume4d_concat = volume4d;

% concatenates last volumes
for i = 1:nb_volumes
    volume4d_concat = cat(4,volume4d_concat,volume4d_concat(:,:,:,size_volume4d));
end

% concatenates first volumes
for i = 1:nb_volumes
    volume4d_concat = cat(4,volume4d_concat(:,:,:,1),volume4d_concat);
end

