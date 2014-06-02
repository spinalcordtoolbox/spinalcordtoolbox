function [volume4d_delete] = j_deleteVolume4d(volume4d,nb_volumes)

% [volume4d_delete] = j_deleteVolume4D(volume4d,nb_volumes)
%
% This function deletes extrem values of a volume4d after having filtered.
%ONLY TEMPORAL DIMENSION is taken in accompt
%
%
% INPUTS
% volume4d          4d-volume
% nb_volumes        number of first volumes to delete (the same number
%of last volumes will be deleted)
%
% OUTPUTS
% volume4d_delete   new 4d-volume
%
% DEPENDENCES
% (-)
%
% COMMENTS
% Julien Cohen-Adad 14/11/2005


size_volume4d = size(volume4d,4);
volume4d_delete = volume4d(:,:,:,(nb_volumes+1):(size_volume4d-nb_volumes));
%volume4d_delete = volume4d(:,:,:,1:(size_volume4d-2*nb_volumes));