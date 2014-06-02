% =========================================================================
% FUNCTION
% j_centerOfMass
%
% Compute center of mass.
%
% INPUT
% data              3d array.
%
% OUTPUT
% centmass          float.
%
% COMMENTS
% julien cohen-adad 2009-02-10
% =========================================================================
function centmass = j_centerOfMass(data)


[nx ny nz] = size(data);

for ix=1:nx
    data_x(ix) = mean(mean(data(ix,:,:)));
end
ind_x = (1:nx);
centmass(1) = sum(ind_x.*data_x)/sum(data_x);
    
for iy=1:ny
    data_y(iy) = mean(mean(data(:,iy,:)));
end
ind_y = (1:ny);
centmass(2) = sum(ind_y.*data_y)/sum(data_y);
    
for iz=1:nz
    data_z(iz) = mean(mean(data(:,:,iz)));
end
ind_z = (1:nz);
centmass(3) = sum(ind_z.*data_z)/sum(data_z);
