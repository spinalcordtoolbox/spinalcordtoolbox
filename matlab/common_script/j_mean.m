function mean_data = j_mean(data,x0,xn,y0,yn,z0,zn)

% [j_mask] = j_mask(x0,xn,y0,yn,z0,zn)
%
% This function calculates the mean of values contained within a mask
% applied to a 4d matrix.
%
% INPUTS
% data              4d matrix
% x0                start value of the mask for x
% xn                end value of the mask for x
% y0                start value of the mask for y
% yn                end value of the mask for y
% z0                start value of the mask for z
% zn                end value of the mask for z
%
% OUTPUTS
% mean_data         mean of the values
%
% DEPENDENCES
% (-)
%
% COMMENTS
% Julien Cohen-Adad 22/01/2006




xmax = size(data(:,1,1,1),1);
ymax = size(data(1,:,1,1),2);
zmax = size(data(1,1,:,1),3);
tn = size(data(1,1,1,:),4);
m=0;

% start timer
tic

fprintf('Calculate mean. Please wait...');

% construct a 1d-matrix with all values
for i = x0:xn
    for j = y0:yn
        for k = z0:zn
            for l = 1:tn
                m=m+1;
                A(m) = data(i,j,k,l);
            end
        end
    end
end

% calculate the mean
mean_data = mean(A);
var_data = var(A);

% display elapsed time
fprintf(' OK (elapsed time %s seconds) \n',num2str(toc));
