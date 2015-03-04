function [m_phantom] = scs_phantom_gen(dims, mean_rad, dev_rad, dev_center, is_gauss_filt, gauss_noise, T)
% dims          [xdim ydim zdim thetadim] : 
%   Gives the dimensions of the phantom in x, y and z plus the number of
%   angles used to construct it.
%
% mean_rad      [xrad_mean yrad_mean] : 
%   Gives the mean radius in the x and y directions.
% 
% dev_rad       [xrad_dev yrad_dev] :
%   Gives the amplitude of the deviation to the mean_rad.
%
% dev_center    [xcntr_dev ycntr_dev]
%   Gives the amplitude of the variation of the center-line along z.
%
% is_gauss_filt
%   Gives the indication to apply or not a gaussian filter to the image.
%   is_gauss_filt = 1 -> gaussian filter applied
%   is_gauss_filt = 0 -> gaussian filter not applied
%
% gauss_noise   
%   Gives the amplitude of the gaussian noise applied. Between 0 and 1. If 
%   no noise is wanted, put this parameter to 0. 
% 
% 


% Definitions of some variables
xdim = dims(1);
ydim = dims(2);
zdim = dims(3);
thetadim = dims(4); 
angle = 0:2*pi()/thetadim:2*pi()-2*pi()/thetadim;
a0 = mean_rad(1);
b0 = mean_rad(2);

% Definition of the gaussian noise so that its amplitude gives an adequate
% noise.
if gauss_noise > 1
    gauss_noise = .4;
elseif gauss_noise < 0
    gauss_noise = 0;
else
    gauss_noise = gauss_noise * .4;
end

% Initialisation of m_phantom, the matrix containing the output image
m_phantom = ones(xdim,ydim,zdim);

% Evolution of the radii (a->x b->y) along z 
a = a0*ones(zdim,1)+dev_rad(1)*sin((1:zdim)/2)';        % radius dimensions in x(z)
b = b0*ones(zdim,1)+dev_rad(2)*cos((1:zdim)/2)';        % radius dimensions in y(z)

% Evolution of the center line along z
fx = round(xdim/2+dev_center(1)*zdim^-2*((1:zdim)-zdim/2).^2);       %position of the x component of the center_line (z)
fy = round(ydim/2+dev_center(2)*zdim^-3*((1:zdim)-zdim/2).^3);      %position of the y component of the center_line (z)

% Generation of the spinal cord
for z = 1:dims(3)
    x = round(a(z)*cos(angle));
    y = round(b(z)*sin(angle));
    c = 1;
    for j=y
        xind = fx(z)-x(c):fx(z)+x(c);
        yind = fy(z)-j:fy(z)+j;
        zind = z;
        % Correction if the radius/centerline goes beyond the dims of the image
        xind(xind>xdim) = xdim;
        yind(yind>ydim) = ydim;
        xind(xind<=0) = 1;
        yind(yind<=0) = 1;
        m_phantom(xind,yind,zind) = 0;
        c = c+1;
    end
end   

if is_gauss_filt==1
    % Definition of the gaussian mask
    gaussian_size = 3;
    sigma = 0.7;
    gaussian_mask_2D = fspecial('gaussian',gaussian_size,sigma);
    gaussian_mask_1D = fspecial('gaussian',[gaussian_size 1],sigma);
        for i=1:gaussian_size
            gaussian_mask_3D(:,:,i) = gaussian_mask_2D*gaussian_mask_1D(i);
        end
    % Application of the gaussian mask
    m_phantom = convn(m_phantom, gaussian_mask_3D, 'same');
    mu = .2;
    % Application of the gaussian noise
    sigma = gauss_noise;
    m_phantom_noise = random('Normal',mu,sigma,xdim,ydim,zdim);
    m_phantom = m_phantom_noise + m_phantom;
elseif is_gauss_filt == 0
    % Application of the gaussian noise
    mu = .2;
    sigma = gauss_noise;
    m_phantom_noise = random('Normal',mu,sigma,xdim,ydim,zdim);
    m_phantom = m_phantom_noise + m_phantom;
end

% Put the intensities between 0 and 1
m_phantom = mat2gray(m_phantom);

if T==1
    m_phantom=1-m_phantom;
end

