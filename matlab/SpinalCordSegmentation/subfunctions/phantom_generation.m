%%
% This codes generates a semi-random phantom of the spinal cord. The
% position of the center-line is defined by the fx and fy function and
% respectively indicate the x and y position of the center of the spinal
% cord along the z direction. 

%% Creating a phantom of the spinal cord
clear all; close all; clc
dims=[50 50 100 1];  %definition of the phantom size
theta=64;   %number of angles used to generate the phantom
m_phantom = ones(dims)*0.8;
a=3.75*ones(dims(3),1)+0.5*sin((1:dims(3))/8)'; %radius dimensions in x(z)
b=4.25*ones(dims(3),1)+cos((1:dims(3))/12)'; %radius dimensions in y(z)
 
fx=round(dims(1)/2+10*dims(3)^-2*((1:dims(3))-dims(3)/2).^2);   %position of the x component of the center_line (z)
fy=round(dims(2)/2+30*dims(3)^-3*((1:dims(3))-dims(3)/2).^3);   %position of the y component of the center_line (z)

angle=0:2*pi()/theta:2*pi()-2*pi()/theta;
% generation of the spinal cord
for z=1:dims(3)
    x=a(z)*cos(angle);
    y=b(z)*sin(angle);
    x=round(x);
    y=round(y);
    c=1;
    for j=y
        m_phantom(fx(z)-x(c):fx(z)+x(c),fy(z)-j:fy(z)+j,z)=0.2;
        c=c+1;
    end
end   
scs_slider_phantom(m_phantom)   %visualisation

%% Adding a poisson noise and filtering 
% definition of the gaussian mask
gaussian_size = 3;
sigma = 0.7;
gaussian_mask_2D = fspecial('gaussian',gaussian_size,sigma);
gaussian_mask_1D = fspecial('gaussian',[gaussian_size 1],sigma);
for i=1:gaussian_size
    gaussian_mask_3D(:,:,i)=gaussian_mask_2D*gaussian_mask_1D(i);
end

% application of the gaussian mask
m_phantom_gauss = convn(m_phantom, gaussian_mask_3D, 'same');
% scs_slider_phantom(m_phantom_gauss)

% adding poisson noise
lambda = 50;
m_phantom_noise = random('poiss',m_phantom_gauss*lambda,size(m_phantom_gauss))/lambda;
% scs_slider_phantom(m_phantom_noise)

% application of the gaussian mask
m_phantom_final = convn(m_phantom_noise, gaussian_mask_3D, 'same');

% visualisation
scs_slider_phantom(m_phantom_final)

m_phantom=m_phantom_final;



%%
save m_phantom m_phantom dims

%% Visualisation of fx, fy, a and b
figure
subplot(1,2,1); plot(fx); title('fx')
subplot(1,2,2); plot(fy); title('fy')

figure
subplot(1,2,1); plot(a); title('a');
subplot(1,2,2); plot(b); title('b');
