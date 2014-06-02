function roi = j_createROI(dimx,dimy,dimz,x0,y0,z0,xn,yn,zn)

% This function creates a structure for a ROI
%
% INPUTS
% x,y,z             dimension of the matrix containing the ROI
% x0                start value of the ROI for x
% y0                start value of the ROI for y
% z0                start value of the ROI for z
% xn                end value of the ROI for x
% yn                end value of the ROI for y
% zn                end value of the ROI for z
%
% OUTPUTS
% mask3d            3d matrix mask which has same dimension as the 3d matrix
%
% DEPENDANCES
% (-)
%
% COMMENTS
% Julien Cohen-Adad 02/02/2006


roi = struct('dimx',dimx,'dimy',dimy,'dimz',dimz,'x0',x0,'y0',y0,'z0',z0,'xn',xn,'yn',yn,'zn',zn);
