% =========================================================================
% FUNCTION
% j_minimize2d.m
%
% Minimize cost function - notably used for image rigid registration.
% 
% INPUT
% fname_data          string array.
%
% OUTPUT
% (-)
% 
% COMMENTS
% Julien Cohen-Adad 2008-07-23
% =========================================================================
function err = j_minimize2d(init,imsrc,imdest,cost_function,imsrc_new)


[nx ny] = size(imsrc);

% Apply transformation with various scale parameters until the correlation between the base and the aligned image is acceptable  
r = init(1);
tx = init(2);
ty = init(3);

Trotation = [cos(r) sin(r) 0; -sin(r) cos(r) 0; 0 0 1;];
Ttranslation = [1 0 0; 0 1 0; tx ty 1;];
T = Trotation*Ttranslation;

Tform = maketform('affine',T);
imsrc_new = imtransform(imsrc,Tform,'Xdata',[1 ny],'Ydata',[1 nx]);

% Compute cost function between the base and the aligned images
switch(cost_function)
	case 'Correlation'
		err = -abs(corr2(imsrc_new,imdest));

	case 'MI'
		mi = j_compute_MI(imsrc_new(:),imdest(:));
		err = 1/mi^2;
end
