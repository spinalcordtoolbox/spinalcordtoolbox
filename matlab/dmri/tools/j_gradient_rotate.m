% =========================================================================
% FUNCTION
% j_gradient_rotate()
%
% Rotate gradient list for dMRI studies.
%
% INPUT
% (fname)				/poufpouf/bvecs.txt
% (rot_angle)			[rx ry rz]
% (replace)				binary. Replace file
% 
% OUTPUT
% (-)
%
% COMMENTS
% julien cohen-adad 2007-11-28
% =========================================================================
function j_gradient_rotate(fname,rot_angle,replace_usr)


% initialization
fname_gradient      = '/Users/julien/mri/hreflex/pilote_02/average_01-04/bvecs_mediria.txt';
angle_x             = 90; % X rotation angle in degree
angle_y             = -90; % Y rotation angle in degree
angle_z             = 0;
suffixe_write       = '_rot';
replace				= 0

% user initialization
if nargin>0, fname_gradient = fname; end
if nargin>1
	angle_x = rot_angle(1);
	angle_y = rot_angle(2);
	angle_z = rot_angle(3);
end
if nargin>2, replace = replace_usr; end

% load gradient file
opt.read_method		= 'linePerLine';
opt.drop_line		= 1;
[gradient fname_gradient] = j_readFile(fname_gradient,opt);
if ~gradient, return; end
nb_directions = size(gradient,1)-1;

% compute rotation matrices
a_x = angle_x*pi/180;
a_y = angle_y*pi/180;
a_z = angle_z*pi/180;
rot_x = [1 0 0 ; 0 cos(a_x) sin(a_x) ; 0 -sin(a_x) cos(a_x)]; % X rotation
rot_y = [cos(a_y) 0 -sin(a_y) ; 0 1 0 ; sin(a_y) 0 cos(a_y)]; % Y rotation of a_y (in gradient)
rot_z = [cos(a_z) sin(a_z) 0 ; -sin(a_z) cos(a_z) 0 ; 0 0 1];

% rotate each gradient vector
gradient = rot_x*gradient';
gradient = gradient';
gradient = rot_y*gradient';
gradient = gradient';
gradient = rot_z*gradient';
gradient = gradient';

if replace
	fname_write = fname;
else
	% write new gradient file using suffixe
	suffixe = strcat(suffixe_write,'X',num2str(angle_x),'Y',num2str(angle_y),'Z',num2str(angle_z));
	[a b c]=fileparts(fname_gradient);
	file_write = strcat(b,suffixe,c);
	fname_write = strcat(a,filesep,file_write);
end
fid_w = fopen(fname_write,'w');
fprintf(fid_w,'%i\n',nb_directions+1); % number of directions + 1
fprintf(fid_w,'0 0 0\n'); % B0 value
for i=2:nb_directions+1
    G = [gradient(i,1),gradient(i,2),gradient(i,3)];
    fprintf(fid_w,'%1.10f %1.10f %1.10f\n',G(1),G(2),G(3));
end
fclose(fid_w);

% display output
if ~replace
	fprintf('-> file created: %s\n',file_write);
end
