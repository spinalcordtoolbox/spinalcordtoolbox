function [ m_center_line,radius_interpolated,m_tangent_cord ] = scs_center_line_update(m_radius,m_center_line_user,nom_radius,m_center_line_previous,angles,scales,param)

% scs_center_line_update 
%   calculate the new centers after the radius update
%   calculate the new center-line associated with these new centers 
%   adjust the center-line with a shear force calculated with the center-line initialized by the user
%   calculate the new tangent vector normalized associated with the new center-line
%
% SYNTAX:
% [M_CENTER_LINE,RADIUS_NORM,M_TANGENT_CORD] = scs_center_line_update(M_RADIUS,M_CENTER_LINE_USER,NOM_RADIUS,M_CENTER_LINE_PREVIOUS,ANGLES,SCALES,PARAM)
%
% 
% INPUTS:
%
% M_RADIUS    
%   (number_of_slices x number_of_angles array) Radius smoothed
%
% CENTER_LINE_USER
%   (Nx3 array) Coordinates of the spinal cord for each slice of M_VOLUME
%
% NOM_RADIUS
%   Scalar nominal radius
%
% M_CENTER_LINE_PREVIOUS
%   (Nx3 array) Coordinates of the spinal cord center line for each previous slice of M_VOLUME
%
% ANGLES
%   (number_of_angles array) Contains the value of angles
%
% SCALES
%   (4x1 array) Scales in mm of X, Y and Z axes in M_VOLUME
%
% PARAM
%   Struct containing an assortment of user defined parameters (see user
%   guide for details)
%_________________________________________________________________________
% OUTPUTS:
%
% M_CENTER_LINE    
%   (Nx3 array) Coordinates of the spinal cord new center line for each slice of M_VOLUME
%
% RADIUS_NORM
%   (number_of_slices x number_of_angles array) Update of the radius with
%   the new centers according with the previous angles (intially defined)
%
% M_TANGENT_CORD
%   Coordinates of the vector unit tangent

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The core of the function starts here %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %   Detailed explanation goes here
% 
% % Version : January 10th, 2013
% 
%% Test 1: le cercle
% clear all
% clc
% close all
% num_angles = 100000;
% angles_g=0:2*pi()/(num_angles):2*pi()-1/(num_angles);
% % par rapport à (0,0)
% m_radius_g = ones(1,num_angles);
% m_x = m_radius_g.*cos(angles_g);
% m_y = m_radius_g.*sin(angles_g);
% % pour l'affichage
% x =[m_x m_x(1)];
% y =[m_y m_y(1)];
% figure
% plot(x,y)
% % on impose le centerline à (0.5,0.5);
% m_center_line = [0.8; -0.5;1];
% % rayon par rapport au centerline
% m_x = m_x - m_center_line(1);
% m_y = m_y - m_center_line(2);
% % on reconstruit m_radius
% m_radius = sqrt(m_x.^2+m_y.^2);
% % on reconstruit angles
% for i = 1 : size(m_x,2)
%     if m_x(i)>0 && m_y(i)>= 0
%         angles(i) = atan(m_y(i)/m_x(i));
%     elseif m_x(i)>0 && m_y(i)<0
%         angles(i) = atan(m_y(i)/m_x(i)) + 2*pi;
%     elseif m_x(i) <0
%         angles(i) = atan(m_y(i)/m_x(i))+pi;
%     elseif m_x(i)==0 && m_y>0
%         angles(i) = pi/2;
%     elseif m_x(i) == 0 && m_y(i) < 0
%         angles(i) = -pi/2;
%     end
% end
% m_center_line_previous = m_center_line;

%% Centroïd computation 

% Triangle decomposition of the domain
size_vector = size(m_center_line_previous,2);

% Initialization of centroïd
m_center_line = zeros(size_vector,3);
m_center_line_x = zeros(1);
m_center_line_y = zeros(1);

for i = 1 : size_vector           
    [x,y] = pol2cart(angles,m_radius(i,:));
    % in the image system of coordinate
    x = x + m_center_line_previous(1,i);
    y = y + m_center_line_previous(2,i);
    P1=[m_center_line_previous(1,i) m_center_line_previous(2,i)];
    for j = 1:size(m_radius,2)
        if j==size(m_radius,2)
            P2=[x(j) y(j)];
            P3=[x(1) y(1)];
            area(j) = triangle_area([P1;P2;P3]); % Compute triangle area
            % Compute the centroïd for the triangle in x and y
            
            Cx(j) = (m_center_line_previous(1,i) + x(j) + x(1))/3;
            Cy(j) = (m_center_line_previous(2,i) + y(j) + y(1))/3;
        else
            P2=[x(j) y(j)];
            P3=[x(j+1) y(j+1)];
            area(j) = triangle_area([P1;P2;P3]); % Compute triangle area
            % Compute the centroïd for the triangle in x and y
            
            Cx(j) = (m_center_line_previous(1,i) + x(j) + x(j+1))/3;
            Cy(j) = (m_center_line_previous(2,i) + y(j) + y(j+1))/3;
        end
    end
    m_center_line_x(i) = sum(Cx.*area)/sum(area); % global centroid in x
    m_center_line_y(i) = sum(Cy.*area)/sum(area); % global centroid in y
    % Compute the new center_line
    m_center_line(i,:) = [m_center_line_x(i) m_center_line_y(i) i]; 
end

%% Radius update according to the new center_line

% Initialization of matrix size for radius in cartesian coordinates
radius_x = zeros(size(m_radius));
radius_y = zeros(size(m_radius));
% Intialization of the size of m_radius_new, in the new referential 
m_radius_new = zeros(size(m_radius));
% Initialization of the matrix containing the radius updated from the
% interpolation with a cubic spline
radius_interpolated = zeros(size(m_radius_new));

% Transformation of m_radius, containing the radius of each angle (see
% angles matrix) and slice in cartesian coordinates.
% The result is the contour of the spinal cord in cartesian coordinates.
for i=1:size(m_center_line_previous,2)
   [radius_x(i,:) radius_y(i,:)]=pol2cart(angles, m_radius(i,:));

% Translation from the old center to the new center in x
   translation_x = m_center_line(:,1)-m_center_line_previous(1,:)'; %$%$%$%$%$%$%$%$%$%$%$% pas absolue parce qu'il ne tient pas compte des négatifs
% Translation from the old center to the new center in y
   translation_y = m_center_line(:,2)-m_center_line_previous(2,:)';
   
% Expression of the radius and angles in polar coordinates associated with
% the new center line 
    radius_x(i,:) = radius_x(i,:)-translation_x(i); % we applied one translation to a slice
    radius_y(i,:) = radius_y(i,:)-translation_y(i);
    
% Transformation of m_radius in polar coordinates considering the change
% with the new center
    [m_angles_new, m_radius_new(i,:)] = cart2pol(radius_x(i,:), radius_y(i,:));

% Check if there are angles that should be positive intead of negative and
% vice versa
    angles_diff = m_angles_new-angles;
    for p=1:length(angles)
        if angles_diff(p) > pi()
            m_angles_new(p) = m_angles_new(p) - 2*pi();
        elseif angles_diff(p) < -pi()
            m_angles_new(p) = m_angles_new(p) + 2*pi();
        end
    end
    
    pp = spline(m_angles_new,[radius_x(i,:);radius_y(i,:)]);
    xy_coord = ppval(pp, angles);
    
    [m_angles_new2, radius_interpolated(i,:)] = cart2pol(xy_coord(1,:), xy_coord(2,:));
    
% Transformation to make the angles in radians positive 
    %m_angles_new = m_angles_new + pi-pi/size(angles,2);
%     m_angles_new = wrapToPi(m_angles_new);

% Ascending sorting of the angles with their corresponding radius
% 	m_radius_angles_sorted = sortrows([m_radius_new(i,:)' m_angles_new'],2); % sorting with column 2, which is the angles in radians
%     m_radius_new(i,:) = m_radius_angles_sorted(:,1)';  % transpose and keep in memory
%     m_angles_new = m_radius_angles_sorted(:,2)'; % transpose
% 
% % Checking if there is similar angles with a tolerance of 1e-02
%     for k=1:2:size(m_angles_new,2)-1
%         if abs(m_angles_new(k)-m_angles_new(k+1))<1e-02 && (m_angles_new(k)<0)
%             m_angles_new(k+1)=m_angles_new(k+1)+2*pi();
%         elseif abs(m_angles_new(k)-m_angles_new(k+1))<1e-02 && (m_angles_new(k)>0)
%             m_angles_new(k+1)=m_angles_new(k+1)-2*pi();
%         end
%     end
%         
% % Ascending sorting of the angles with their corresponding radius
% 	m_radius_angles_sorted = sortrows([m_radius_new(i,:)' m_angles_new'],2); % sorting with column 2, which is the angles in radians
%     m_radius_new(i,:) = m_radius_angles_sorted(:,1)';  % transpose and keep in memory
%     m_angles_new = m_radius_angles_sorted(:,2)'; % transpose
%     
%     
%     
% %     m_angles_new(end)=-m_angles_new(1);
% %     m_radius_new(end)=m_radius_new(1);
%     
% %     m_angles_new(1)=-m_angles_new(end);
% %     m_radius_new(1)=m_radius_new(end);
%     
% % Interpolation of each slice contour with a cubic spline to get more 
% % contour coordinates which is necessary to get the new radius associated 
% % with the initial angles and the new center_line
%     radius_interpolated(i,:) = interp1(m_angles_new, m_radius_new(i,:),angles,'spline');
    %$%$%$%$%$%$%$ Résultat des rayons : de 3,2972 à 5,3449 -> ce qui a de
    %l'allure - il reste à tester...
  
end
% radius_interpolated(:,1) = m_radius (:,1);   % Enlève la réinterpolation du 1er angle qui cause l'apparition d'un pic.

m_center_line = m_center_line';


fc_x=(4*(m_center_line_user(1,:)-m_center_line(1,:)).*abs(m_center_line_user(1,:)-m_center_line(1,:)))/nom_radius; % in pixels
fc_y=(4*(m_center_line_user(2,:)-m_center_line(2,:)).*abs(m_center_line_user(2,:)-m_center_line(2,:)))/nom_radius; % in pixels
fc=[fc_x; fc_y];

m_center_line = [m_center_line([1 2],:) + param.shear_force_multiplier * fc; m_center_line(3,:)];

% Calcul du vecteur tangent lié à la nouvelle ligne centrale.
x=m_center_line(1,:);
y=m_center_line(2,:);

% Unit vector tangent
%   Computation of the derivatives
dx=x(2:size(x,2))-x(1:size(x,2)-1);
dy=y(2:size(y,2))-y(1:size(y,2)-1);
dz=ones(size(dx,2),1)*scales(3);
%   Computation of the norm
m_tangent_cord = [dx' dy' dz];

%   Convert in mm
m_tangent_cord(:,1)=m_tangent_cord(:,1)*scales(1);
m_tangent_cord(:,2)=m_tangent_cord(:,2)*scales(2);

norm_tangent=sqrt(sum(m_tangent_cord.*m_tangent_cord, 2));  %euclidian norm

%   Computation of the unit vector tangent
m_tangent_cord(:,1)=m_tangent_cord(:,1)./norm_tangent;
m_tangent_cord(:,2)=m_tangent_cord(:,2)./norm_tangent;
m_tangent_cord(:,3)=m_tangent_cord(:,3)./norm_tangent;

%   Convert in pixels
m_tangent_cord(:,1)=m_tangent_cord(:,1)/scales(1);
m_tangent_cord(:,2)=m_tangent_cord(:,2)/scales(2);

% Transpose matrix (modification par J et Y)
m_tangent_cord = m_tangent_cord';
m_tangent_cord = [m_tangent_cord m_tangent_cord(:,end)]; %Add another tangent to have the same length as m_center_line

% end % END OF SCS_CENTER_LINE_UPDATE_V2


