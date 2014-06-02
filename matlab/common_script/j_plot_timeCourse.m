function j_plot_timeCourse(donnees2D, voxel)

% visualisation of sICA time courses
%
%
% INPUTS
% donnees2D     matrix 2D of data
% voxel         the voxel number to visualize 
%
% OUTPUTS
% (-)
%
% COMMENTS
% Julien Cohen-Adad 11/11/05


TR = 2;
a = donnees2D(voxel,:);
t = linspace(0,TR*size(a,2),size(a,2));
figure
plot(t,a)
xlabel('time (s)')
name{1}=num2str(voxel);
title(strcat('Time course : voxel #',name{1}))
zoom on
 