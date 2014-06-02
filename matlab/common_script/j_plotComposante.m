function j_plotComposante(file_sica,composante)

% visualisation of sICA time courses fora given file and composante
%
% INPUTS
% file_sica     file name
% composante    number of composante to display
%
% OUTPUTS
% (-)
%
% COMMENTS
% Julien Cohen-Adad 01/12/05


% parameters to change if necessary
TR = 2;
nb_scans = 300;

load(file_sica);
timeCourse = sica.A((composante-1)*300+1:(composante-1)*300+300);
t = linspace(0,TR*size(timeCourse,2),size(timeCourse,2));
figure
plot(t,timeCourse)
xlabel('time (s)')
name{1}=num2str(composante);
title(strcat('Time course : composante #',name{1}))
zoom on
 