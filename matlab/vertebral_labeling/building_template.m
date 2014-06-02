

% Build a template of generic intervertebral distance
%
% N.B. THE INPUT SHOULD BE .nii, not .nii.gz!!!
%The building_template script enable to build a vector of mean intervertebral distance needed by the labelling_vertebrae library.
%Intervertebral distances are averaged for each vertebral level over anatomical images given in inputs. 
%
%
% SYNTAX
% ========================================================================
% building_template
%
%
% INPUTS
% ========================================================================
% set of images
%
%
% OUTPUTS
% ========================================================================
%- mean_distance vector containing distance between intervertebral disk. The first
%  component correspond with the distance of C1, the second component
%  corespond to the distance of C2, etc...
%- std_distance vector containing standard deviation corresponding to the
%  means
%- figure of the distribution of means intervertebral distances 
%
% DEPENDENCES
% ========================================================================
% - FSL       
%
%
% Copyright (c) 2013  NeuroPoly, Polytechnique Montreal <www.neuropoly.info>
% Created by: eugénie Ullmann
%
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included in
% all copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
% THE SOFTWARE.
% ========================================================================



% initialization 
clear all
close all

% enter the path of the images for building the template
image={'T1_errsm05'}; %image={'T1_errsm05','T1_errsm09','T1_errsm10'}; %
nb_vertebrae=20; % number of visible vertebrae on the images
output_path_mean='template'; % path for the mean distances vector 
output_path_std='std_template';
output_figure1='fig_template1.pdf';
output_figure2='fig_template2.pdf';


% Don't touch below this line
%---------------------------------------------------------------------
 
%initialization of x and y disk position 
x_disk=zeros(nb_vertebrae,length(image));
y_disk=zeros(nb_vertebrae,length(image));

%initialization distances between disks
distance=zeros(nb_vertebrae-1, length(image));

%loop on each images
for i_img=1:length(image)
    input_anat=image{i_img};
    [anat_ini,dims,scales,bpp,endian] = read_avw(input_anat); % load image
    
    %plot image with an average on 10 slices 
    slice=round(dims(3)/2); 
    anat=anat_ini(:,:,slice-5);
    for i=1:10
        anat=anat+anat_ini(:,:,slice-5+i);
    end
    anat=uint16(anat);
    anat=imadjust(anat);
    
    
    f=figure(i_img);
    scrsz = get(0,'ScreenSize'); % full screen
    set(f,'Position',[1 1 scrsz(3) scrsz(4)]), imagesc(anat), colormap gray, axis image
    
    ylimi=get(gca,'YLim');xlimi=get(gca,'XLim');
    text(xlimi(1),ylimi(1), '\fontsize{20} Click on the centerline of the spine the vertebraal levels, start by C1', 'VerticalAlignment','bottom','HorizontalAlignment','left','Color','k')
    
    % get points
    for j=1:nb_vertebrae
        ptH = impoint;
        pt = getPosition(ptH);
        x_disk(j,i_img) = pt(1); y_disk(j,i_img) = pt(2);
    end
    
    %calculation of distances 
    for j=1:length(x_disk)-1
        distance(j,i_img)=sqrt((x_disk(j+1,i_img)-x_disk(j,i_img))^2+(y_disk(j+1,i_img)-y_disk(j,i_img))^2);
    end
    close (i_img)
    
end
distance=distance';
mean_distance=mean(distance); %mean of distance over all images
std_distance=std(distance); % standard deviation
    
figure(1)
errorbar(mean_distance,std_distance,'xr','linewidth',1.1)
title(' Distribution of distances between intervertebral disks : Mean and standard deviation')
xlabel('vertebral level')
ylabel('Distance mean between 2 adjacent intervertebral disk (mm)')
B=figure(1);
set(1,'Units','Normalized','Outerposition',[0 0 1 1]);
saveas(B,output_figure1)



figure(2)
for fig=1:length(image)
  plot(distance(fig,:),'*','Color',[rand,rand,rand],'linewidth',1.1)  
  hold on
end
xlim([0 nb_vertebrae-1])    
plot(mean_distance,'o','linewidth',1.1)
title('Distribution of distances between intervertebral disks for each image (the circle symbol is the mean)')
xlabel('Intervertebral disk')
ylabel('Distance mean between 2 adjacent intervertebral disk (mm)')
C=figure(2);
set(1,'Units','Normalized','Outerposition',[0 0 1 1]);
saveas(C,output_figure2)



save(output_path_mean,'mean_distance')
save(output_path_std,'std_distance')






