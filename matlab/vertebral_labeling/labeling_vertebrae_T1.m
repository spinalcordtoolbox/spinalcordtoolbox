function [label] = labeling_vertebrae_T1(label)

% Compute a labeled spinal cord centerline 
%
% N.B. THE INPUT SHOULD BE .nii, not .nii.gz!!!
%
% The automatic labeling of vertebrae allows to get a labeling of the level of vertebrae. The process is based on an analysis of the intensity profile along the spine. A template of typical vertebral distance increases in an original way the robustness and the accuracy of disks detection towards low contrast-to-noise ratio, altered/missing disks and susceptibility artifacts.
%
% SYNTAX
% ========================================================================
% label = labeling_vertebrae_T1(label)
%
%
% INPUTS
% ========================================================================
% label
%   input_anat                   string. File name of input spine  T1 MR image. 
%   input_path                   string. path for the input file
%   (output_path)                string. path for output
%   (output_labeled_centerline)  string. File name of outpout labeled centerline of the spinal cord. 
%   (output_labeled_surface)     string. File name of outpout labeled surface of the spinal cord.
%   (surface_do)                 1 or 0 give as output a labeled surface
%                                too
%   (segmentation.do)            compute a segmentation of the spinal cord
%                                (if you don't have input centerline or surface MR image)
%   (segmentation)               structure of parameters for the
%                                segmentation of the spinal cord
%   (input_centerline)           string. File name of input centerline of the
%                                spinal cord
%   (input_surface)              string. File name of input surface of the
%                                spinal cord
%   (shift_AP)                   shift the centerline on the spine in mm default : 17 mm
%   (size_AP)                    mean around the centerline in the anterior-posterior direction in mm
%   (size_RL)                    mean around the centerline in the right-left direction in mm
%   (verbose)                    display figures
%
%
% OUTPUTS
% ========================================================================
% labeled_centerline
%(surface_centerline)
%
% DEPENDENCES
% ========================================================================
% - FSL       
% - SPM
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




close all


%==================================================
% Initialization
%==================================================

fsloutput = ['export FSLOUTPUTTYPE=NIFTI;'];

% check fields
if ~isfield(label,'input_path'), error('no input path given'); end
if ~isfield(label,'input_anat'), error('no input file given'); end
if ~isfield(label,'output_path'), label.output_path='./'; end
if ~isfield(label,'output_labeled_centerline'), label.output_labeled_centerline='centerline_labeled'; end
if ~isfield(label,'surface_do'), label.surface_do=0; end
if ~isfield(label.segmentation,'do'), label.segmentation.do=0; end
if ~isfield(label,'input_centerline'),error('no centerline input given. give a centerline or do a spinal cord segmentation'); end
if ~isfield(label,'input_surface'), label.input_surface=''; end
if ~isfield(label,'shift_AP'), label.shift_AP=17; end
if ~isfield(label,'size_AP'), label.size_AP=6; end
if ~isfield(label,'size_RL'), label.shift_AP=5; end
if ~isfield(label,'verbose'), label.verbose=0; end
if ~isfield(label.segmentation,'interval'), label.segmentation.interval=30; end
if ~isfield(label.segmentation,'nom_radius'), label.segmentation.nom_radius=5; end
if ~isfield(label.segmentation,'tolerance'), label.segmentation.tolerance=0.01; end
if ~isfield(label.segmentation,'ratio_criteria'), label.segmentation.ratio_criteria=0.05; end
if ~isfield(label.segmentation,'num_angles'), label.segmentation.num_angles=64; end
if ~isfield(label.segmentation,'update_multiplier'), label.segmentation.update_multiplier=0.8; end
if ~isfield(label.segmentation,'shear_force_multiplier'), label.segmentation.shear_force_multiplier=0.5; end
if ~isfield(label.segmentation,'max_coeff_horizontal'), label.segmentation.max_coeff_horizontal=10; end
if ~isfield(label.segmentation,'max_coeff_vertical'), label.segmentation.max_coeff_vertical=10; end


input_anat=[label.input_path,label.input_anat];
if label.segmentation.do
    input_centerline=[label.output_path,label.segmentation.centerline]; 
    input_surface=[label.output_path,label.segmentation.surface];
else
    input_centerline=[label.input_path,label.input_centerline];
    input_surface=[label.input_path,label.input_surface];
end


output_centerline_vertebra=[label.output_path,label.output_labeled_centerline];
output_surface_vertebra=[label.output_path,label.output_labeled_surface];
surface_do=label.surface_do;
input_anat_reorient='';
input_centerline_reorient='';
input_surface_reorient='';


%==================================================
% Reorientation of the data if needed
%==================================================

% Read and store the raw image in m_nifti

[status result] = unix(['fslhd ' input_anat]); if status, error(result); end

%Plane view
% Read the orientation of the image
orientation{1} = strtrim(result(findstr(result,'qform_xorient')+13:findstr(result,'qform_yorient')-1));
orientation{2} = strtrim(result(findstr(result,'qform_yorient')+13:findstr(result,'qform_zorient')-1));
orientation{3} = strtrim(result(findstr(result,'qform_zorient')+13:findstr(result,'sform_name')-1));

% Only keeps the first letter of the orient structure
%   where each letter corresponds to :
% R: Right-to-Left         L: Left-to-Right
% P: Posterior-to-Anterior A: Anterior-to-Posterior
% I: Inferior-to-Superior  S: Superior-to-Inferior
orientation{1} = orientation{1}(1);
orientation{2} = orientation{2}(1);
orientation{3} = orientation{3}(1);

% Save initial orientation
orient_init = cell2mat(orientation);

if ~strcmp(orient_init, 'ASR')
    
    % Copy data
    input_anat_reorient=[input_anat,'_reorient'];
    cmd=['cp ',input_anat,'.nii',' ',input_anat_reorient,'.nii'];
    [status result] = unix(cmd); if status, error(result); end
    
    input_centerline_reorient=[input_centerline,'_reorient'];
    cmd=['cp ',input_centerline,'.nii',' ',input_centerline_reorient,'.nii'];
    [status result] = unix(cmd); if status, error(result); end
    
    if surface_do
        input_surface_reorient=[input_surface,'_reorient'];
        cmd=['cp ',input_surface,'.nii',' ',input_surface_reorient,'.nii'];
        [status result] = unix(cmd); if status, error(result); end
    end
    
    % Force radiological orientation
    qform=spm_get_space([input_anat_reorient,'.nii']);
    
    if det(qform)>0
        cmd=['fslorient -forceradiological ',input_anat_reorient];
        [status result] = unix(cmd); if status, error(result); end
        
        cmd=['fslorient -forceradiological ',input_centerline_reorient];
        [status result] = unix(cmd); if status, error(result); end
        
        if surface_do
            cmd=['fslorient -forceradiological ',input_surface_reorient];
            [status result] = unix(cmd); if status, error(result); end
        end
        
    end
    
    % reorient data top get PSL orientation
    cmd=[fsloutput,' fslswapdim ',input_anat_reorient,' AP SI RL ',input_anat_reorient];
    [status result]=unix(cmd); if status, error(result); end
    
    cmd=[fsloutput,' fslswapdim ',input_centerline_reorient,' AP SI RL ',input_centerline_reorient];
    [status result]=unix(cmd); if status, error(result); end
    
    if surface_do
        cmd=[fsloutput,' fslswapdim ',input_surface_reorient,' AP SI RL ',input_surface_reorient];
        [status result]=unix(cmd); if status, error(result); end
    end
    
    
    
    % load images
    [anat_ini,dims,scales,bpp,endian] = read_avw(input_anat_reorient);
    centerline = read_avw(input_centerline_reorient);
    if surface_do
        surface=read_avw(input_surface_reorient);
    end
    
else
    % load images
    [anat_ini,dims,scales,bpp,endian] = read_avw(input_anat);
    centerline = read_avw(input_centerline);
    if surface_do
        surface= read_avw(input_surface);
    end
end



%==================================================
% Calculation of the profile intensity
%==================================================

shift_AP=label.shift_AP*scales(1);% shift the centerline on the spine in mm default : 17 mm
size_AP=label.size_AP*scales(1);% mean around the centerline in the anterior-posterior direction in mm
size_RL=label.size_RL*scales(3);% mean around the centerline in the right-left direction in mm


anat=anat_ini;

anat=uint16(anat);

% find coordinates of the centerline
[x,yz]=find(centerline==1);
z=floor((yz-1)/size(centerline,2))+1;
y=yz-(z-1)*size(centerline,2);

% reorder x,y,z with y in the growing sense
[y,ord] = sort(y);
x=x(ord);
z=z(ord);

% eliminate double in y 
index_double=[];
for i=1:length(y)-1
    if y(i)==y(i+1)
        index_double=[index_double i];
    end
end

y(index_double)=[];
x(index_double)=[];
z(index_double)=[];

% shift the centerline to the spine of shift_AP
x1=round(x-shift_AP/scales(1));

% build intensity profile along the centerline
I=zeros(length(y),1);
for index=1:length(y)
    lim_plus=index+5;
    lim_minus=index-5;
    
    if lim_minus<1, lim_minus=1; end
    if lim_plus>length(x1), lim_plus=length(x1); end
    
    % normal vector of the orthogonal plan to the centerline (=tangent vector to the centerline)
    Vx=x1(lim_plus)-x1(lim_minus);
    Vz=z(lim_plus)-z(lim_minus);
    Vy=y(lim_plus)-y(lim_minus);
    
    % find d in the plan equation : Vx*x +Vy*y+ Vz*z -d =0, the
    % point(x1(index),y(index),z(index)) belongs to the plan
    d=Vx*x1(index)+Vy*y(index)+Vz*z(index);
    
    % average  
    for i_slice_RL=1:2*round(size_RL/scales(3))
        for i_slice_AP=1:2*round(size_AP/scales(1))
            result=(d-Vx*(x1(index)+i_slice_AP-size_AP-1)-Vz*z(index))/Vy;
            if result>size(anat,2), result=size(anat,2); end
            I(index)=I(index)+anat(round(x1(index)+i_slice_AP-size_AP-1),round(result),round(z(index)+i_slice_RL-size_RL-1));
        end
    end
    
end


% detrend intensity, detrending different if centerline longer is smaller
% than 300 mm
start_centerline_y=y(1);
I(I==0)=[];

if label.verbose
    figure(1), plot(I)
    title('Intensity profile along the shifted spinal cord centerline')
    xlabel('direction superior-inferior')
    ylabel('intensity')
end

% intensity filtered, if the centerline is small, the filter is different
if length(I)*scales(2)<300/scales(2)
    I_detrend=j_detrend_new_v2(I',5,'cos',1);
else
    I_detrend=j_detrend_new_v2(I',20,'cos',1);
end

% basic normalisation of the intensity profile
I_detrend=I_detrend';
I_detrend=I_detrend/max(I_detrend);

if label.verbose
figure(2), plot(I_detrend)
xlabel('direction superior-inferior')
ylabel('intensity')
title('Intensity profile along the shifted spinal cord centerline after detrending and basic normalization')
xlim([0 length(I_detrend)])
end

% ask if the first vertebrae is the C1 one, if not, ask to specify it.
info{1}=questdlg('Is the more rostral vertebrae the C1 or C2 one ?','Confirmation','yes','no','yes');
if strcmp(info{1},'no')
    level_start=inputdlg('enter the level of the more rostral vertebra','choice of the more rostral vertebral level of the field of view ');
    level_start=str2double(cell2mat(level_start)); % level of the first vertebrae
elseif strcmp(info{1},'yes')
    level_start=2;
end
%==================================================
% Prepare the pattern and load distances mean
%==================================================

load mean_distance
%mean_distance(1) = 16;
%mean_distance(2) = 12;
%mean_distance(3) = 5;
%mean_distance(4) = 5;
%mean_distance(5) = 5;
C1C2_distance=mean_distance(1:2);
% mean_distance=mean_distance(:,3:end);

mean_distance=mean_distance(:,level_start-1:end);


% pattern
space=linspace(-5/scales(2),5/scales(2),round(11/scales(2)));
pattern=sinc(space*scales(2)/15).^20;
[~,xmax_pattern]=max(pattern); % position of the peak in the pattern
%==================================================
% step 1 : find the first peak
%==================================================


%correlation between the pattern and the intensity profile
[corr_all,lag_all]=xcorr(pattern,I_detrend);

%find the maxima of the correlation
[value,loc_corr]=findpeaks(corr_all,'MINPEAKHEIGHT',0.1);
loc_corr=loc_corr+min(lag_all)-1;


%find the first peak

loc_first_peak=xmax_pattern-loc_corr(find(value>1,1,'last'));
Mcorr1=value(find(value>1,1,'last'));

% build the pattern which be added at each loop in step 2
if xmax_pattern<loc_first_peak
    template_troncated=[zeros(1,loc_first_peak-xmax_pattern) pattern];
else
    template_troncated=pattern(:,xmax_pattern-loc_first_peak:end);
end
xend=find(template_troncated>0.02, 1, 'last' );
pixend=xend-loc_first_peak; % number of pixel after the peaks in the pattern

if label.verbose
    figure(3), plot(template_troncated,'g'),hold on, plot(I_detrend)
    title('Detection of the first peak')
    xlabel('direction anterior-posterior (mm)')
    ylabel('intensity')
    legend('pattern matching with the fisrt peak','intensity profile')
    xlim([0 length(I_detrend)])
end

%normalisation of the peaks to get an unbiased correlation
%found roughly maxima
[value,loc_peak_I]=findpeaks(I_detrend,'MINPEAKDISTANCE',round(10/scales(2))); 
loc_peak_I(value<0.15)=[];
value(value<0.15)=[];


%fitting the roughly maxima found with a smoothing spline
P=fit(loc_peak_I,value,'smoothingspline');
P=feval(P,1:length(I_detrend));

for i=1:length(P)
    if P(i,1)>0.1
        I_detrend(i,1)=I_detrend(i,1)./(P(i,1));
    end
end

if label.verbose
figure(4), plot(loc_peak_I,value)
hold on, plot(I_detrend, 'r')
hold on, plot(P,'g')
title('Setting values of peaks at one by fitting a smoothing spline')
xlim([0 length(I_detrend)])
xlabel('direction superior-inferior (mm)')
ylabel('normalized intensity')
legend('roughly found maxima','intensity profile','smoothing spline')

figure(5), plot(I_detrend)
xlim([0 length(I_detrend)])
xlabel('direction superior-inferior (mm)')
ylabel('normalized intensity')
title('Final intensity profile')
end

%=====================================================================
% step 2 : Cross correlation between the adjusted template and the intensity profile
% local moving of template's peak from the first peak already found
%===========================================================

%for each loop, a peak is added, firstly located at the postion the most
%likely and then local adjustement. the position of the next peak is calculated from previous positions

mean_distance_new=mean_distance; % vector of distance between peaks re evaluted after each loop
mean_ratio=zeros(1,length(mean_distance)); %  mean of ratios between distances of two adjacent loops
L=round(1.2*max(mean_distance))-round(0.8*min(mean_distance));
corr_peak=zeros(L,length(mean_distance)); % initialization of the coeffiecnt of correlation vector

%loop on each peak
for i_peak=1:length(mean_distance)
    
    scale_min=round(0.80*mean_distance_new(i_peak))-xmax_pattern-pixend;% window of +-20% at the left and right of the presumed peak 
    if scale_min<0, scale_min=0; end
    scale_max=round(1.2*mean_distance_new(i_peak))-xmax_pattern-pixend;
    scale_peak=scale_min:scale_max; %number of zeros which be added to move localy the peak. at the middle the peak have the mean position
    
    for i_scale=1:length(scale_peak)
        % build the template with a peak added
        template_resize_peak=[template_troncated zeros(1,scale_peak(i_scale)) pattern];
        
        %cross correlation
        [corr_template_peak,lag_peak]=xcorr(template_resize_peak,I_detrend);
        index_lag_nul=find(lag_peak==0);
        corr_peak(i_scale,i_peak)=corr_template_peak(index_lag_nul);
        
        if label.verbose
            figure(6), plot(I_detrend), hold on, plot(template_resize_peak,'r'), hold off
            xlim([0 length(I_detrend)])
            figure(7), plot(corr_peak(:,i_peak),'*'),hold off
            title('correlation value against the displacement of the peak (px)')
        end
    end
    % find for the current peak the cross-correlation over the window 
    [max_peak,index_scale_peak]=max(corr_peak(:,i_peak));
    good_scale_peak=scale_peak(index_scale_peak);
    Mcorr=max(corr_peak(:,1:i_peak));
    Mcorr=[Mcorr1 Mcorr];
   
    % if the correlation coefficient is two low, put the peak at the mean position
%     if i_peak>2 && max_peak-max(corr_peak(:,i_peak-1))<1.2
    if i_peak>1 && (Mcorr(i_peak+1)-Mcorr(i_peak))<0.4*mean(Mcorr(2:i_peak+1)-Mcorr(1:i_peak))
        test=i_peak;
        template_resize_peak=[template_troncated zeros(1,round(mean_distance(i_peak))-xmax_pattern-pixend) pattern];
        good_scale_peak=round(mean_distance(i_peak))-xmax_pattern-pixend;
    elseif i_peak==1 && (Mcorr(i_peak+1)-Mcorr(1))<0.4*Mcorr(1)
        template_resize_peak=[template_troncated zeros(1,round(mean_distance(i_peak))-xmax_pattern-pixend) pattern];
        good_scale_peak=round(mean_distance(i_peak))-xmax_pattern-pixend;
    else
        template_resize_peak=[template_troncated zeros(1,good_scale_peak) pattern];
    end
    
    %update mean-distance by a adjustement ratio 
    mean_distance_new(i_peak)=good_scale_peak+xmax_pattern+pixend;
    mean_ratio(i_peak)=mean(mean_distance_new(:,1:i_peak)./mean_distance(:,1:i_peak));
    if i_peak<length(mean_distance)
        mean_distance_new(i_peak+1)=mean_distance(i_peak+1)*mean_ratio(i_peak);
    end
    template_troncated=template_resize_peak;
     if label.verbose
        figure(6), plot(I_detrend), hold on, plot(template_troncated,'g'),hold off
        xlim([0 length(I_detrend)])
     end
end

%find maxima of the adjusted template
minpeakvalue=0.50;
[~,loc_disk]=findpeaks(template_troncated,'MINPEAKHEIGHT',minpeakvalue);
loc_disk(loc_disk<0)=[];
loc_disk(loc_disk>size(I_detrend,1))=[];
loc_disk=loc_disk+start_centerline_y-1;


%=====================================================================
% Step 3: Building of the labeled centerline and surface
%=====================================================================

% orthogonal projection of the position of disk centers on the spinal cord center line

for i=1:length(loc_disk)
    
    % find which index of y matches with the disk
    index=find(y==loc_disk(i));
    lim_plus=index+5;
    lim_minus=index-5;
    
    if lim_minus<1, lim_minus=1; end
    if lim_plus>length(x), lim_plus=length(x); end
    
    % normal vector of the orthogonal plan to the centerline (=tangent vector to the centerline)
    Vx=x(lim_plus)-x(lim_minus);
    Vz=z(lim_plus)-z(lim_minus);
    Vy=y(lim_plus)-y(lim_minus);
    
    % find d in the plan equation : Vx*x +Vy*y+ Vz*z +d =0, the
    % point(x1(index),y(index),z(index)) belongs to the plan
    d=Vx*x1(index)+Vy*y(index)+Vz*z(index);
    
    %intersection between the orthogonal plan to the centerline of the spine and the centerline of the spinal cord
    intersection=ones(length(x),1);
    for j=1:length(x)
        intersection(j)=abs(Vx*x(j)+Vy*y(j)+Vz*z(j)-d);
    end
    
    [min_intersection,index_intersection]=min(intersection);
    loc_disk(i)=y(index_intersection);
end

%initialization labelled surface
center_disk=centerline;


% created a new centerline where voxel value matches with the number of the
% vertebra

for i=1:length(loc_disk)-1
    tmp=center_disk(:,loc_disk(i):loc_disk(i+1),:);
    tmp(tmp==1)=i+level_start;    
    center_disk(:,loc_disk(i):loc_disk(i+1),:)=tmp;
end
center_disk(center_disk==1)=0;

%add C1 and C2
if level_start==2
    center_disk(x(1),round(loc_disk(1)-C1C2_distance(2)):loc_disk(1),z(1))=2;
    center_disk(x(1),round(loc_disk(1)-C1C2_distance(1)-C1C2_distance(2)):round(loc_disk(1)-C1C2_distance(2))-1,z(1))=1;
end

% projected centerline on the middle slice to a better visualising with fsl
% if visualising
%     center_disk_projection=sum(centerline,3);
%     center_disk_projection_2=[center_disk_projection(2:end,:);zeros(1,size(center_disk_projection,2))];
%     %center_disk_projection_2=[zeros(size(center_disk_projection,1),1) center_disk_projection(:,1:end-1)];
%     center_disk_projection3=center_disk_projection+center_disk_projection_2;
%     center_disk_projection3(center_disk_projection>0)=1;
%     centerline_mod_fsl=zeros(size(center_disk,1),size(center_disk,2),size(center_disk,3));
%     slice=round(dims(3)/2);
%     centerline_mod_fsl(:,:,slice)=center_disk_projection3;
%     center_disk=centerline_mod_fsl;
%     for i=1:length(loc_disk)-1
%         tmp=center_disk(:,loc_disk(i):loc_disk(i+1),:);
%         tmp(tmp==1)=i+2;
%         center_disk(:,loc_disk(i):loc_disk(i+1),:)=tmp;
%     end
%     center_disk(center_disk==1)=0;
%
%     center_disk(x(1),round(loc_disk(1)-C1C2_distance(2)):loc_disk(1),slice)=2;
%     center_disk(x(1)+1,round(loc_disk(1)-C1C2_distance(2)):loc_disk(1),slice)=2;
%     center_disk(x(1),round(loc_disk(1)-C1C2_distance(1)-C1C2_distance(2)):round(loc_disk(1)-C1C2_distance(2))-1,slice)=1;
%     center_disk(x(1)+1,round(loc_disk(1)-C1C2_distance(1)-C1C2_distance(2)):round(loc_disk(1)-C1C2_distance(2))-1,slice)=1;
% end


% surface labeling
if surface_do
    labelled_surface=surface;
    % created a new surface where voxel value matches with the number of the
    % vertebra
    
    for i=1:length(loc_disk)-1
        tmp=labelled_surface(:,loc_disk(i):loc_disk(i+1),:);      
        tmp(tmp==1)=i+level_start;      
        labelled_surface(:,loc_disk(i):loc_disk(i+1),:)=tmp;
    end
    labelled_surface(labelled_surface==1)=0;
    %add C1 and C2    
    if level_start==2
        for i_V2=1:round(C1C2_distance(2))+1
            labelled_surface(:,loc_disk(1)-i_V2,:)=labelled_surface(:,loc_disk(1),:);
            tmp=labelled_surface(:,loc_disk(1)-i_V2,:);
            tmp(tmp==3)=2;
            labelled_surface(:,loc_disk(1)-i_V2,:)=tmp;
        end
        
        for i_V1=1:round(C1C2_distance(1))+1
            labelled_surface(:,loc_disk(1)-i_V1-round(C1C2_distance(2)),:)=labelled_surface(:,loc_disk(1),:);
            tmp=labelled_surface(:,loc_disk(1)-i_V1-round(C1C2_distance(2)),:);
            tmp(tmp==3)=1;
            labelled_surface(:,loc_disk(1)-i_V1-round(C1C2_distance(2)),:)=tmp;
        end
    end
end

%save modified centerline and oreint the final modified centerline with the
%same orientation than the raw image

save_avw_v2(center_disk,output_centerline_vertebra,'s',scales(1:3));
if surface_do
    save_avw_v2(labelled_surface,output_surface_vertebra,'s',scales(1:3));
end

if ~strcmp(orient_init, 'ASR')
    a = orient_init(1);
    b = orient_init(2);
    c = orient_init(3);
    
    if strcmp(a,'A'), a='AP'; end
    if strcmp(a,'P'), a='PA'; end
    if strcmp(a,'S'), a='SI'; end
    if strcmp(a,'I'), a='IS'; end
    if strcmp(a,'R'), a='RL'; end
    if strcmp(a,'L'), a='LR'; end
    
    if strcmp(b,'A'), b='AP'; end
    if strcmp(b,'P'), b='PA'; end
    if strcmp(b,'S'), b='SI'; end
    if strcmp(b,'I'), b='IS'; end
    if strcmp(b,'R'), b='RL'; end
    if strcmp(b,'L'), b='LR'; end
    
    if strcmp(c,'A'), c='AP'; end
    if strcmp(c,'P'), c='PA'; end
    if strcmp(c,'S'), c='SI'; end
    if strcmp(c,'I'), c='IS'; end
    if strcmp(c,'R'), c='RL'; end
    if strcmp(c,'L'), c='LR'; end
    
    
    
    %copy the header of reorient data in the centerline nifti
    cmd = [fsloutput,' fslcpgeom ', input_anat_reorient, ' ', output_centerline_vertebra, ' -d'];
    [status result] = unix(cmd); if status, error(result); end
    
    if surface_do
        cmd = [fsloutput,' fslcpgeom ', input_anat_reorient, ' ', output_surface_vertebra, ' -d'];
        [status result] = unix(cmd); if status, error(result); end
    end
    
    % change the orientation to match the initial orientation
    if det(qform)>0
        cmd=[fsloutput,'fslswapdim ',output_centerline_vertebra,' -x y z ',output_centerline_vertebra];
        [status result] = unix(cmd); if status, error(result); end
        
        cmd=[fsloutput,' fslorient -swaporient ',output_centerline_vertebra];
        [status result]=unix(cmd); if status, error(result); end
        
        cmd=[fsloutput,' fslswapdim ',output_centerline_vertebra, ' ',a, ' ', b, ' ', c, ' ',output_centerline_vertebra];
        [status result]=unix(cmd); if status, error(result); end
        
        if surface_do
            cmd=[fsloutput,'fslswapdim ',output_surface_vertebra,' -x y z ',output_surface_vertebra];
            [status result] = unix(cmd); if status, error(result); end
            
            cmd=[fsloutput,' fslorient -swaporient ',output_surface_vertebra];
            [status result]=unix(cmd); if status, error(result); end
            
            cmd=[fsloutput,' fslswapdim ',output_surface_vertebra, ' ',a, ' ', b, ' ', c, ' ',output_surface_vertebra];
            [status result]=unix(cmd); if status, error(result); end
        end
        
    else
        
        cmd=[fsloutput,' fslswapdim ',output_centerline_vertebra, ' ',a, ' ', b, ' ', c, ' ',output_centerline_vertebra];
        [status result]=unix(cmd); if status, error(result); end
        
        if surface_do
            cmd=[fsloutput,' fslswapdim ',output_surface_vertebra, ' ',a, ' ', b, ' ', c, ' ',output_surface_vertebra];
            [status result]=unix(cmd); if status, error(result); end
        end
        
    end
    
    %copy the initial header to the centerline data
    cmd = [fsloutput,' fslcpgeom ',input_anat, ' ', output_centerline_vertebra, ' -d'];
    [status result] = unix(cmd); if status, error(result); end
    
    %copy the initial header to the surface data
    
    if surface_do
        cmd = [fsloutput,' fslcpgeom ',input_anat, ' ', output_surface_vertebra, ' -d'];
        [status result] = unix(cmd); if status, error(result); end
    end
    
else
    
    %copy the initial header to the centerline data
    cmd = [fsloutput,' fslcpgeom ',input_anat, ' ', output_centerline_vertebra, ' -d'];
    [status result] = unix(cmd); if status, error(result); end
    
    %copy the initial header to the surface data
    if surface_do
        cmd = [fsloutput,' fslcpgeom ',input_anat, ' ', output_surface_vertebra, ' -d'];
        [status result] = unix(cmd); if status, error(result); end
    end
    
end

% delete file
delete([input_anat_reorient,'.nii']);
delete([input_centerline_reorient,'.nii']);
delete([input_surface_reorient,'.nii']);




