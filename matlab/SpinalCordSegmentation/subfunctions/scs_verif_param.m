function [param] = scs_verif_param(param)
% scs_verif_param
% This function verifies the paramaters in the param struct. If certain
% fields are not specified, the function creates them and assigns a default
% value
%
% SYNTAX:
% [param] = scs_verif_param(param)
% _________________________________________________________________________
% INPUTS:  
%
% PARAM
%   Contains different parameters that are used to fine-tune the
%   segmentation
% _________________________________________________________________________
% OUTPUTS:
%
% PARAM
%   Contains different parameters that are used to fine-tune the
%   segmentation

%%
if isfield(param,'language')==0
    param.language='English';
else
    filename = 'String_definition.xlsx';
    [~,txt,~] = xlsread(filename);  % JULIEN 2013-05-22
    counter = 0;
    for i=2:size(txt,2)
        if strcmp(param.language,txt{1,i}) == 1
            counter = 1;
        end
    end
    if counter == 0;
        scs_warning_error(221,param);
    end
end

if isfield (param,'slices')==0
    param.slices = 10;
else
    if length(param.slices)>1
        if param.slices(1) < 1 || param.slices(2) < 1 || param.slices(1) > param.slices(2)
            scs_warning_error(201, param)
        end
    else
        if param.slices < 1 
            scs_warning_error(222,param);
        end
    end
end

if isfield(param,'interval')==0
    param.interval = 20;
end

if isfield(param,'shear_force_multiplier')==0
    param.shear_force_multiplier = 0.1;
else
    if param.shear_force_multiplier < 0 || param.shear_force_multiplier > 1
        scs_warning_error(223,param);
    end
end
    

if isfield(param,'image_type')==0
    param.image_type = 2;
else
    if param.image_type ~= 1 && param.image_type ~= 2
        scs_warning_error(202, param)
    end
end

if isfield(param,'nom_radius')==0
    param.nom_radius=5;
else
    if param.nom_radius < 1
        scs_warning_error(203, param)
    elseif param.nom_radius > 8
        scs_warning_error(101, param)
    end
end

if isfield(param,'resampling')==0
    param.resampling=1;
else
    if mod(param.resampling,2) == 1 && param.resampling ~=1
        scs_warning_error(102, param)
    end
    if param.resampling < 1
        scs_warning_error(204, param)
    end
end

if isfield(param,'num_angles')==0
    param.num_angles=64;
else
    if param.num_angles < 1
        scs_warning_error(205, param)
    elseif param.num_angles > 360 || param.num_angles < 16
        scs_warning_error(103, param)
    end
end

if isfield(param,'tolerance')==0
    param.tolerance=0.01;
else
    if param.tolerance <=0
        scs_warning_error(206, param)
	elseif param.tolerance > 10
        scs_warning_error(104, param)
    end
end

if isfield(param,'nbIterMax')==0
    param.nbIterMax=10;
else
    if param.nbIterMax <=0
        scs_warning_error(207, param)
	elseif param.nbIterMax > 100 
        scs_warning_error(105, param)
    end
end

if isfield(param,'ratio_criteria')==0
    param.ratio_criteria=0.1;
else
    if param.ratio_criteria < 0
        scs_warning_error(208, param)
	elseif param.ratio_criteria > 0.2 
        scs_warning_error(106, param)
    end    
end

if isfield(param,'update_multiplier')==0
    param.update_multiplier=0.8;
else
    if param.update_multiplier <= 0
        scs_warning_error(209, param)
	elseif param.update_multiplier > 5 || param.update_multiplier < 0.2 
        scs_warning_error(107, param)
    end
end

if isfield(param,'max_coeff_horizontal')==0
    param.max_coeff_horizontal=round(param.num_angles/6);
else
    if param.max_coeff_horizontal <1 || param.max_coeff_horizontal >= param.num_angles/2-2
        scs_warning_error(210, param)
	elseif param.max_coeff_horizontal > param.num_angles/4
        scs_warning_error(108, param)
    end
end

if isfield(param,'max_coeff_vertical')==0
    if length(param.slices)==2
        param.max_coeff_vertical = round((param.slices(2)-param.slices(1))/10);
    end
else
    if param.max_coeff_vertical <1
        scs_warning_error(211, param)
    end
end

if isfield(param,'centerline')==0
    param.centerline=0;  
end

