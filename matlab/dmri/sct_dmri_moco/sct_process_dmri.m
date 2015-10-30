function [sct status] = sct_process_dmri(sct)
% Process dMRI data (reorient, crop, motion correction, etc.).
%
% The algorithm for the motion correction module is here:
% https://docs.google.com/drawings/d/1FoKXYbyFh_q20zsvl_mEcxlUR405gZ4c8DcrUvBxsIM/edit?hl=en_US
%
% N.B. THE INPUT SHOULD BE .nii, not .nii.gz!!!
%
%
% INPUTS
% ========================================================================
% sct
%   file_in             string. File name of input image. 
%   (my_param)          integer.   ---> if argument is optional, put it in brackets
%
%
% OUTPUTS
% ========================================================================
% sct
%
%
% DEPENDENCES
% ========================================================================
% - FSL
%
%
% COMMENTS
% ========================================================================  
% 2011-06-13: includes a module that does the motion correction
% 2011-10-07: Fix bugs for volume-based moco, because the exportation into b0 volume screws up the ordering of the b0
% 2011-10-09: Allows to remove interspersed b=0 images. Flag: sct.dmri.removeInterspersed_b0
% 2011-10-22: Outputs motion correction file for each volume
% 2011-10-23: new dmri field to allow NIFTI_GZ output type
% 2011-11-06: Fix bug related to extension.
% 2011-11-07: Fix the way to find b=0 images when no bvals file is available.
% 2011-11-26: v10. Added Eddy-current correction module (based on reversed gradient polarity)
% 2011-11-26: % To decrease computational time, the output for this process is set to NIFTI, but the final output is set by the user. The field for the final output will be called sct.outputtype
% 2011-11-29: fix deletion of the log file.
% 2011-12-03: v11. Combine moco and eddy-correct modules to only split data once and apply interpolation once. Handle error outputs.
% 2011-12-17: no masking.
% 2012-01-24: Gradient non-linearity distortion correction (wrapper to Jon's code)
% 2013-05-02: Cleaning the code, check bvecs written by lines or columns (TD)
% 2013-05-02: NO MORE MULTI-FOLDER MANAGING!!!! (TD)
% 2013-09-28: transitionning towards the removal of the "sct" parent field --> remove all sct.XX and replace by sct.dmri.XX (JCA)
% 2013-09-28: remove bugs (e.g., non-existence of dmri.file_dwi_mean). Delete dwi_1, dwi_2, etc. at the end of the process.
%
% Copyright (c) 2013 NeuroPoly, Polytechnique Montreal <www.neuropoly.info>
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




% ================================================================================================================================
% INITIALIZATION
% ================================================================================================================================


file_motionCorrection = 'motion_correction.txt';
% fname_mask = [sct.dmri.path,sct.dmri.file_mask];
status = 0; % for error flag

% check fields
if ~isfield(sct,'log'), sct.log = 'sct_process_dmri.txt'; end
if ~isfield(sct.dmri,'outputtype'), sct.outputtype = 'NIFTI'; end
if ~isfield(sct.dmri,'file_bvals'), sct.dmri.file_bvals = ''; end
if ~isfield(sct,'output_path'), sct.output_path='./process_dmri_data/'; end
if ~exist(sct.output_path,'dir')
    mkdir(sct.output_path);
end

% Initialize transformation matrices
mat_folders.names = {};
mat_folders.nb = 0;
suffix_data = '';
suffix_bvecs = '';

% initialize file names
 if ~exist([sct.output_path,'outputs'],'dir'), mkdir([sct.output_path,'outputs']); end
 sct.dmri.file_b0        = [sct.output_path,'outputs/bo'];
 sct.dmri.file_dwi       = [sct.output_path,'outputs/dwi'];
 sct.dmri.file_raw       = sct.dmri.file;

% Find which SHELL is running
j_disp(sct.log,['\nFind which SHELL is running...'])
[status result] = unix('echo $0');
if ~isempty(findstr(result,'bash'))
    sct.dmri.shell = 'bash';
elseif ~isempty(findstr(result,'tsh'))
    sct.dmri.shell = 'tsh';
elseif ~isempty(findstr(result,'tcsh'))
    sct.dmri.shell = 'tsh';
else
    j_disp(sct.log,['.. Failed to identify shell. Using default.'])
    sct.dmri.shell = 'tsh';
end
j_disp(sct.log,['.. Running: ',sct.dmri.shell])
% FSL output
if strcmp(sct.dmri.shell,'bash')
    fsloutput = ['export FSLOUTPUTTYPE=',sct.outputtype,'; ']; % if running BASH
elseif strcmp(sct.dmri.shell,'tsh') || strcmp(sct.dmri.shell,'tcsh')
    fsloutput = ['setenv FSLOUTPUTTYPE ',sct.outputtype,'; ']; % if you're running C-SHELL
else
    error('Check SHELL field.')
end

% extension
if strcmp(sct.outputtype,'NIFTI')
    sct.ext = '.nii';
elseif strcmp(sct.outputtype,'NIFTI_GZ')
    sct.ext = '.nii.gz';
end


% =========================================================================
% START THE SCRIPT
% =========================================================================



% START FUNCTION
j_disp(sct.log,['\n\n\n=========================================================================================================='])
j_disp(sct.log,['   Running: sct_process_dmri'])
j_disp(sct.log,['=========================================================================================================='])
j_disp(sct.log,['.. Started: ',datestr(now)])


% Check parameters
j_disp(sct.log,['\nCheck parameters:'])
j_disp(sct.log,['.. Input data:            ',[sct.dmri.path,sct.dmri.file]])
j_disp(sct.log,['.. bvecs file:            ',[sct.dmri.path,sct.dmri.file_bvecs]])
j_disp(sct.log,['.. Grad. nonlin correc:   ',num2str(sct.dmri.grad_nonlin.do)])
j_disp(sct.log,['.. Eddy correction:       ',num2str(sct.dmri.eddy_correct.do)])
j_disp(sct.log,['.. Reorient data:         ',num2str(sct.dmri.reorient.do)])
j_disp(sct.log,['.. Cropping:              ',sct.dmri.crop.method])
j_disp(sct.log,['.. Motion correction:     ',sct.dmri.moco_intra.method])
j_disp(sct.log,['.. Masking:               ',sct.dmri.mask.method])



% ================================================================================================================================
% CHECK DATA INTEGRITY AND PREPARE DATA
% ================================================================================================================================


% Get data dimension
fprintf('\n');
j_disp(sct.log,'Get dimensions of the data...')
fname_data = [sct.dmri.path,sct.dmri.file];
[~, dims] = read_avw(fname_data);
sct.dmri.nx = dims(1);
sct.dmri.ny = dims(2);
sct.dmri.nz = dims(3);
sct.dmri.nt = dims(4);
j_disp(sct.log,['... data dimension "','": ',num2str(dims(1)),' x ',num2str(dims(2)),' x ',num2str(dims(3)),' x ',num2str(dims(4))])


% Check if input data is NIFTI or NIFTI_GZ
fname_data = [sct.dmri.path,sct.dmri.file];
j_disp(sct.log,['\nCheck extension of input data: [',fname_data,']...'])
if isempty(strfind(sct.dmri.file,'.gz'))
    j_disp(sct.log,['.. NIFTI --> good!'])
    sct.dmri.file = strrep(sct.dmri.file,'.nii','');
else
    j_disp(sct.log,['.. NIFTI_GZ --> convert to NIFTI'])
    cmd = ['fslchfiletype NIFTI ',fname_data];
    j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
    sct.dmri.file = strrep(sct.dmri.file,'.nii.gz','');
end

% get bvecs
j_disp(sct.log,['\nGet bvecs...'])
fname_bvecs = [sct.dmri.path_bvecs,sct.dmri.file_bvecs];
sct.dmri.data_bvecs = load(fname_bvecs);
j_disp(sct.log,['.. File bvecs: ',fname_bvecs])


% get bvals
j_disp(sct.log,['\nGet bvals...'])
if exist(sct.dmri.file_bvals,'file')
	j_disp(sct.log,['.. File bvals: ',sct.dmri.file_bvals])
	sct.dmri.data_bvals = load(sct.dmri.file_bvals);
else
	j_disp(sct.log,['.. !! bvals file is empty. Must be DSI data.'])
end


% Check if bvecs is written by lines
j_disp(sct.log,['\nCheck if bvecs is written by lines or by columns...'])
if size(sct.dmri.data_bvecs,1)==3 && size(sct.dmri.data_bvecs,2)~=3
    j_disp(sct.log,['\nbvecs seems to be written in columns, transposing it...'])
    sct.dmri.data_bvecs=sct.dmri.data_bvecs';
    j_disp(sct.log,['\nwriting new file :',sct.dmri.file_bvecs,'_transposed']);
    fname_bvecs_transposed = [sct.dmri.path_bvecs,sct.dmri.file_bvecs,'_transposed'];
    fid = fopen(fname_bvecs_transposed,'w');
    for i=1:size(sct.dmri.data_bvecs,1)
        fprintf(fid,'%f %f %f\n',sct.dmri.data_bvecs(i,1),sct.dmri.data_bvecs(i,2),sct.dmri.data_bvecs(i,3));
    end
    sct.dmri.file_bvecs=[sct.dmri.file_bvecs,'_transposed'];
else
	j_disp(sct.log,['.. OK! written by lines'])
end


% Check if bvecs matches dimension of the data
j_disp(sct.log,['\nCheck if bvecs matches dimension of the data...'])
j_disp(sct.log,['.. Number of volumes:	 ',num2str(sct.dmri.nt)])
j_disp(sct.log,['.. Number of directions: ',num2str(size(sct.dmri.data_bvecs,1))])
nb_b0_to_add = sct.dmri.nt - size(sct.dmri.data_bvecs,1);
if nb_b0_to_add
    j_disp(sct.log,['!! ERROR: Doesn''t match!! Probably because the scanner added some b=0 images at the beggining. Check your bvecs/bvals and play again.'])
    status = 1;
    return
end







% ================================================================================================================================
%	CROP DATA
% ================================================================================================================================


switch (sct.dmri.crop.method)
    
    case 'manual'
        
        j_disp(sct.log,['\n\n   Crop data'])
        j_disp(sct.log,['-----------------------------------------------'])
        
        j_disp(sct.log,['.. Cropping method: ',sct.dmri.crop.method])
        
        % display stuff
        % split the data into Z dimension
        j_progress('Split the data into Z dimension ...............')
        fname_data = [sct.dmri.path,sct.dmri.file];
        fname_data_splitZ = [sct.output_path,'tmp.dmri.data_splitZ'];
        cmd = [fsloutput,'fslsplit ',fname_data,' ',fname_data_splitZ,' -z'];
        [status result] = unix(cmd); % run UNIX command
        if status, error(result); end % check error
        j_progress(1)
        % split the mask into Z dimension
        j_progress('Split the cropping mask into Z dimension ......')
        fname_mask = [sct.dmri.path,sct.dmri.crop.file_crop];
        fname_mask_splitZ = [sct.output_path,'tmp.dmri.mask_splitZ'];
        cmd = [fsloutput,'fslsplit ',fname_mask,' ',fname_mask_splitZ,' -z'];
        [status result] = unix(cmd); % run UNIX command
        if status, error(result); end % check error
        j_progress(1)
        % Crop each slice individually
        j_progress('Crop each slice individually ..................')
        numZ = j_numbering(sct.dmri.nz,4,0);
        for iZ = 1:sct.dmri.nz
            % load mask
            fname_mask = [fname_mask_splitZ,numZ{iZ}];
            mask = read_avw(fname_mask);
            if length(mask)==1, error('CHECK FILE NAME FOR THE MASK! Exit program.'); end
            % Find the size of the mask
            for i=1:size(mask,3)
                [x y] = find(mask(:,:,i));
                if ~isempty(x) && ~isempty(y)
                    minX = min(x);
                    maxX = max(x);
                    minY = min(y);
                    maxY = max(y);
                    z(i) = i;
                    minZ = min(find(z));
                    maxZ = max(find(z));
                end
            end
            
            % save box coordonates
            save([sct.output_path 'tmp.dmri.crop_box.mat'],'minX','maxX','minY','maxY','minZ','maxZ');
            j_disp(sct.log,['... File created: ','tmp.dmri.crop_box.mat'])
            
            nx_tmp = maxX-minX+1;
            ny_tmp = maxY-minY+1;
            nz_tmp = maxZ-minZ+1;
            % Crop data
            fname_data_splitiZ = [fname_mask_splitZ,numZ{iZ}];
            fname_data_crop_splitiZ = [fname_mask_splitZ,'_crop',numZ{iZ}];
            cmd = [fsloutput,'fslroi ',fname_data_splitiZ,' ',fname_data_crop_splitiZ,' ',...
                num2str(minX),' ',...
                num2str(nx_tmp),' ',...
                num2str(minY),' ',...
                num2str(ny_tmp),' ',...
                '0 1'];
            [status result] = unix(cmd); % run UNIX command
            if status, error(result); end % check error
        end %  iZ
        j_progress(1)
        % Merge data along Z
        j_progress('Merge moco b0 along Z dimension ...............')
        fname_data_crop_splitZ = [fname_mask_splitZ,'_crop*.*'];
        fname_data_crop = [sct.dmri.path,sct.dmri.file,sct.dmri.suffix_crop];
        cmd = [fsloutput,'fslmerge -z ',fname_data_crop,' ',fname_data_crop_splitZ];
        [status result] = unix(cmd);
        if status, error(result); end
        j_progress(1)
        % delete temp files
        delete([sct.output_path,'tmp.dmri.*'])
        

        
    case 'box'
        
        j_disp(sct.log,['\n\n   Crop data'])
        j_disp(sct.log,['-----------------------------------------------'])
        
        j_disp(sct.log,['.. Cropping method: ',sct.dmri.crop.method])
        j_disp(sct.log,['... Crop size: ',sct.dmri.crop.size])
        j_progress('Crop data .....................................')
        
        fname_data = [sct.dmri.path,sct.dmri.file];
        fname_datacrop = [sct.dmri.path,sct.dmri.file,sct.dmri.suffix_crop];
        cmd = [fsloutput,'fslroi ',fname_data,' ',fname_datacrop,' ',sct.dmri.crop.size];
        
        [status result] = unix(cmd); % run UNIX command
        if status, error(result); end % check error
        
        % save box coordonates
        box = strread(sct.dmri.crop.size);
        minX = box(1); maxX = box(2)-box(1); minY = box(3); maxY = box(4)-box(3); minZ = box(5); maxZ = box(6)-box(5);
        save([sct.output_path 'tmp.dmri.crop_box.mat'],'minX','maxX','minY','maxY','minZ','maxZ');
        j_disp(sct.log,['... File created: ','tmp.dmri.crop_box.mat'])
                

    case 'autobox'
        
        j_disp(sct.log,['\n\n   Crop data'])
        j_disp(sct.log,['-----------------------------------------------'])
        
        j_disp(sct.log,['.. Cropping method: ',sct.dmri.crop.method])
        
        if sct.dmri.crop.margin < 3, margin=15; else margin = sct.dmri.crop.margin; end % crop size around centerline
        fname_data = [sct.dmri.path,sct.dmri.file];
        fname_datacrop = [sct.dmri.path,sct.dmri.file,sct.dmri.suffix_crop];
        cmd = [fsloutput,'fslroi ',fname_data,' ',fname_datacrop,' ',num2str(sct.dmri.nx/2-margin),' ',num2str(2*margin),' ',num2str(sct.dmri.ny/2-margin),' ',num2str(2*margin),' 0 -1'];
        j_disp(sct.log,['>> ',cmd]);
        [status result] = unix(cmd); % run UNIX command
        if status, error(result); end % check error
        
        % save box coordonates
        box = strread(sct.dmri.crop.size);
        minX = sct.dmri.nx/2-margin; maxX = sct.dmri.nx/2+margin; minY = sct.dmri.ny/2-margin; maxY = sct.dmri.ny/2 + margin; minZ = 0; maxZ = sct.dmri.nz-1;
        save([sct.output_path 'tmp.dmri.crop_box.mat'],'minX','maxX','minY','maxY','minZ','maxZ');
        j_disp(sct.log,['... File created: ','tmp.dmri.crop_box.mat'])
        
 
        
    case 'centerline'
        j_disp(sct.log,['\n\n   Crop data'])
        j_disp(sct.log,['-----------------------------------------------'])
        j_disp(sct.log,['.. Cropping method: ',sct.dmri.crop.method])
        
        % extract one image (to reslice center_line from anat to dw images)
        cmd = [fsloutput,'fslmaths ',fname_data,' -Tmean ','tmp.dmri.crop.',sct.dmri.file, '_1'];
        j_disp(sct.log,['>> ',cmd]);
        [status result] = unix(cmd);
        if status, error(result); end
        
        file_data_1 = ['tmp.dmri.crop.',sct.dmri.file, '_1'];
        j_disp(sct.log,['... File created: ',file_data_1])
        
        
        % reslice centerline to dmri space
        if exist([sct.output_path,'anat/Spinal_Cord_Segmentation/centerline.nii.gz']) || exist([sct.output_path,'anat/Spinal_Cord_Segmentation/centerline.nii'])
            fname_data = [sct.dmri.path,sct.dmri.file];
            
            % convert centerline.nii.gz to centerline.nii
            if exist([sct.output_path,'anat/Spinal_Cord_Segmentation/centerline.nii.gz'])
                cmd = ['fslchfiletype NIFTI ',sct.output_path,'anat/Spinal_Cord_Segmentation/centerline'];
                j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
            end
            
            % reslice centerline in file_data space
            resflags = struct(...
                'mask',1,... % don't mask anything
                'mean',0,... % write mean image
                'which',1,... % write everything else
                'wrap',[0 0 0]',...
                'interp',1,... % linear interp?
                'output',['tmp.dmri.crop.centerline_resliced.nii']);
            spm_reslice2({[file_data_1,'.nii'];[sct.output_path,'anat/Spinal_Cord_Segmentation/centerline.nii']},resflags);
            
            sct.dmri.centerline_file = 'tmp.dmri.crop.centerline_resliced';
            j_disp(sct.log,['... File created: ',sct.dmri.centerline_file])
        end
        
        
        j_progress('Crop data .....................................')
        fname_data = [sct.dmri.path,sct.dmri.file];
        fname_datacrop = [sct.dmri.path,sct.dmri.file,sct.dmri.suffix_crop];
        
        if isfield(sct.dmri,'centerline_file')
            % read centerline file
            [centerline]=read_avw(sct.dmri.centerline_file);
            [dimX,dimY,dimZ]=size(centerline);
            
            % clean the centerline matrix
            centerline(isnan(centerline)) = 0;
            centerline = boolean(centerline);
            
            % define xmin xmax ymin ymax zmin zmax
            [X,Y] = find(centerline);
            
            if isempty(X)
                param.interval = floor(sct.dmri.nz/3);
                param.img = file_data_1;
                centerline = sct_get_centerline(param);
                
                if sct.dmri.crop.margin < 3, margin=15; else margin = sct.dmri.crop.margin; end % crop size around centerline
                minX = min(centerline(:,1))- margin;
                maxX = max(centerline(:,1))+ margin;
                minY = min(centerline(:,2))- margin;
                maxY = max(centerline(:,2))+ margin;
                minZ = 1;
                maxZ = sct.dmri.nz;
            else
                Z = floor(Y/dimY);
                Y = Y - dimY*Z;
                Z = Z + 1;
                
                if sct.dmri.crop.margin < 3, margin=15; else margin = sct.dmri.crop.margin; end % crop size around centerline
                minX = min(X) - margin; maxX = max(X) + margin;
                minY = min(Y) - margin; maxY = max(Y) + margin;
                minZ = min(Z) - margin; maxZ = max(Z) + margin;
            end
            
            
            
            
        else
            centerline = sct_get_centerline_manual([file_data_1 '.nii']);
            
            if sct.dmri.crop.margin < 3, margin=15; else margin = sct.dmri.crop.margin; end % crop size around centerline
            minX = min(centerline(:,1))- margin;
            maxX = max(centerline(:,1))+ margin;
            minY = min(centerline(:,2))- margin;
            maxY = max(centerline(:,2))+ margin;
            minZ = 0;
            maxZ = sct.dmri.nz;
        end
        
        % prevent a crop bigger than the image data
        if minX<0, minX=0; end, if maxX>sct.dmri.nx, maxX=sct.dmri.nx; end
        if minY<0, minY=0; end, if maxY>sct.dmri.ny, maxY=sct.dmri.ny; end
        if minZ<0, minZ=0; end, if maxZ>sct.dmri.nz, maxZ=sct.dmri.nz; end
        
        % save box coordonates
        save([sct.output_path 'tmp.dmri.crop_box.mat'],'minX','maxX','minY','maxY','minZ','maxZ');
        j_disp(sct.log,['... File created: ','tmp.dmri.crop_box.mat'])
        
        % compute centerline_crop
        centerline(:,1)=centerline(:,1)-minX;
        centerline(:,2)=centerline(:,2)-minY;
        
        % perform cropping with whole spine min/max postions
        cmd = [fsloutput,'fslroi ',fname_data,' ',fname_datacrop,' ',num2str(minX),' ',num2str(maxX-minX),' ',num2str(minY),' ',num2str(maxY-minY),' ',num2str(minZ),' ',num2str(maxZ-minZ)];
        j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); % run UNIX command
        if status, error(result); end % check error
        
        % delete temp files
        j_disp(sct.log,['\nDelete temporary files...'])
        cmd = ['rm -rf tmp.dmri.crop.*'];
        j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
        
    case 'none'
        j_disp(sct.log,['\n\n   Crop data'])
        j_disp(sct.log,['-----------------------------------------------'])
        j_disp(sct.log,['.. No croping'])
        
    otherwise
        j_disp(sct.log,['\n\n   Crop data'])
        j_disp(sct.log,['-----------------------------------------------'])
        error(['croping method ' sct.dmri.crop.method ' isn''t correct'])
end

if ~strcmp(sct.dmri.crop.method,'none')
        % change the default data file name
        suffix_data = [suffix_data, sct.dmri.suffix_crop];
        sct.dmri.file = [sct.dmri.file,sct.dmri.suffix_crop];
        if sct.dmri.crop.apply_moco
            sct.dmri.file_raw = sct.dmri.file; % Apply on croped data
        end
        j_disp(sct.log,['... File created: ',sct.dmri.file])
        
        % get data dimensions
        fprintf('\n');
        j_disp(sct.log,'Get dimensions of the data after cropping...')
        
        fname_data = [sct.dmri.path,sct.dmri.file];
        [~,dims] = read_avw(fname_data);
        sct.dmri.nx = dims(1);
        sct.dmri.ny = dims(2);
        sct.dmri.nz = dims(3);
        sct.dmri.nt = dims(4);
        j_disp(sct.log,['... data dimension "','": ',num2str(dims(1)),'x',num2str(dims(2)),'x',num2str(dims(3)),'x',num2str(dims(4))])
end
save('workspace')






% ================================================================================================================================
%	GRADIENT NON-LINEARITY CORRECTION
% ================================================================================================================================
% Gradient non-linearity distortion correction
if sct.dmri.grad_nonlin.do
    
    j_disp(sct.log,['\n\nGradient non-linearity distortion correction...'])
    j_disp(sct.log,['-----------------------------------------------'])
    
    fname_data = [sct.dmri.path,sct.dmri.file,'.nii'];
    fname_corrected = [sct.dmri.path,sct.dmri.file,'_disco','.nii'];
    mris_gradient_nonlin__unwarp_volume__batchmode(fname_data,fname_corrected,sct.dmri.grad_nonlin.gradient_name,sct.dmri.grad_nonlin.method,sct.dmri.grad_nonlin.polarity,sct.dmri.grad_nonlin.biascor,sct.dmri.grad_nonlin.interp,sct.dmri.grad_nonlin.JacDet);
    
    % change the default data file name
    suffix_data = [suffix_data, '_disco'];
    sct.dmri.file = [sct.dmri.file,'_disco'];
    j_disp(sct.log,['... File created: ',sct.dmri.file])
    
end

save('workspace')















% ================================================================================================================================
%	EDDY-CURRENT CORRECTION
% ================================================================================================================================
% Eddy-current correction using the reversed-gradient polarity method.
if sct.dmri.eddy_correct.do
    
    j_disp(sct.log,['\n\n   Eddy-current correction'])
    j_disp(sct.log,['-----------------------------------------------'])
    
    % run Eddy-current correction module
    fname_data = [sct.dmri.path,sct.dmri.file];
    fname_bvecs = [sct.dmri.path_bvecs,sct.dmri.file_bvecs];
    opt = sct.dmri.eddy_correct;
    opt.outputtype = 'NIFTI';
    opt.fname_log = sct.log;
    opt.split_data = 1;
    opt.find_transfo = 1;
    opt.fname_data_splitT = 'tmp.dmri.data_splitT';
    opt.folder_mat = [sct.output_path,'mat_eddy/'];
    opt.output_path = sct.output_path;
    opt.apply_transfo = 1;
    opt.merge_back = 1;
    opt.compute_rmse = 0;
    j_dmri_eddyCorrect(fname_data, fname_bvecs, opt);
    
    % change the default data file name
    sct.dmri.file = [sct.dmri.file,sct.dmri.eddy_correct.outputsuffix];
    j_disp(sct.log,['... File created: ',sct.dmri.file])
    
    %Note transformation matrix folder for subsequent correction
    mat_folders.nb = mat_folders.nb + 1;
    mat_folders.names{mat_folders.nb} = [sct.output_path 'mat_eddy/'];
    mat_folders.slicewise(mat_folders.nb) = 1;
    
    
end

save('workspace')









% ================================================================================================================================
%	REORIENT DATA
% ================================================================================================================================
% reorient data
if sct.dmri.reorient.do
    
    j_disp(sct.log,['\n\n   Reorient data'])
    j_disp(sct.log,['-----------------------------------------------'])
    
    j_disp(sct.log,'Re-orient data according to MNI template...')
    % build file names
    fname_data_raw = [sct.dmri.path,sct.dmri.file];
    fname_data_reorient = [sct.dmri.path,sct.dmri.file,sct.dmri.suffix_orient];
    % re-orient data
    cmd = [fsloutput,'fslswapdim ',fname_data_raw,' ',sct.dmri.reorient,' ',fname_data_reorient];
    j_disp(sct.log,['>> ',cmd]);
    [status result] = unix(cmd);
    if status, error(result); end
    
    % change the default data file name
    suffix_data = [suffix_data, sct.dmri.suffix_orient];
    sct.dmri.file = [sct.dmri.file,sct.dmri.suffix_orient];
    j_disp(sct.log,['... File created: ',sct.dmri.file])
    
end










% ================================================================================================================================
%	CHECK ORIENTATION
% ================================================================================================================================

% get data dimensions and orientation
j_disp(sct.log,'\nGet dimensions of the data...')
fname_data = [sct.dmri.path,sct.dmri.file];
[~,dims] = read_avw(fname_data);
sct.dmri.nx = dims(1);
sct.dmri.ny = dims(2);
sct.dmri.nz = dims(3);
sct.dmri.nt = dims(4);
j_disp(sct.log,['... data dimension "','": ',num2str(dims(1)),' x ',num2str(dims(2)),' x ',num2str(dims(3)),' x ',num2str(dims(4))])
cmd = ['fslhd ',sct.dmri.path,sct.dmri.file];
[status result] = unix(cmd); if status, error(result); end
[properties values]=strread(result,'%s %s');
properties = [properties values];


% Check X dimension orientation
Xind = find(~cellfun(@isempty,strfind(properties,'xorient')));
switch (values{Xind(1)})
    case 'Left-to-Right'
        sct.dmri.X_orientation = 'LR';
        
    case 'Right-to-Left'
        sct.dmri.X_orientation = 'RL';
        
    case 'Posterior-to-Anterior'
        sct.dmri.X_orientation = 'PA';
        
    case 'Anterior-to-Posterior'
        sct.dmri.X_orientation = 'AP';
        
    case 'Inferior-to-Superior'
        sct.dmri.X_orientation = 'IS';
        
    case 'Superior-to-Inferior'
        sct.dmri.X_orientation = 'SI';
end
clear('Xind')

% Check Y dimension orientation
Yind = find(~cellfun(@isempty,strfind(properties,'yorient')));
switch (values{Yind(1)})
    case 'Left-to-Right'
        sct.dmri.Y_orientation = 'LR';
        
    case 'Right-to-Left'
        sct.dmri.Y_orientation = 'RL';
        
    case 'Posterior-to-Anterior'
        sct.dmri.Y_orientation = 'PA';
        
    case 'Anterior-to-Posterior'
        sct.dmri.Y_orientation = 'AP';
        
    case 'Inferior-to-Superior'
        sct.dmri.Y_orientation = 'IS';
        
    case 'Superior-to-Inferior'
        sct.dmri.Y_orientation = 'SI';
end
clear('Yind')


% Check Z dimension orientation
Zind = find(~cellfun(@isempty,strfind(properties,'zorient')));
switch (values{Zind(1)})
    case 'Left-to-Right'
        sct.dmri.Z_orientation = 'LR';
        
    case 'Right-to-Left'
        sct.dmri.Z_orientation = 'RL';
        
    case 'Posterior-to-Anterior'
        sct.dmri.Z_orientation = 'PA';
        
    case 'Anterior-to-Posterior'
        sct.dmri.Z_orientation = 'AP';
        
    case 'Inferior-to-Superior'
        sct.dmri.Z_orientation = 'IS';
        
    case 'Superior-to-Inferior'
        sct.dmri.Z_orientation = 'SI';
end
clear('Zind')

j_disp(sct.log,['X orientation is: ',sct.dmri.X_orientation]);
j_disp(sct.log,['Y orientation is: ',sct.dmri.Y_orientation]);
j_disp(sct.log,['Z orientation is: ',sct.dmri.Z_orientation]);


save('workspace')














% ================================================================================================================================
%	EPI DISTORTION CORRECTION
% ================================================================================================================================
% Gradient non-linearity distortion correction
if sct.disco.do
    fprintf(['\n\n\n=========================================================================================================='])
    fprintf(['\n   Running: distortion correction'])
    fprintf(['\n=========================================================================================================='])
    fprintf(['\n.. Started: ',datestr(now),'\n'])
    
    % crop and mean epi image
    fname_plus = [sct.output_path,'disco/', sct.disco.file];
    fname_plus_croped = [sct.output_path,'disco/', sct.disco.file, sct.dmri.suffix_crop];
    % perform cropping with whole spine min/max postions
    if exist([sct.output_path 'tmp.dmri.crop_box.mat'])
        load([sct.output_path 'tmp.dmri.crop_box.mat']);
        cmd = [fsloutput,'fslroi ',fname_plus,' ',fname_plus_croped,' ',num2str(minX),' ',num2str(maxX-minX),' ',num2str(minY),' ',num2str(maxY-minY),' ',num2str(minZ),' ',num2str(maxZ-minZ)];
        j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); % run UNIX command
        if status, error(result); end % check error
        fname_plus = fname_plus_croped;
    end
    % average epi images
    j_disp(sct.log,['\nAverage epi images...'])
    fname_plus_mean = [fname_plus,'_mean.nii'];
    cmd = [fsloutput,'fslmaths ',fname_plus,' -Tmean ',fname_plus_mean];
    j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
    j_disp(sct.log,['.. File created: ',fname_plus_mean])
    
    % disco - file
    sct.disco.fname_plus			= fname_plus_mean;
    sct.disco.fname_minus			= [sct.dmri.path,sct.dmri.file_b0_mean];
    sct.disco.fname_data			= [sct.dmri.path, sct.dmri.file]; % DWI data here. If empty, only corrects 'epi_plus'.
    
    % disco - todo
    sct.disco.slice_numb			= 0; % correct for one specific slice only and output detailed disco results (put 0 to correct all slices)
    sct.disco.nbVolToDo				= 0; % number of volumes to correct. Put 0 for all volumes in data folder.
    sct.disco.permute_data			= [1 2 3]; % permute data so that Y is the phase-encoding direction. For no permutation, put [1 2 3]
    
    % disco - mask
    % If you have high intensity structure on the edge of either image (plus or minus), the estimation of the deformation field will
    % be wrong and will yield improper distortion correction. In that case, it is suggested that you create a mask and enter the file
    % name of the mask in these fields.
    
    sct.disco.fname_mask_plus		= ''; % Default = ''
    sct.disco.fname_mask_minus		= ''; % Default = ''
    sct.disco.dilate_mask			= 0; % put size of dilation. Put 0 for no dilation.
    
    % disco - visu
    sct.disco.flip_data				= 0; % flip up/down. Put 0 for faster processing.
    sct.disco.c_lim					= [0 1000]; % color dynamic. Put [] for automatic scaling
    sct.disco.save_deformation      = 1; % Save deformation field and Jacobian matrix as nifti volume. Defualt = 1.
    
    % run disco
    sct.disco = j_distortion_correction_reversedGradient_v5(sct.disco);
    
    % convert nii.gz to nii
    cmd = ['fslchfiletype NIFTI ',sct.dmri.path,sct.dmri.file,sct.disco.suffixe_output];
    j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
    
    % change the default data file name
    suffix_data = [suffix_data, sct.disco.suffixe_output];
    sct.dmri.file = [sct.dmri.file,sct.disco.suffixe_output];
    j_disp(sct.log,['... File created: ',sct.dmri.file])
end


save('workspace')









% =========================================================================
%	UPSAMPLE
% =========================================================================

if sct.dmri.upsample.do
    j_disp(sct.log,['Interpolate data by a factor of 2 ...'])
    %     fid = fopen('tmp_interp_matrix','w');
    %     fprintf(fid, '%i %i %i %i\n %i %i %i %i\n %i %i %i %i\n %i %i %i %i',0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 1);
    %     fclose(fid);
    %     cmd = [fsloutput 'flirt -in ' sct.dmri.path sct.dmri.file ' -ref ' sct.dmri.path sct.dmri.file ' -applyxfm -init tmp_interp_matrix -out ' ct.dmri.path sct.dmri.file '_interp'];
    %     j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
    
    %     cmd = [fsloutput 'fslsplit ' sct.dmri.path sct.dmri.file ' tmp_' sct.dmri.file];
    %     j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
    %
    %     numT = j_numbering(sct.dmri.nt,4,0);
    %     for iT = 1:sct.dmri.nt
    %         cmd = ['c3d tmp_' sct.dmri.file numT{iT} '.nii -interpolation Cubic -resample ' num2str(2*sct.dmri.nx) 'x' num2str(2*sct.dmri.ny) 'x' num2str(sct.dmri.nz) 'vox -o tmp_' sct.dmri.file '_interp' numT{iT} '.nii'];
    %         j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
    %     end
    %     cmd = [fsloutput 'fslmerge -t ' sct.dmri.path sct.dmri.file '_interp tmp_' sct.dmri.file '_interp*'];
    %     j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
    %  unix('rm -f tmp_*');
    
    [dmri_matrix,dims,scales] = read_avw([sct.dmri.path sct.dmri.file]);
    dmri_matrix_interp = zeros(2*sct.dmri.nx-1,2*sct.dmri.ny-1,sct.dmri.nz,sct.dmri.nt);
    for iT = 1:sct.dmri.nt
        for iZ = 1:sct.dmri.nz
            dmri_matrix_interp(:,:,iZ,iT) =  interp2(dmri_matrix(:,:,iZ,iT),1,'linear');
        end
    end
    scales(1:2)=scales(1:2)/2;
    save_avw_v2(dmri_matrix_interp,[sct.dmri.path sct.dmri.file '_interp'],'f',scales);
    
    % clear vars
    clear('dmri_matrix_interp','dmri_matrix','dims');
    % change the default data file name
    sct.dmri.file = [sct.dmri.file '_interp'];
    j_disp(sct.log,['... File created: ',sct.dmri.file])
end










% =========================================================================
%	PREPARE DATA FOR MOCO (ON CROPED DATA)
% =========================================================================
% Split data here.

j_disp(sct.log,['\n\n   Prepare data'])
j_disp(sct.log,['-----------------------------------------------'])

% find where are the b=0 images
j_disp(sct.log,['Identify b=0 images...'])
switch sct.dmri.moco_intra.method
    case 'dwi_lowbvalue'
        j_disp(sct.log,['Using low diffusion instead...'])
        if exist([sct.dmri.schemefile])
            fid = fopen(sct.dmri.schemefile,'r');
            fgetl(fid);fgetl(fid);fgetl(fid); % skip first 3 lines
            scheme = fscanf(fid,'%f %f %f %f %f %f %f',[7,Inf]); scheme = scheme';
            % All values must be in SI units
            DELTA = scheme(:,5);
            delta = scheme(:,6);
            bvecs = scheme(:,1:3);
            grad_val = scheme(:,4).*sqrt(sum(bvecs.^2,2));
            gamma = 42.57*10^6; % Hz/T
            bvals = (2*pi*gamma*delta.*grad_val).^2.*(DELTA - delta/3)*10^-6;% s/mm^2
        elseif exist(sct.dmri.file_bvals,'file')
            bvals = load(sct.dmri.file_bvals);
        else
            
            [sct.dmri.file_bvals,sct.dmri.path_bvals] = uigetfile('*','Select bvals file') ;
            sct.dmri.file_bvals = [sct.dmri.path_bvals,sct.dmri.file_bvals];
            bvals = load(sct.dmri.file_bvals);
        end
        index_b0 = find(bvals>429 & bvals<4000)';
        if isempty(index_b0), error('\nNo low diffusion images... change motion correction method'); end
        % process 'dwi_lowbvalue' method as if it were a b0 method (b0 index is just not the same)
        sct.dmri.moco_intra.method = 'b0';
        
    otherwise 'b0'
        index_b0 = find(sum(sct.dmri.data_bvecs.^2,2)<0.005)';
end


sct.dmri.index_b0 = index_b0; index_b0=index_b0(:);
j_disp(sct.log,['.. Index of b=0 images: ',num2str(index_b0')])
nb_b0 = length(index_b0);
sct.dmri.nb_b0 = nb_b0;

% split into T dimension
j_disp(sct.log,['\nSplit along T dimension...'])
fname_data = [sct.dmri.path,sct.dmri.file];
cmd = [fsloutput,'fslsplit ',fname_data,' ',sct.output_path,'tmp.dmri.data_splitT'];
j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
numT = j_numbering(sct.dmri.nt,4,0);

% Merge b=0 images
j_disp(sct.log,['\nMerge b=0 images...'])
fname_b0_merge = [sct.dmri.file_b0,suffix_data];
cmd = [fsloutput,'fslmerge -t ',fname_b0_merge];
for iT = 1:nb_b0
    cmd = cat(2,cmd,[' ',sct.output_path,'tmp.dmri.data_splitT',numT{index_b0(iT)}]);
end
j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
j_disp(sct.log,['.. File created: ',fname_b0_merge])

% change the default b0 file name
sct.dmri.file_b0=[sct.dmri.file_b0,suffix_data];

% Average b=0 images
j_disp(sct.log,['\nAverage b=0 images...'])
fname_b0_mean = [sct.dmri.file_b0,'_mean'];
cmd = [fsloutput,'fslmaths ',fname_b0_merge,' -Tmean ',fname_b0_mean];
j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
j_disp(sct.log,['.. File created: ',fname_b0_mean])



j_disp(sct.log,['\nMerge and average groups of DW images into one file'])
% Make DWI groups (Nimages) index (without b0)

% find where are the dwi
index_dwi=1:sct.dmri.nt; %initialize with all images
index_dwi=index_dwi(~ismember(index_dwi,index_b0)); % remove b0
sct.dmri.nb_dwi=sct.dmri.nt-sct.dmri.nb_b0;
% Number of dwi groups
sct.dmri.nb_groups = floor(sct.dmri.nb_dwi/sct.dmri.moco_intra.dwi_group_size);
% Generate groups indexes
for iGroup = 0:(sct.dmri.nb_groups-1)
	sct.dmri.group_indexes{iGroup+1} = index_dwi((iGroup*sct.dmri.moco_intra.dwi_group_size+1):((iGroup+1)*sct.dmri.moco_intra.dwi_group_size));
end
% add the remaining images to the last DWI group
nb_remaining=mod(sct.dmri.nb_dwi,sct.dmri.moco_intra.dwi_group_size); % number of remaining images
if nb_remaining<3
sct.dmri.group_indexes{sct.dmri.nb_groups}=[sct.dmri.group_indexes{sct.dmri.nb_groups}, index_dwi(end-nb_remaining+1:end)];
else
    sct.dmri.nb_groups=sct.dmri.nb_groups+1;
    sct.dmri.group_indexes{sct.dmri.nb_groups}=index_dwi(end-nb_remaining+1:end);
end
sct.dmri.group_indexes = sct.dmri.group_indexes(~cellfun(@isempty, sct.dmri.group_indexes));% Remove empty groups index

% Size of dwi groups
for iGroup = 1:sct.dmri.nb_groups
	j_disp(sct.log,['\nGroup ',num2str(iGroup),' of DW images'])

	j_disp(sct.log,['.. Index of DW images: ',num2str(sct.dmri.group_indexes{iGroup})])
	index_dwi_i = sct.dmri.group_indexes{iGroup};
    nb_dwi_i = length(index_dwi_i);
    
	% Merge DWI images
	j_disp(sct.log,['\nMerge DW images...'])
	fname_dwi_merge_i = [sct.dmri.file_dwi,suffix_data,'_',num2str(iGroup)];
	cmd = [fsloutput,'fslmerge -t ',fname_dwi_merge_i];
	for iT = 1:nb_dwi_i
		cmd = cat(2,cmd,[' ',sct.output_path,'tmp.dmri.data_splitT',numT{index_dwi_i(iT)}]);
	end
	j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
	j_disp(sct.log,['.. File created: ',fname_dwi_merge_i])
        

	% Average DWI images
	j_disp(sct.log,['\nAverage DWI images...'])
	fname_dwi_mean = [sct.dmri.file_dwi,suffix_data,'_mean','_',num2str(iGroup)];
	cmd = [fsloutput,'fslmaths ',fname_dwi_merge_i,' -Tmean ',fname_dwi_mean];
	j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
	j_disp(sct.log,['.. File created: ',fname_dwi_mean])
end % DW images group
% Update dwi file name
sct.dmri.file_dwi=[sct.dmri.file_dwi,suffix_data];

j_disp(sct.log,['\nMerge and average all DW images into one file'])


% Merge DWI groups means
j_disp(sct.log,['\nMerging DW files...'])
fname_dwi_groups_means_merge = [sct.dmri.path,'dwi_groups_mean'];
cmd = [fsloutput,'fslmerge -t ',fname_dwi_groups_means_merge];
for iGroup = 1:sct.dmri.nb_groups
    cmd = cat(2,cmd,[' ',sct.dmri.file_dwi,'_mean_',num2str(iGroup)]);
end
j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
j_disp(sct.log,['.. File created: ',fname_dwi_groups_means_merge])

% Average DWI images
j_disp(sct.log,['\nAveraging all DW images...'])
fname_dwi_mean = [sct.dmri.path,'dwi_mean'];
cmd = [fsloutput,'fslmaths ',fname_dwi_groups_means_merge,' -Tmean ',fname_dwi_mean];
j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
j_disp(sct.log,['.. File created: ',fname_dwi_mean])


save([sct.output_path 'workspace.mat'])










% ====================================================================
%	INTRA-RUN MOTION CORRECTION
% ====================================================================

if ~strcmp(sct.dmri.moco_intra.method,'none')
    
    j_disp(sct.log,['\n\n   Intra-run motion correction:'])
    j_disp(sct.log,['-----------------------------------------------'])
    
    % use target image from these data
    
    % identify target file
    j_disp(sct.log,['.. Motion correction method: "',sct.dmri.moco_intra.method,'"'])
    switch sct.dmri.moco_intra.method
        
        case 'b0'
            % use b=0 image. In case b=0 images are interspersed,
            % use the b=0 closest to the DWI for registration
            if strcmp(sct.dmri.moco_intra.ref,'b0_mean')
                fname_target = [sct.dmri.file_b0,'_mean'];
            else
                if ~str2num(sct.dmri.moco_intra.ref)
                    fname_data = [sct.dmri.path,sct.dmri.file];
                    data=load_nii_data([fname_data sct.ext]);
                    data=data(:,:,:,index_b0);
                    figure(14)
                    imagesc(data(:,:,ceil(end/2),1)); colormap gray; axis image;
                    msgbox({'Use the slider (figure 14, bottom) to select the ref (highest contrast, no CSF)' 'Press any key when are done..'})
                    hsl = uicontrol('Style','slider','Min',1,'Max',sct.dmri.nt,...
                        'SliderStep',[1 1]./sct.dmri.nt,'Value',1,...
                        'Position',[20 20 200 20]);
                    set(hsl,'Callback',@(hObject,eventdata)  display_midleslice(hObject,data))
                    pause
                    
                    num_b0=round(get(hsl,'Value'));
                    close(14)
                else
                    num_b0 = str2num(sct.dmri.moco_intra.ref);
                end
                % Extract corresponding b=0 image
                j_disp(sct.log,['.. Extract b=0 images #',num2str(num_b0),' to use for registration target'])

                fname_b0 = [sct.dmri.path,'ref_moco.b0',num2str(num_b0)];
                cmd = [fsloutput,'fslroi ',fname_data,' ',fname_b0,' ',num2str(index_b0(num_b0)-1),' 1'];
                j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
                fname_target = ['ref_moco.b0',num2str(num_b0)];
                j_disp(sct.log,['.. Target image: "',fname_target,'"'])
            end
            
        case 'dwi'
            % use the group's mean image for registration
            fname_target = cell(1,sct.dmri.nb_groups);
            for j = 1:sct.dmri.nb_groups
                j_disp(sct.log,['\nCreate registration target for dwi group number',num2str(j)])
                fname_target{j} = [sct.dmri.file_dwi,'_mean_',num2str(j)];
                j_disp(sct.log,['.. Target image: "',fname_target{j},'"'])
            end
    end % end switch
    
    
    
    % Estimate motion correction
    
    if strcmp(sct.dmri.moco_intra.method,'b0')
        
        % Estimate motion based on b=0 images
        j_disp(sct.log,['\nEstimate motion based on b=0 images...'])
        param = sct.dmri.moco_intra;
        param.todo = 'estimate'; % only estimate matrix (to minimize interpolation)
        param.split_data = 0;
        param.folder_mat = [sct.output_path,'mat_moco/'];
        param.output_path= [sct.output_path,'tmp_moco/'];
        param.fname_data = [sct.dmri.path,sct.dmri.file];
        param.fname_data_moco = [sct.dmri.file_b0,sct.dmri.suffix_moco];
        param.fname_target = [sct.dmri.path,fname_target];
        param.fname_data_splitT = [sct.output_path 'tmp.dmri.data_splitT'];
        param.nt = sct.dmri.nt;
        param.nz = sct.dmri.nz;
        param.fsloutput = fsloutput;
        param.index = sct.dmri.index_b0; % correct b0 only
        param.fname_log = sct.log;
        param.suffix = sct.dmri.suffix_moco;
        if any(strcmp(who,'centerline')), param.centerline=centerline; end
        
        j_mri_moco_v8(param);
        save([sct.output_path 'workspace.mat'])
        
        
        % Find registration matrix to the closest DWI data
        j_disp(sct.log,['\nFind registration matrix to the closest DWI data...'])
        for i_file = [1 sct.dmri.nt]
            % find which mat file to use
            [min_closest_b0 i_b0] = min(abs(index_b0'-i_file));
            % copy mat file
            if sct.dmri.moco_intra.slicewise
                for iZ=1:sct.dmri.nz
                    fname_mat_tmp = [sct.output_path 'mat_moco/','mat.T',num2str(index_b0(i_b0)),'_Z',num2str(iZ),'.txt'];
                    fname_mat{i_file} = [sct.output_path 'mat_moco/','mat.T',num2str(i_file),'_Z',num2str(iZ),'.txt'];
                    if i_file ~= index_b0(i_b0)
                        copyfile(fname_mat_tmp,fname_mat{i_file});
                        % display which mat to use
                        j_disp(sct.log,['.. Correct volume #',num2str(i_file),' using "',fname_mat_tmp,'" (distance to b=0: ',num2str(min_closest_b0),')']);
                    end
                end
            else
                fname_mat_tmp = [sct.output_path 'mat_moco/','mat.T',numT{index_b0(i_b0)},'.txt'];
                fname_mat{i_file} = [sct.output_path 'mat_moco/','mat.T',numT{i_file},'.txt'];
                if i_file ~= index_b0(i_b0)
                    copyfile(fname_mat_tmp,fname_mat{i_file});
                    % display which mat to use
                    j_disp(sct.log,['.. Correct volume #',num2str(i_file),' using "',fname_mat_tmp,'" (distance to b=0: ',num2str(min_closest_b0),')']);
                end
            end
        end
        
        % Note transformation matrix folder for subsequent correction
        mat_folders.nb = mat_folders.nb + 1;
        mat_folders.names{mat_folders.nb} = [sct.output_path 'mat_moco/'];
        mat_folders.slicewise(mat_folders.nb) = sct.dmri.moco_intra.slicewise;
        
    elseif strcmp(sct.dmri.moco_intra.method,'dwi')
        
%         % call motion correction module for each dwi group
%         j_disp(sct.log,['\n\n\n\n\n==============================================='])
%         j_disp(sct.log,['    STEP 1/3 : Estimate motion correction for each group dwi based on dwi_iGroup_mean'])
%         j_disp(sct.log,['===============================================\n'])
%         for  iGroup = 1:sct.dmri.nb_groups
%             j_disp(sct.log,['\n    Estimate motion correction of dwi group ',num2str( iGroup),' / ',num2str(sct.dmri.nb_groups)])
%             j_disp(sct.log,['-----------------------------------------------\n'])
%             param = sct.dmri.moco_intra;
%             param.todo = 'estimate_and_apply'; % only estimate matrix (to minimize interpolation)
%             param.split_data = 0;
%             param.fname_data_splitT = [sct.output_path 'tmp.dmri.data_splitT'];
%             param.folder_mat = [sct.output_path 'mat_moco/'];
%             param.output_path = [sct.output_path 'tmp_moco/'];
%             param.fname_data = [sct.dmri.path,sct.dmri.file];
%             param.fname_data_moco = [sct.dmri.file_dwi,sct.dmri.suffix_moco,'_',num2str( iGroup)];
%             param.fname_target = [sct.dmri.path,fname_target{ iGroup}];
%             param.nt = sct.dmri.nt;
%             param.nz = sct.dmri.nz;
%             param.fsloutput = fsloutput;
%             param.index = sct.dmri.group_indexes{iGroup}; % correct volumes of group number  iGroup
%             param.fname_log = sct.log;
%             param.suffix = sct.dmri.suffix_moco;
%             
%             j_mri_moco_v8(param);
%             save([sct.output_path 'workspace.mat'])
%             delete(['tmp.dmri.data_croped_splitT',sct.dmri.suffix_moco,'*']);
%             
%             
%             % Average DWI images
%             j_disp(sct.log,['\nAverage DWI images...'])
%             fname_dwi_merge_i = [sct.dmri.file_dwi,sct.dmri.suffix_moco,'_',num2str( iGroup)];
%             fname_dwi_mean_i    = [sct.dmri.file_dwi,sct.dmri.suffix_moco,'_mean','_',numT{iGroup}];
%             cmd = [fsloutput,'fslmaths ',fname_dwi_merge_i,' -Tmean ',fname_dwi_mean_i];
%             j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
%             j_disp(sct.log,['.. File created: ',sct.dmri.file_dwi,sct.dmri.suffix_moco,'_mean','_',numT{iGroup}]);
%             
%         end
%         
%         
%         
%         % Merge all DWI_mean_group images
%         j_disp(sct.log,['\nMerging all DW files...'])
%         fname_dwi_merge = [sct.dmri.file_dwi,sct.dmri.suffix_moco,'_mean','_Groups'];
%         cmd = [fsloutput,'fslmerge -t ',fname_dwi_merge];
%         for iGroup = 1:sct.dmri.nb_groups
%             cmd = cat(2,cmd,[' ',sct.dmri.file_dwi,sct.dmri.suffix_moco,'_mean','_',numT{iGroup}]);
%         end
%         j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
%         j_disp(sct.log,['.. File created: ',sct.dmri.file_dwi,sct.dmri.suffix_moco,'_mean','_Groups'])
%         
%         % Average all DWI images
%         j_disp(sct.log,['\nAverage all DWI images...'])
%         fname_dwi_mean    = [sct.dmri.file_dwi,sct.dmri.suffix_moco,'_mean'];
%         cmd = [fsloutput,'fslmaths ',fname_dwi_merge,' -Tmean ',fname_dwi_mean];
%         j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
%         j_disp(sct.log,['.. File created: ',fname_dwi_mean])
        
        % call motion corection module to registrate dwi_mean_iGroup on dwi_mean total
        j_disp(sct.log,['\n\n\n\n\n====================================================================='])
        j_disp(sct.log,['    STEP 1/2 : Estimate motion correction of dwi_mean_iGroup on dwi_mean_total '])
        j_disp(sct.log,['=====================================================================\n'])
        param = sct.dmri.moco_intra;
        param.todo = 'estimate'; % only estimate matrix (to minimize interpolation)
        param.split_data = 1;
        param.folder_mat = [sct.output_path, 'tmp.dmri.group_mean.mat/'];
        param.output_path = [sct.output_path, 'tmp_moco/'];
        param.fname_data = fname_dwi_groups_means_merge;
        param.fname_target = fname_dwi_mean;
        param.fsloutput = fsloutput;
        param.fname_log = sct.log;
        param.suffix = sct.dmri.suffix_moco;
        
        j_mri_moco_v8(param);
        save([sct.output_path 'workspace.mat'])
        
        
        
        % Copy registration matrix for every dwi based on dwi_groups_mean
        if ~exist([sct.output_path 'mat_moco/']), mkdir([sct.output_path 'mat_moco/']), end
        for iGroup = 1:sct.dmri.nb_groups
            for dwi = 1:length(sct.dmri.group_indexes{iGroup})
                if sct.dmri.moco_intra.slicewise
                    for i_Z = 1:sct.dmri.nz
                        cmd = ['cp ',sct.output_path,'tmp.dmri.group_mean.mat/','mat.T',num2str(iGroup),'_Z', num2str(i_Z),'.txt', ' ',sct.output_path,'mat_moco/','mat.T',num2str(sct.dmri.group_indexes{iGroup}(dwi)), '_Z', num2str(i_Z),'.txt'];
                        j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
                    end
                else
                    cmd = ['cp ',sct.output_path,'tmp.dmri.group_mean.mat/','mat.T',numT{iGroup},'.txt',' ',sct.output_path,'mat_moco/','mat.T',numT{sct.dmri.group_indexes{iGroup}(dwi)},'.txt'];
                    j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
                end
			end			
        end
        
        % call motion corection module to registrate dwi_mean total on b0 mean with SPM
        j_disp(sct.log,['\n\n\n\n\n====================================================================='])
        j_disp(sct.log,['    STEP 2/2 : Estimate registration between each b0 on b0(end)'])
        j_disp(sct.log,['=====================================================================\n'])
        
        % call motion correction module for b0 group
        param = sct.dmri.moco_intra;
        param.todo = 'estimate'; % only estimate matrix (to minimize interpolation)
        param.split_data = 0;
        param.fname_data_splitT = [sct.output_path, 'tmp.dmri.data_splitT'];
        param.folder_mat = [sct.output_path,'mat_moco_b0/'];
        param.output_path = [sct.output_path, 'tmp_b0_moco'];
        param.fname_data = [sct.dmri.path,sct.dmri.file];
        param.fname_data_moco = [sct.dmri.path,sct.dmri.file,sct.dmri.suffix_moco];
        param.fname_target = [sct.output_path, 'tmp.dmri.data_splitT', numT{index_b0(end)}];
        param.fsloutput = fsloutput;
        param.index = sct.dmri.index_b0; % correct volumes of b0 group
        param.fname_log = sct.log;
        param.suffix = sct.dmri.suffix_moco;
        if any(strcmp(who,'centerline')), param.centerline=centerline; end
        
        j_mri_moco_v8(param);
        
        
        % Copy registration matrix of the closest dwi (closest to b0(1)) for all b0
        [dummy_variable,I] = min(abs(index_b0(end)-index_dwi));
        for iT = 1:length(index_b0)
            if sct.dmri.moco_intra.slicewise
                for i_Z = 1:sct.dmri.nz
                    cmd = ['cp ',sct.output_path,'mat_moco/','mat.T',num2str(index_dwi(I)),'_Z', num2str(i_Z),'.txt', ' ',sct.output_path,'mat_moco/','mat.T',num2str(index_b0(iT)), '_Z', num2str(i_Z),'.txt'];
                    j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
                end
            else
                cmd = ['cp ',sct.output_path,'mat_moco/','mat.T',numT{index_dwi(I)},' ',sct.output_path,'mat_moco/','mat.T',numT{index_b0(iT)},'.txt'];
                j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
            end
        end
        
        
        % Note transformation matrix folder for subsequent correction
        mat_folders.nb = mat_folders.nb + 1;
        mat_folders.names{mat_folders.nb} = [sct.output_path 'mat_moco/'];
        mat_folders.slicewise(mat_folders.nb) = sct.dmri.moco_intra.slicewise;
        
        % Note transformation matrix folder for subsequent correction
        mat_folders.nb = mat_folders.nb + 1;
        mat_folders.names{mat_folders.nb} = [sct.output_path 'mat_moco_b0/'];
        mat_folders.slicewise(mat_folders.nb) = sct.dmri.moco_intra.slicewise;
        save([sct.output_path 'workspace.mat'])
    end
    
    %----------------------------------------------------------------------
    % Smooth estimated motion
    if sct.dmri.moco_intra.smooth_motion
        sct_moco_spline([sct.output_path 'mat_moco/*T*Z*.txt'], sct.log)
    end
    
else
    % No intra-run motion correction
    j_disp(sct.log,'Skip this step.')
end





% ====================================================================
%	Apply motion correction
% ====================================================================
%
if ~strcmp(sct.dmri.moco_intra.method,'none')
    
    if mat_folders.nb % mat_folders.nb = nb of transformations done
        
        j_disp(sct.log,['\n\n   Apply transformations'])
        j_disp(sct.log,['-----------------------------------------------'])
        
        j_disp(sct.log,['.. Number of transformations: ',num2str(mat_folders.nb)])
      
        
        % Loop across additional transformations and multiply to the final matrices
        j_disp(sct.log,['\nLoop across additional transformations and multiply to the final matrices...'])
        fname_mat_final = [sct.output_path, 'mat_final/'];
        slicewise = sct_combine_transfo_matrix(mat_folders,[sct.dmri.nt,sct.dmri.nz],fname_mat_final,sct.log);
        
        % Apply transformation
        j_disp(sct.log,['\nApply transformation to each volume...'])
        param = sct.dmri.moco_intra;
        param.todo = 'apply'; % only estimate matrix (to minimize interpolation)
        param.split_data = 1;
        param.slicewise = slicewise;
        param.folder_mat = [sct.output_path, 'mat_final/'];
        param.output_path = [sct.output_path, 'tmp_moco/'];
        param.fname_data = [sct.dmri.path,sct.dmri.file_raw];
        param.fname_data_moco = [sct.dmri.path,sct.dmri.file,sct.dmri.suffix_moco]; % not used, not merged
        param.fsloutput = fsloutput;
        param.index = ''; % correct for all volumes
        param.fname_log = sct.log;
        param.merge_back = 0; % don't merge back data (in case b=0 need to be removed-- see next step)
        param.suffix = sct.dmri.suffix_moco;
        param.fname_target = param.fname_data;
        j_mri_moco_v8(param);
        save([sct.output_path 'workspace.mat'])
        
        
        
    end
    
    
    % Compute mean b=0 and DWI from corrected data
    % =========================================================================
    j_disp(sct.log,['\n   Compute mean b=0 and DWI from corrected data'])
    j_disp(sct.log,['-----------------------------------------------'])
    
    % Merge b=0 images
    j_disp(sct.log,['\nMerge b=0 images...'])
    fname_b0_merge = [sct.dmri.file_b0,sct.dmri.suffix_moco];
    cmd = [fsloutput,'fslmerge -t ',fname_b0_merge];
    for iT = 1:nb_b0
        cmd = cat(2,cmd,[' ' param.output_path 'tmp_moco.data_splitT_moco',numT{index_b0(iT)}]);
    end
    j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
    j_disp(sct.log,['.. File created: ',fname_b0_merge])
    
    % Average b=0 images
    j_disp(sct.log,['\nAverage b=0 images...'])
    fname_b0_mean = [fname_b0_merge,'_mean'];
    cmd = [fsloutput,'fslmaths ',fname_b0_merge,' -Tmean ',fname_b0_mean];
    j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
    j_disp(sct.log,['.. File created: ',fname_b0_mean])
    
    % Merge DWI images
    j_disp(sct.log,['\nMerge DWI images...'])
    for iT = 1:sct.dmri.nb_dwi
        opt_merge.fname_split{iT} = [param.output_path 'tmp_moco.data_splitT_moco',numT{index_dwi(iT)}];
    end
    fname_dwi_merge = [sct.dmri.file_dwi,sct.dmri.suffix_moco];
    opt_merge.fname_merge = fname_dwi_merge;
    opt_merge.fname_log   = sct.log;
    j_mri_merge(opt_merge); % this function is used to deal with potentially large number of files (>1000)
    j_disp(sct.log,['.. File created: ',fname_dwi_merge])
    
    % Average DWI images
    j_disp(sct.log,['\nAverage DWI images...'])
    fname_dwi_mean = [sct.dmri.file_dwi,sct.dmri.suffix_moco,'_mean'];
    cmd = [fsloutput,'fslmaths ',opt_merge.fname_merge,' -Tmean ',fname_dwi_mean];
    j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
    j_disp(sct.log,['.. File created: ',fname_dwi_mean])
    
    % update file name
    sct.dmri.file_b0 = [sct.dmri.file_b0,sct.dmri.suffix_moco];
    sct.dmri.file_dwi = [sct.dmri.file_dwi,sct.dmri.suffix_moco];
    
    
    % Correct bvecs
    % ============================================================
    if sct.dmri.moco_intra.correct_bvecs
        j_disp(sct.log,['\n\n  Correct bvecs'])
        j_disp(sct.log,['-----------------------------------------------'])
        % Open bvecs file
        j_disp(sct.log,['\nOpen bvecs file...'])
        fname_bvecs = [sct.dmri.path_bvecs,sct.dmri.file_bvecs];
        j_disp(sct.log,['.. File: ',fname_bvecs])
        bvecs = load(fname_bvecs);
        bvecs_new = zeros(sct.dmri.nt,3);
        fname_bvecs_moco = [sct.dmri.path,sct.dmri.file_bvecs,sct.dmri.suffix_moco];
        fid = fopen(fname_bvecs_moco,'w');
        
        for i_file = 1:sct.dmri.nt
            % read transfo matrix (if slicewise use the middle slice)
            fname_mat_final_T = dir([fname_mat_final 'mat.T',num2str(iT),'*']);
            fname_mat_final_T = [fname_mat_final fname_mat_final_T(ceil(end/2)).name];
            mat_final_T = load(fname_mat_final_T);
            
            %use rotation matrix to correct bvec
            R = mat_final_T(1:3,1:3);
            bvecs_new(i_file,:) = (R*bvecs(i_file,:)')';
            
            fprintf(fid,'%f %f %f\n',bvecs_new(i_file,1),bvecs_new(i_file,2),bvecs_new(i_file,3));
            
        end
        
        j_disp(sct.log,['.. File written: ',fname_bvecs_moco])
        
        % change the default file name
        suffix_bvecs = [suffix_bvecs,sct.dmri.suffix_moco];
        sct.dmri.file_bvecs = [sct.dmri.file_bvecs,suffix_bvecs];
    end
    
    
    
    
    %	Merge data back
    % ============================================================
    
    if sct.dmri.removeInterspersed_b0
        
        j_disp(sct.log,['\n\n\nRemove interspersed b=0 and merge data back'])
        j_disp(sct.log,['-----------------------------------------------'])
        
        
        % Merge data
        % =================================================================
        j_disp(sct.log,['\nMerge DW images and add the mean b=0 at the beginning...'])
        fname_data_merge = [sct.dmri.path,sct.dmri.file,sct.dmri.suffix_moco,sct.dmri.suffix_clean];
        cmd = [fsloutput,'fslmerge -t ',fname_data_merge];
        % add b=0 (take the first one)
        cmd = cat(2,cmd,' ',fname_b0_mean);
        % add DW images
        cmd = cat(2,cmd,' ',fname_dwi_merge);
        j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
        suffix_data = [suffix_data,sct.dmri.suffix_moco,sct.dmri.suffix_clean];
        sct.dmri.file = [sct.dmri.file,sct.dmri.suffix_moco,sct.dmri.suffix_clean];
        j_disp(sct.log,['.. File created: ',sct.dmri.file])
        
        
        % Clean bvecs/bvals
        % =================================================================
        j_disp(sct.log,['\nClean bvecs/bvals...'])
        
        % open bvecs/bvals
        fname_bvecs = [sct.dmri.path,sct.dmri.file_bvecs];
        bvecs = textread(fname_bvecs);
        if ~isempty(sct.dmri.file_bvals)
            sct.dmri.file_bvals = [sct.dmri.path,sct.dmri.file_bvals];
            bvals = textread(sct.dmri.file_bvals);
        end
        
        % create new bvecs/bvals
        bvecs_new = [];
        if ~isempty(sct.dmri.file_bvals), bvals_new = []; end
        
        % add b=0
        bvecs_new = bvecs(index_b0(1),:);
        if ~isempty(sct.dmri.file_bvals), bvals_new = bvals(index_b0(1)); end
        
        % add DW
        for iT = 1:sct.dmri.nb_dwi
            bvecs_new = cat(1,bvecs_new,bvecs(index_dwi(iT),:));
            if ~isempty(sct.dmri.file_bvals), bvals_new = cat(1,bvals_new,sct.dmri.gradients.bvals(index_dwi(iT))); end
        end
        j_disp(sct.log,['.. New number of directions: ',num2str(size(bvecs_new,1)-1),'+1'])
        
        % write new files
        fname_bvecs_new = [sct.dmri.path,sct.dmri.file_bvecs,sct.dmri.suffix_clean];
        j_dmri_gradientsWrite(bvecs_new,fname_bvecs_new,'fsl');
        sct.dmri.file_bvecs = [sct.dmri.file_bvecs,sct.dmri.suffix_clean];
        j_disp(sct.log,['.. File created: ',sct.dmri.file_bvecs])
        if ~isempty(sct.dmri.file_bvals)
            fname_bvals_new = [sct.dmri.path,sct.dmri.file_bvals,sct.dmri.suffix_clean];
            j_dmri_gradientsWriteBvalue(bvals_new,fname_bvals_new,'fsl');
            if ~isempty(sct.dmri.file_bvals), sct.dmri.file_bvals = [sct.dmri.file_bvals,sct.dmri.suffix_clean]; end
            j_disp(sct.log,['.. File created: ',sct.dmri.file_bvals]);
        end
        
        
    else
        
        j_disp(sct.log,['\n   Merge data back'])
        j_disp(sct.log,['-----------------------------------------------'])
        
        % Merge data back
        j_disp(sct.log,['\nConcatenate along T...'])
        fname_data_merge = [sct.dmri.path,sct.dmri.file,sct.dmri.suffix_moco];
        cmd = [fsloutput,'fslmerge -t ',fname_data_merge];
        for iT = 1:sct.dmri.nt
            cmd = cat(2,cmd,[' ' param.output_path 'tmp_moco.data_splitT',sct.dmri.suffix_moco,numT{iT}]);
        end
        j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
        suffix_data = [suffix_data,sct.dmri.suffix_moco];
        sct.dmri.file = [sct.dmri.file,sct.dmri.suffix_moco];
        j_disp(sct.log,['.. File created: ',sct.dmri.file])
        
    end
    
end

% delete temp files
j_disp(sct.log,['\nDelete temporary files...'])
cmd = ['rm -rf ' sct.output_path 'tmp*moco*'];
j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
cmd = ['rm -rf ' sct.output_path 'tmp*splitZ*'];
j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
cmd = ['rm -rf ' sct.output_path 'tmp*'];
j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end










% =========================================================================
%	REORDER DATA
% =========================================================================
if sct.dmri.reorder_data.do
    fname_data = [sct.dmri.path,sct.dmri.file];
    switch sct.dmri.reorder_data.method
        case 'bvecs_target'
            fname_bvecs_src = [sct.dmri.path,sct.dmri.file_bvecs];
            fname_bvecs_targ = sct.dmri.reorder_data.fname_target;
            j_dmri_reorganize_data(fname_data,fname_bvecs_src,fname_bvecs_targ);
        case 'bvalues'
            if exist(sct.dmri.reorder_data.orderingfile)
                fname_orderingfile = sct.dmri.reorder_data.orderingfile;
            elseif exist(sct.dmri.file_bvals)
                fname_orderingfile = sct.dmri.file_bvals;
            else
                error('can''t reorder data, no bvals files...')
            end
            sct_dmri_OrderByBvals(fname_data, fname_orderingfile)
    end
    
end

save([sct.output_path 'workspace.mat'])












% =========================================================================
%	CREATE MASK
% =========================================================================

j_disp(sct.log,['\n\n\n   Create mask'])
j_disp(sct.log,['-----------------------------------------------'])


j_disp(sct.log,['.. Mask generation method: "',sct.dmri.mask.method,'"'])

if strcmp(sct.dmri.mask.ref,'b0')
    mask_ref = [sct.dmri.file_b0,'_mean'];
elseif strcmp(sct.dmri.mask.ref,'dwi')
    mask_ref = [sct.dmri.file_dwi,'_mean'];
end
j_disp(sct.log,['.. Mask reference: "',sct.dmri.mask.ref,'"'])


switch sct.dmri.mask.method
    
    case 'bet' % create mask using BET
        
        j_disp(sct.log,['\nCreate mask using BET...'])
        fname_ref = [sct.dmri.path,mask_ref];
        fname_mask = [sct.dmri.path,sct.dmri.file_mask];
        cmd = [fsloutput,'bet2 ',fname_ref,' ',fname_mask,' -m -f ',num2str(sct.dmri.mask.bet_threshold)];
        j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
        
        % display mask
        if sct.dmri.mask.display
            reply1 = 'n';
            while strcmp(reply1,'n')
                mask = read_avw(fname_mask);
                j_displayMRI(mask);
                reply1 = input('Do you like this mask? y/n [y]: ', 's');
                if strcmp(reply1,'n')
                    txt = ['What threshold would you like? [previous value was ',num2str(sct.dmri.mask.bet_threshold),']: '];
                    reply2 = input(txt);
                    sct.dmri.mask.bet_threshold = reply2;
                    j_disp(sct.log,['\nGenerate new mask...'])
                    fname_ref = [sct.dmri.path,mask_ref];
                    fname_mask = [sct.dmri.path,sct.dmri.file_mask];
                    cmd = [fsloutput,'bet ',fname_ref,' ',fname_mask,' -f ',num2str(sct.dmri.mask.bet_threshold),' -m'];
                    j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
                end
            end
            clear reply1 reply2
            close % close figure
        end % if sct.dmri.mask.display
        
    case 'auto'
        
        j_disp(sct.log,['\nCreate mask by thresholding...'])
        % display stuff
        j_disp(sct.log,['.. FWHM=',num2str(sct.dmri.mask.auto.fwhm),'\n.. Threshold=',num2str(sct.dmri.mask.auto.threshold)])
        
        % display mask
        reply1 = 'n';
        while strcmp(reply1,'n')
            % smooth mean DWI
            j_disp(sct.log,['\nSmooth ref image...'])
            fname_ref = [sct.dmri.path,mask_ref];
            fname_dwi_smooth = [sct.dmri.path,'tmp_dwi_smooth'];
            cmd = [fsloutput,'fslmaths ',fname_ref,' -s ',num2str(sct.dmri.mask.auto.fwhm),' ',fname_dwi_smooth];
            j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
            % create mask
            j_disp(sct.log,['\nThreshold image...'])
            cmd = [fsloutput,'fslmaths ',fname_dwi_smooth,' -thr ',num2str(sct.dmri.mask.auto.threshold),' -bin ',sct.dmri.path,sct.dmri.file_mask];
            j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
            % display mask
            if sct.dmri.mask.display
                % load dwi_mean
                dwi_mean = read_avw(fname_ref);
                % load mask
                mask = read_avw([sct.dmri.path,sct.dmri.file_mask]);
                % multiply both images for display purpose
                dwi_mean_masked = dwi_mean.*mask;
                [min_mask index_min_mask] = sort([size(mask,1) size(mask,2) size(mask,3)],'descend');
                dwi_mean_masked = permute(dwi_mean_masked,index_min_mask);
                j_displayMRI(dwi_mean_masked);
                % 					j_displayMRI(mask);
                reply1 = input('Do you like this mask? y/n [y]: ', 's');
                if strcmp(reply1,'n')
                    txt = ['What FWHM would you like? [previous value was ',num2str(sct.dmri.mask.auto.fwhm),']: '];
                    reply2 = input(txt);
                    sct.dmri.mask.auto.fwhm = reply2;
                    txt = ['What intensity threshold would you like? [previous value was ',num2str(sct.dmri.mask.auto.threshold),']: '];
                    reply3 = input(txt);
                    sct.dmri.mask.auto.threshold = reply3;
                end
                close % close figure
            else
                reply1 = 'y';
            end
        end
        
        % Delete datasub
        j_disp(sct.log,['\nDelete temporary files...'])
        delete([sct.dmri.path,'tmp*.*']);
        
    case 'manual'
        
        if sct.dmri.mask.manual.ask
            % Ask the user to create a mask...
            j_disp(sct.log,['\n** Open a Terminal and go to the following directory: "',sct.dmri.path,'"'])
            j_disp(sct.log,['** Then, generate a mask using fslview based on the mean dwi image. To do this, type: "fslview dwi_mean"'])
            j_disp(sct.log,['** Once you''re happy with the mask, save it under the name "nodif_brain_mask.nii"'])
            j_disp(sct.log,['** Then go back to Matlab and press a key'])
            pause
        end
        
    case 'copy'
        
        j_disp(sct.log,'\nCopy mask...')
        copyfile(sct.dmri.mask.copy.fname,[sct.dmri.path])
        
    case 'none'
        
        j_disp(sct.log,'\nNo mask prescribed. Creating one that encompasses the whole volume...')
        
        %         % use the b0 image to create the mask
        %         fname_b0_mean = [sct.dmri.file_b0,'_mean',ext];
        %         fname_mask = [sct.dmri.path,sct.dmri.file_mask,ext];
        %         cmd = ['cp ',fname_b0_mean,' ',fname_mask];
        %         j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
        
end


save([sct.output_path 'workspace.mat'])




% ====================================================================
% PROCESS DATA
% ====================================================================

% Estimate the tensors
if sct.dmri.dti.do_each_run
    
    j_disp(sct.log,['\n\n   Estimate the tensors'])
    j_disp(sct.log,['-----------------------------------------------'])
    
    j_disp(sct.log,['Estimate DTI for each run...'])
    % estimate tensors using FSL
    cmd = [fsloutput,'dtifit -k ',sct.dmri.path,sct.dmri.file,...
        ' -m ',sct.dmri.path,sct.dmri.file_mask,...
        ' -o ',sct.dmri.path,sct.dmri.file_dti,...
        ' -r ',sct.dmri.path,sct.dmri.file_bvecs,...
        ' -b ',sct.dmri.path,sct.dmri.file_bvals];
    j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
    
    % Compute radial diffusivity
    j_disp(sct.log,['Compute radial diffusivity...'])
    fname_L2 = [sct.dmri.path,sct.dmri.file_dti,'_L2'];
    fname_L3 = [sct.dmri.path,sct.dmri.file_dti,'_L3'];
    fname_radialDiff = [sct.dmri.path,sct.dmri.file_radialDiff];
    cmd = [fsloutput,'fslmaths ',fname_L2,' -add ',fname_L3,' -div 2 ',fname_radialDiff];
    j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
    
    % display results
    j_disp(sct.log,['To check results, type:'])
    j_disp(sct.log,['fslview ',sct.dmri.path,'dti_FA ',sct.dmri.path,'dti_V1 ',sct.dmri.file_b0,'_mean ',sct.dmri.file_dwi,'_mean'])
    
end


% q-ball estimation
if sct.dmri.dtk.do
    j_disp(sct.log,['Estimate Q-Ball...'])
    
    % get numbers of B0
    % N.B. B0 SHOULD ALWAYS BE AT THE BEGGINING OF THE ACQUISITION!!! (NOT IN THE MIDDLE)
    bvals = textread(sct.dmri.file_bvals);
    nb_b0 = max(find(bvals==0));
    % create a gradient vector list compatible with DTK (should contain no b0)
    fname_bvecs = [sct.dmri.path,sct.dmri.file_bvecs];
    bvecs_dtk = textread(fname_bvecs);
    bvecs_dtk_nob0 = bvecs_dtk(nb_b0+1:end,:);
    fname_bvecs_dtk = [sct.dmri.path,sct.dmri.dtk.file_bvecs_dtk];
    fid = fopen(fname_bvecs_dtk,'w');
    fprintf(fid,'%f %f %f \n',bvecs_dtk_nob0);
    fclose(fid);
    % copy matrices file from DTK
    copyfile([sct.dmri.dtk.folder_mat,'*.dat'],'.');
    % estimate q-ball using DTK
    fname_data = [sct.dmri.path,sct.dmri.file];
    fname_qball = [sct.dmri.path,sct.dmri.dtk.file_qball];
    nb_dirs = size(bvecs_dtk_nob0,1);
    j_disp(sct.log,'\nEstimate q-ball')
    cmd = ['hardi_mat ',fname_bvecs_dtk,' ','temp_mat.dat',' -ref ',fname_data];
    [status result] = unix(cmd);
    cmd = ['odf_recon ',fname_data,' ',num2str(nb_dirs+1),' 181 ',fname_qball,' -b0 ',num2str(nb_b0),' -mat temp_mat.dat -nt -p 3 -sn 1 -ot nii'];
    [status result] = unix(cmd);
    % delete temp file
    delete('temp_mat.dat');
    delete('DSI_*.dat')
    
end


% % Generate shell scripts to launch BEDPOSTX on super-computer
% j_progress('Generate batch to run BedpostX ..........................')
% fname_batch = [sct.dmri.path,filesep,'batch_bedpostx.sh'];
% fid = fopen(fname_batch,'w');
% for i_nex = 1:sct.dmri.nex
% 	fprintf(fid,'echo ******************************************\n');
% 	fprintf(fid,'echo * Process series %s ...\n',['average_01-',num{i_nex}]);
% 	fprintf(fid,'echo ******************************************\n');
% 	fprintf(fid,'bedpostx %s -n 1\n',['average_01-',num{i_nex}]);
% 	j_progress(i_nex/sct.dmri.nex)
% end
% fclose(fid);
% j_progress(1)


save([sct.output_path 'workspace.mat']);
j_disp(sct.log,'done. ')
% j_disp(sct.log,'** To compute angular uncertainty, go to each folder and type: "bedpostx . -n 1 -j 1000 -s 10"')
% j_disp(sct.log,'** Then go back to Matlab and run "j_dmri_compute_uncertainty" in each folder individually')
j_disp(sct.log,' ')








% =========================================================================
% FUNCTION
% retrieve_gradients
%
% Retrieve gradient vectors for each header.
% =========================================================================
function [gradient_list] = retrieve_gradient(bvecs)

max_distance_vector = 0.001; % euclidean distance between two gradient vectors considered as being the same
% if ~exist('max_distance_vector'), max_distance_vector = 0.001; end
%
nb_headers = size(bvecs,1);
nb_directions = 0; % !!! INCLUDES B0 VALUE!!!!
for i_file=1:nb_headers
    % retrieve actual gradient
    gradient_tmp = bvecs(i_file,:);
    % compare actual gradient with previous ones
    found_existing_gradient = 0;
    for i_gradient=1:nb_directions
        distance_vector = (gradient_tmp(1)-gradient_list(i_gradient).direction(1))^2+(gradient_tmp(2)-gradient_list(i_gradient).direction(2))^2+(gradient_tmp(3)-gradient_list(i_gradient).direction(3))^2;
        distance_vector_neg = (gradient_tmp(1)+gradient_list(i_gradient).direction(1))^2+(gradient_tmp(2)+gradient_list(i_gradient).direction(2))^2+(gradient_tmp(3)+gradient_list(i_gradient).direction(3))^2;
        if (distance_vector < max_distance_vector) | (distance_vector_neg < max_distance_vector)
            % attibute present file index to existing direction
            gradient_list(i_gradient).index = cat(1,gradient_list(i_gradient).index,i_file);
            found_existing_gradient = 1;
        end
    end
    if ~found_existing_gradient
        % create new entry
        nb_directions = nb_directions + 1;
        gradient_list(nb_directions).direction = gradient_tmp;
        gradient_list(nb_directions).index = i_file;
    end
end

function display_moco(fname_mat)
list=dir(fname_mat);
list=sort_nat({list.name});
for imat=1:length(list), M_tmp=load(list{imat}); X(imat,:)=[M_tmp(1,4) M_tmp(2,4)]; end
A=X(:,1); A(A==0)=[];
plot(X(:,1))

function display_midleslice(hObject,data)
set(hObject,'Value',round(get(hObject,'Value'))); imagesc(data(:,:,ceil(end/2),round(get(hObject,'Value')))); colormap gray; axis image;




