function sct = sct_process_mtr_v1(sct)
% =========================================================================
% Module to coregistrate and generate MTR
% 
% INPUT
% mtr				structure
%
% MANDATORY
%   file_ON	    	string		
%   file_OFF			string
%
% FACULTATIVE
%	path            	string		
%	cost_spm_coreg      string      'mi'  - Mutual Information, 'nmi' - Normalised Mutual Information, 'ecc' - Entropy Correlation Coefficient, 'ncc' - Normalised Cross Correlation. default: 'nmi'
%   fname_output        string      default : MTR
%   fsloutput           string      default : NIFTI
%
% OUTPUT
% param
% 
% 
% =========================================================================

% debug if error
dbstop if error



% INITIALIZATIONS
file_ON				= sct.mtr_ON.file;
file_OFF           	= sct.mtr_OFF.file;
if isfield(sct.mtr,'path'), path = sct.mtr.path; else path = './'; end
if isfield(sct.mtr,'cost_spm_coreg'), cost_spm_coreg = sct.mtr.cost_spm_coreg; else cost_spm_coreg = 'nmi'; end
if isfield(sct.mtr,'fname_output'), fname_output = sct.mtr.fname_output; else fname_output='MTR'; end
if isfield(sct.mtr,'outputtype'), outputtype = sct.outputtype; else outputtype = 'NIFTI'; end
if isfield(sct.mtr,'file_log'), file_log = sct.log; else file_log = 'log_process_mtr.txt'; end
if isfield(sct.mtr,'file_log'), file_log = sct.log; else file_log = 'log_process_mtr.txt'; end
if isfield(sct.mtr,'shell'), shell = sct.mtr.shell; else shell = 'bash'; end

options_spm_coreg = struct(...
                    'graphics',0,... % Don't display graphics for spm_coreg2
                    'cost_fun',cost_spm_coreg); 
                      

% FSL output
if strcmp(shell,'bash')
        fsloutput = ['export FSLOUTPUTTYPE=',outputtype,'; ']; % if running BASH
elseif strcmp(sct.mtr.shell,'tsh') || strcmp(sct.mtr.shell,'tcsh')
        fsloutput = ['setenv FSLOUTPUTTYPE ',outputtype,'; ']; % if you're running C-SHELL
else
        error('Check SHELL field.')
end
                
% START FUNCTION
j_disp(file_log,['\n\n\n=========================================================================================================='])
j_disp(file_log,['   Running: sct_process_mtr_v1(mtr)'])
j_disp(file_log,['=========================================================================================================='])
j_disp(file_log,['.. Started: ',datestr(now)])



% Check parameters
j_disp(file_log,['\nCheck parameters:'])
j_disp(file_log,['.. path:              ',path])
j_disp(file_log,['.. file_ON:          ',file_ON])
j_disp(file_log,['.. file_OFF:         ',file_OFF])
j_disp(file_log,['.. fname_output:      ',fname_output])





j_disp(file_log,['\n   Crop MTR files'])
j_disp(file_log,['-----------------------------------------------'])
if sct.mtr.crop.do == 1
    if ~exist([sct.output_path,'anat/Spinal_Cord_Segmentation/centerline.nii.gz']) && ~exist([sct.output_path,'anat/Spinal_Cord_Segmentation/centerline.nii'])
        msgbox(['No segmentation files, can''t apply center_line crop. Activate option segmentation.do'], 'NO CROP','warn');
    else
        % Get files dimension
        cmd = ['fslsize ',path,file_ON];
        [status result] = unix(cmd);
        if status, error(result); end
        dims = j_mri_getDimensions(result);
        sct.mtr.nx = dims(1);
        sct.mtr.ny = dims(2);
        sct.mtr.nz = dims(3);
        sct.mtr.nt = dims(4);
        
        % reslice centerline in file_data space
        resflags = struct(...
            'mask',1,... % don't mask anything
            'mean',0,... % write mean image
            'which',1,... % write everything else
            'wrap',[0 0 0]',...
            'interp',1,... % linear interp?
            'output',['tmp.mtr.crop.centerline_resliced.nii']);
        spm_reslice2({[path,file_ON,'.nii'];[sct.output_path,'anat/Spinal_Cord_Segmentation/centerline.nii']},resflags);
        
        crop.centerline_file = 'tmp.mtr.crop.centerline_resliced';
        j_disp(sct.log,['... File created: ',crop.centerline_file])
        
        % read centerline file
        [centerline]=read_avw(crop.centerline_file);
        [dimX,dimY,dimZ]=size(centerline);
        
        % clean the centerline matrix
        centerline(isnan(centerline)) = 0;
        centerline = boolean(centerline);
        
        % define xmin xmax ymin ymax zmin zmax
        [X,Y] = find(centerline);
        if isempty(X)
            msgbox(['Segmentation file is not on the field of the data'], 'NO CROP','warn');
        else
            Z = floor(Y/dimY);
            Y = Y - dimY*Z;
            Z = Z + 1;
            
            if sct.mtr.crop.margin < 3, margin=15; else margin = sct.mtr.crop.margin; end % crop size around centerline
            minX = min(X) - margin; maxX = max(X) + margin;
            minY = min(Y) - margin; maxY = max(Y) + margin;
            minZ = min(Z) - margin; maxZ = max(Z) + margin;
            
            % prevent a crop bigger than the image data
            if minX<1, minX=1; end, if maxX>sct.mtr.nx, maxX=sct.mtr.nx; end
            if minY<1, minY=1; end, if maxY>sct.mtr.ny, maxY=sct.mtr.ny; end
            if minZ<1, minZ=1; end, if maxZ>sct.mtr.nz, maxZ=sct.mtr.nz; end
            
            
            % perform cropping with whole spine min/max postions
            cmd = [fsloutput,'fslroi ',path,file_ON,' ',path,file_ON,'_croped ',num2str(minX),' ',num2str(maxX-minX),' ',num2str(minY),' ',num2str(maxY-minY),' ',num2str(minZ-1),' ',num2str(maxZ-minZ+1)];
            j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); % run UNIX command
            cmd = [fsloutput,'fslroi ',path,file_OFF,' ',path,file_OFF,'_croped ',num2str(minX),' ',num2str(maxX-minX),' ',num2str(minY),' ',num2str(maxY-minY),' ',num2str(minZ-1),' ',num2str(maxZ-minZ+1)];
            j_disp(sct.log,['>> ',cmd]); [status result] = unix(cmd); % run UNIX command
            if status, error(result); end % check error
            
            % change default files name
            file_ON = [file_ON,'_croped'];
            file_OFF = [file_OFF,'_croped'];
        end
    end
end





j_disp(file_log,['\n   Calculate MTR'])
j_disp(file_log,['-----------------------------------------------'])
j_disp(file_log,['\nCo-register ON/OFF images...'])


% Add .nii extension to files name
file_ON = [file_ON,'.nii'];
file_OFF = [file_OFF,'.nii'];

% Read images
source=spm_vol([path,file_ON]); 
ref=spm_vol([path,file_OFF]);

% Estimate coregistration matrix
transfo=spm_coreg2(ref,source,options_spm_coreg); 
matrix_transfo=spm_matrix(transfo);

% apply estimation on file_ON's vox2real matrix
spm_get_space([path,file_ON],matrix_transfo^(-1)*source.mat); 

% reslice file_ON to fit file_OFF
options_spm_reslice = struct(...
                          'mask',1,... % don't mask anything
                          'mean',0,... % write mean image
                          'which',1,... % write everything else
                          'wrap',[0 0 0]',...
                          'interp',5,... % linear interp?
                          'output',[path,strrep(file_ON,'.nii',''),'_registrated.nii']);
spm_reslice2({[path,file_OFF];[path,file_ON]},options_spm_reslice); 
file_ON = [strrep(file_ON,'.nii',''),'_registrated.nii'];
j_disp(file_log,['\n       ... File created : ',file_ON])

% fslmaths , compute MTR
j_disp(file_log,['\nCalculate the MTR with fslmaths...'])
cmd = [fsloutput,'fslmaths -dt double ', path, file_OFF, ' -sub ', path, file_ON, ' -mul 100 -div ', path, file_OFF, ' -thr 0 -uthr 100 ', path, fname_output];
j_disp(file_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
j_disp(file_log,['\n       ... File created : ',fname_output])

% end
j_disp(file_log,['\n.. Ended: ',datestr(now)])
j_disp(file_log,['==========================================================================================================\n'])

