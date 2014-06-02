function [sct, status] = sct_dcm_to_nii_v4(sct)
% =========================================================================
% Module that processes DW data. Called by batch_dcm2nii.m.
%
% this function also copies each file into the appropriate folder, e.g.,
% the anatomcial file will be copied into the 'anat' folder.

% The algorithm for the motion correction module is here:

% TODO
% THIS FILE SHOULD BE MADE GENERIC!!  CURRENTLY IT HAS HARD-CODED FILES IN THERE...
%
% =========================================================================


dbstop if error

if isfield(sct,'fname_log'), fname_log = sct.fname_log, else fname_log = 'log_sct_dcm_to_nii.txt'; end
j_disp(fname_log,['\n\n\n=========================================================================================================='])
j_disp(fname_log,['   Running: sct_dcm_to_nii.m'])
j_disp(fname_log,['=========================================================================================================='])
j_disp(fname_log,['.. Started: ',datestr(now)])



% Initialization

bvals = 0;
bvecs = 0;


% name
data_type{1} ='';
if exist([sct.input_path, sct.(sct.input_files_type).dmri.folder],'dir')&& ~isempty(sct.(sct.input_files_type).dmri.folder), data_type = [data_type 'dmri']; end
if exist([sct.input_path, sct.(sct.input_files_type).anat.folder],'dir')&& ~isempty(sct.(sct.input_files_type).anat.folder), data_type = [data_type 'anat']; end
if exist([sct.input_path, sct.(sct.input_files_type).disco.folder],'dir') && ~isempty(sct.(sct.input_files_type).disco.folder), data_type = [data_type 'disco']; end
if exist([sct.input_path, sct.(sct.input_files_type).mtr_ON.folder],'dir') && exist([sct.input_path, sct.(sct.input_files_type).mtr_OFF.folder],'dir') && ~isempty(sct.(sct.input_files_type).mtr_ON.folder) && ~isempty(sct.(sct.input_files_type).mtr_OFF.folder)
    data_type = [data_type 'mtr_ON'];
    data_type = [data_type 'mtr_OFF'];
end
data_type(cellfun(@isempty,data_type)) = [];

% outputtype
switch(sct.outputtype)
    case 'NIFTI'
        ext = '.nii';
    case 'NIFTI_GZ'
        ext = '.nii.gz';
end

% Make directories
for file_img = 1:length(data_type)
    if ~exist([sct.output_path,data_type{file_img},filesep])
		mkdir([sct.output_path,data_type{file_img},filesep]); 
	end
end

% Start conversion for anat, dmri and mtr files



if strcmp(sct.input_files_type,'dicom')
    
    
    
    
    % loop over dicom series
    for file_img = 1:length(data_type)
        
        % Initialize file name
        dicom_file = dir ([sct.input_path,sct.dicom.(data_type{file_img}).folder,'/*.dcm']);
        sct.dicom.(data_type{file_img}).file               = dicom_file(1,1).name;
            

        % Get file names
        j_disp(fname_log,['\nGet file names...'])
        list_fname = dir([sct.input_path,sct.dicom.(data_type{file_img}).folder,filesep]);
        nb_files = size(list_fname,1);
        j_disp(fname_log,['.. Number of files: ',num2str(nb_files)])
        
        
        switch sct.convert_dicom.program
            
            
            case 'MRI_CONVERT'
                
                % use freesurfer tool to convert to nifti
                
                j_disp(fname_log,['\nConvert DICOM to NIFTI using FreeSurfer...'])
                cmd = ['export UNPACK_MGH_DTI=0; mri_convert ',sct.input_path, sct.dicom.(data_type{file_img}).folder,filesep,sct.dicom.(data_type{file_img}).file,' ',[sct.output_path,data_type{file_img},filesep,data_type{file_img},'_data.nii']];
                j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
                
                
            case 'DCM2NII'
                
                % use dcm2nii to convert to nifti
                
                j_disp(fname_log,['\nConvert DICOM to NIFTI using dcm2nii...'])
                cmd=['dcm2nii -a n -c n -d n -e n -f y -g n -m n -n y -o ',sct.output_path,data_type{file_img},' -p n -r n ',[sct.input_path,sct.dicom.(data_type{file_img}).folder]];
                j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
                
                % rename output file
                
                [status, list] = unix(['cd ',sct.output_path,data_type{file_img},'/; ls']);
                list = strread(list,'%s');
                
                for i = 1:length (list)
                    if strfind (list{i,1},'.bval')
                        bvals=i;
                    elseif strfind (list{i,1},'.bvec')
                        bvecs=i;
                    elseif strfind (list{i,1},'.nii')
                        datas = i;
                        
                    end
                end
                
                % rename data
                cmd = ['mv ',[sct.output_path,data_type{file_img},filesep,list{datas,1}],' ',[sct.output_path,data_type{file_img},filesep,data_type{file_img},'_data.nii']];
                j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
                
                
                if strcmp(data_type{file_img},'dmri')
                    
                    if bvals ~=0 || bvecs ~=0
                        
                        % rename bvecs bvals
                        cmd = ['mv ',[sct.output_path,data_type{file_img},filesep,list{bvals,1}],' ',[sct.output_path,data_type{file_img},filesep,'bvals.txt']];
                        j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
                        cmd = ['mv ',[sct.output_path,data_type{file_img},filesep,list{bvecs,1}],' ',[sct.output_path,data_type{file_img},filesep,'bvecs.txt']];
                        j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
                    end
                    
                end
        end
        
        % change default data name
        sct.dicom.(data_type{file_img}).file = [data_type{file_img},'_data'];
    end
    
    
    % save structure
    j_disp(fname_log,['\nSave structure...'])
    fname_struct = [data_type{file_img}];
    j_disp(fname_log,['.. Output file: ',fname_struct,'.mat'])
    save(fname_struct);
    
    % save structure
    save sct sct
    
    

% JULIEN: THIS NEEDS TO BE CHANGED! IF THE USER HAS NII.GZ INSTEAD OF NII, THIS DOES NOT WORK ANYMORE
elseif strcmp(sct.input_files_type,'nifti')
    
    if ~strcmp(sct.output_path,sct.input_path)
        
        nifti_file_anat         = dir ([sct.input_path,sct.nifti.anat.folder,'/*.nii']);
        nifti_file_dmri         = dir ([sct.input_path,sct.nifti.dmri.folder,'/*.nii']);
        [status, nifti_file_bvecs] = unix(['ls ' sct.input_path,sct.nifti.dmri.folder '/*bvec*']);
        nifti_file_bvecs        = strread(nifti_file_bvecs,'%s');
        nifti_file_bvals        = dir ([sct.input_path,sct.nifti.dmri.folder,'/*bval*']);
        nifti_file_mtr_ON       = dir ([sct.input_path,sct.nifti.mtr_ON.folder,'/*.nii']);
        nifti_file_mtr_OFF      = dir ([sct.input_path,sct.nifti.mtr_OFF.folder,'/*.nii']);
        nifti_file_disco        = dir ([sct.input_path,sct.nifti.disco.folder,'/*.nii']);
        
 
        
        for file_img = 1:length(data_type)
            
           if ~strcmp(sct.nifti.(data_type{file_img}).folder,'')
                if strcmp(data_type{file_img},'anat')
                    [path,sct.nifti.anat.file,ext]               = fileparts(nifti_file_anat(1,1).name); % file of anatamical data. IF DICOMS, CHOOSE ONE FILE
                elseif strcmp(data_type{file_img},'dmri')
                    [path,sct.nifti.dmri.file,ext]               = fileparts(nifti_file_dmri(1,1).name); % file of diffusion data. IF DICOMS, CHOOSE ONE FILE
                elseif strcmp(data_type{file_img},'mtr_ON')
                    [path,sct.nifti.mtr_ON.file,ext]             = fileparts(nifti_file_mtr_ON(1,1).name); % magnetization transfert image ON. IF DICOMS, CHOOSE ONE FILE
                elseif strcmp(data_type{file_img},'mtr_OFF')
                    [path,sct.nifti.mtr_OFF.file,ext]            = fileparts(nifti_file_mtr_OFF(1,1).name); % magnetization transfert image OFF. IF DICOMS, CHOOSE ONE FILE
                elseif strcmp(data_type{file_img},'disco')
                    [path,sct.nifti.disco.file,ext]              = fileparts(nifti_file_disco(1,1).name);
            
                end
            end
        end
                
        for file_img = 1:length(data_type)
          if ~strcmp(sct.nifti.(data_type{file_img}).folder,'')  

            % copy to output folder
            cmd = ['cp -f ',[sct.input_path,sct.nifti.(data_type{file_img}).folder,filesep,sct.nifti.(data_type{file_img}).file,'.nii'],' ',[sct.output_path,data_type{file_img},filesep,data_type{file_img},'_data.nii']];
            j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
            
            if strcmp(data_type{file_img},'dmri')
                if ~isempty(nifti_file_bvecs)
                    % copy bvecs and bvals files to output folder
                    cmd = ['cp -f ',nifti_file_bvecs{1},' ',[sct.output_path,'dmri',filesep,'bvecs.txt']];
                    j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
                    if ~isempty(nifti_file_bvals)
                        cmd = ['cp -f ',[sct.input_path,sct.nifti.dmri.folder,filesep,nifti_file_bvals(1,1).name],' ',[sct.output_path,'dmri',filesep,'bvals.txt']];
                        j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
                    end
                else error('No bvec file')
                end
            end
          end
        end
    end
    
    
else error(['input type ', sct.input_files_type,' is incorrect']);
end


% JULIEN
% THIS THING SHOULD NOT GO THERE-- IT HAS NOTHING TO DO WITH DCM2NII CONVERSION!
% % --------------------------------------------------------------------------------------------------------------------------------
% % Process bvals bvecs
if max(strcmp('dmri',data_type))~=0
    % get gradient vectors
    
    if ~exist([sct.output_path,'dmri/bvecs.txt']) && strcmp(sct.input_files_type,'dicom');
        % Get gradient vectors
        j_disp(fname_log,['\nGet gradient vectors...']);
        opt.path_read  = [sct.input_path,sct.dicom.dmri.folder,filesep];
        opt.path_write = [sct.output_path,'dmri/'];
        opt.file_bvecs = 'bvecs.txt';
        opt.file_bvals = 'bvals.txt';
        opt.verbose = 1;
        if strcmp(sct.dmri.gradients.referential,'XYZ')
            
            opt.correct_bmatrix = 1;
            
        elseif strcmp(sct.dmri.gradients.referential,'PRS')
            
            opt.correct_bmatrix = 0;
            
        end
        
        gradients = j_dmri_gradientsGet(opt);
        
    end
    
    
    
    % read in gradient vectors
    
    j_disp(fname_log,['\nRead in gradient vectors...'])
    
    % bvecs
    
    fname_bvecs = [sct.output_path,'dmri/bvecs.txt'];
    j_disp(fname_log,['.. File bvecs: ',fname_bvecs])
    sct.dmri.gradients.bvecs = load(fname_bvecs);
    
    % bvals
    fname_bvals = [sct.output_path,'dmri/bvals.txt'];
    
    if exist(fname_bvals)
        j_disp(fname_log,['.. File bvals: ',fname_bvals])
        sct.dmri.gradients.bvals = load(fname_bvals);
    else
        j_disp(fname_log,['.. !! bvals file is empty. Must be DSI data.'])
    end
    
    % check directions
    
    j_disp(fname_log,['.. Number of directions: ',num2str(size(sct.dmri.gradients.bvecs,1))])
    if exist(fname_bvals)
        j_disp(fname_log,['.. Maximum b-value: ',num2str(max(sct.dmri.gradients.bvals)),' s/mm2'])
    end
    
    
    % flip gradient
    flip = sct.dmri.gradients.flip;
    if flip(1)~=1 || flip(2)~=2 || flip(3)~=3
        j_disp(fname_log,['\nFlip gradients...'])
        j_disp(fname_log,['.. flip options: ',num2str(flip)])
        fname_bvecs = [sct.output_path,'dmri/bvecs.txt'];
        gradient = load(fname_bvecs);
        sct.dmri.file_bvecs = [sct.dmri.file_bvecs,'_flip',num2str(flip(1)),num2str(flip(2)),num2str(flip(3))];
        fname_bvecs_new = [sct.output_path,'dmri/bvecs.txt'];
        fid = fopen(fname_bvecs_new,'w');
        for i=1:size(gradient,1)
            G = [sign(flip(1))*gradient(i,abs(flip(1))),sign(flip(2))*gradient(i,abs(flip(2))),sign(flip(3))*gradient(i,abs(flip(3)))];
            fprintf(fid,'%1.10f %1.10f %1.10f\n',G(1),G(2),G(3));
        end
        fclose(fid);
        j_disp(fname_log,['.. File written: ',fname_bvecs_new])
        
    end
    
    % change default bvecs and bvals files names
    sct.dmri.file_bvecs = 'bvecs.txt';
    sct.dmri.file_bvals = 'bvals.txt';
%     
% end




% --------------------------------------------------------------------------------------------------------------------------------
% Change default files name

for file_img = 1:length(data_type)
      % Merge mtr_ON and mtr_OFF files
    if strfind(data_type{file_img}, 'mtr')
        if ~exist([sct.output_path,'mtr/']), mkdir([sct.output_path,'mtr/']); end
        if exist([sct.output_path,'mtr_ON',filesep,'mtr_ON_data.nii'])
            cmd = ['mv -f ',[sct.output_path,'mtr_ON',filesep,'mtr_ON_data.nii'],' ',sct.output_path,'mtr/'];
            j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end        
        elseif exist([sct.output_path,'mtr_OFF',filesep,'mtr_OFF_data.nii'])
            cmd = ['mv -f ',[sct.output_path,'mtr_OFF',filesep,'mtr_OFF_data.nii'],' ',sct.output_path,'mtr/'];
            j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
    
        end
         sct.mtr.folder = 'mtr';
    end


    % Change default files name
    sct.(data_type{file_img}).file = [data_type{file_img},'_data'];
    sct.(data_type{file_img}).folder = [data_type{file_img}];
    if strfind(data_type{file_img}, 'mtr')
        sct.(data_type{file_img}).folder = 'mtr';
    end
end
status = 0;




end


