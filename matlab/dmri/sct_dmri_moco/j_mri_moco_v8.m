function param = j_mri_moco_v8(param)
% =========================================================================
% Module that performs motion correction. Usable for anatomical, DTI and
% fMRI data.
% For details on the algorithm see:
% https://docs.google.com/drawings/d/1FoKXYbyFh_q20zsvl_mEcxlUR405gZ4c8DcrUvBxsIM/edit?hl=en_US
%
%
% INPUT
% param				structure
%
% MANDATORY
%   fname_data			string
%   fname_target		string      if ~apply
%
% FACULTATIVE
%   todo				string		'estimate' || 'apply' || 'estimate_and_apply'. NB: 'apply' requires input matrix. Default = 'estimate_and_apply'.
%   fname_data_moco		string
%   folder_mat			cell		ONLY IF todo=apply. Contains a cell of strings with file name of mat file used by FLIRT to apply correction.
%   output_path         sting       Default='tmp_moco/'
%	fname_log			string		fname to log file
%   slicewise			binary		0* || 1. slice by slice or volume-based motion correction?
%   index				1xn int		indexation that tells which files to correct
%   cost_function		string		'mutualinfo' || 'woods' || 'corratio' || 'normcorr' || 'normmi' || 'leastsquares'. Default is 'normcorr'.
%   flirt_options		string		Additional FLIRT options. E.g., '-interp sinc'. Default is ''.
%   merge_back			binary		0 || 1*. Merge data back after moco?
%   gaussian_mask       float       <sigma>. Default: 0. Weigth with gaussian mask? Sigma in mm --> std of the kernel. Can be a vector ([sigma_x sigma_y])
%   centerline          matrix
%
%
% HIGHLY FACULTATIVE
%	split_data			binary		0 || 1*. If data are already splitted, then indicated the file name in the following flag.
%       fname_data_splitT	string		Default='[output_path tmp_moco.data_splitT]'
%
% OUTPUT
% param
%
%   Example:
%   TODO
%
%
% TODO
% - manage interspersed for volume-based
% - no need to interpolate sinc when only estimating
% - for DTI, have a test that checks the mean intensity: if too low, use the registration matrix of the previous volume
%
% Julien Cohen-Adad <jcohen@nmr.mgh.harvard.edu>
% 2011-06-13: Created
% 2011-07-07: Modified
% 2011-10-07: esthetic modifs
% 2011-12-04: Only applies. Create a mat/ folder to put all the junk.
% 2011-12-11: Add a flag to avoid merging data after moco
%
% Fran�ois Marcoux <francois.marcoux@polymtl.ca>
% 2013-03-04: Modified so only transformation data is saved in mat folder
% 2013-03-20: In slicewise mode, the files split along z are now tmp_moco files.
%             If a file is not indexed, its moco version is no longer created.
% 2013-04-08: Added a suffix data in param. This suffix is added to the processed data
%
% =========================================================================

% debug if error
dbstop if error



% INITIALIZATIONS
fname_data				= param.fname_data;
[~,dims,scales] = read_avw(fname_data);
nt=dims(4); nz=dims(3);
if isfield(param,'suffix'), suffix = param.suffix; else suffix='_regis'; end
if isfield(param,'fname_data_moco'), fname_data_moco = param.fname_data_moco; else fname_data_moco = [fname_data, suffix]; end
if isfield(param,'todo'), todo = param.todo; else todo = 'estimate_and_apply'; end
if isfield(param,'program'), program = param.program; else program = 'FLIRT'; end
if isfield(param,'fname_target'), fname_target = param.fname_target; else fname_target = fname_data; end
if isfield(param,'folder_mat'), folder_mat = param.folder_mat; else if strcmp(todo,'estimate'), folder_mat = 'mat_moco/'; else folder_mat = 'tmp_moco.mat/';end ; end
if isfield(param,'output_path'), output_path = param.output_path; else output_path = ['tmp_moco',filesep]; end
if isfield(param,'split_data'), split_data = param.split_data; else split_data = 1; end
if isfield(param,'fname_data_splitT')&&~split_data, fname_data_splitT = param.fname_data_splitT; else fname_data_splitT = [output_path 'tmp_moco.data_splitT']; end
if isfield(param,'delete_tmp_files'), delete_tmp_files = param.delete_tmp_files; else delete_tmp_files = 1; end
if isfield(param,'cost_function_flirt'), cost_function_flirt = param.cost_function_flirt; else cost_function_flirt = 'normcorr'; end
if isfield(param,'cost_spm_coreg'), cost_spm_coreg = param.cost_spm_coreg; else cost_spm_coreg = 'nmi'; end
if isfield(param,'flirt_options'), flirt_options = param.flirt_options; else flirt_options = ''; end
if isfield(param,'dof'), dof = param.dof; else dof = 'TxTyTzSxSySzKxKyKz'; end
if isfield(param,'fname_log'), fname_log = param.fname_log; else fname_log = 'log_j_mri_moco_v8.txt'; end
if isfield(param,'fsloutput'), fsloutput = param.fsloutput; else fsloutput = 'export FSLOUTPUTTYPE=NIFTI; '; end
if isfield(param,'merge_back'), merge_back = param.merge_back; else merge_back = 1; end
if isfield(param,'slicewise'), slicewise = param.slicewise; else slicewise = 0; end
if isfield(param,'gaussian_mask'), mask = param.gaussian_mask; mask=mask(:); else mask = 0; end
if isfield(param,'centerline'), centerline = param.centerline; else centerline = repmat(ceil(dims/2)',dims(3),1); end
if isfield(param,'ref_weight'), ref_weight = param.ref_weight; end



nb_fails = 0;

options_spm_coreg = struct(...
    'graphics',0,... % Don't display graphics for spm_coreg2
    'cost_fun',cost_spm_coreg);
options_spm_reslice = struct(...
    'mask',1,... % don't mask anything
    'mean',0,... % write mean image
    'which',1,... % don't reslice the first image
    'wrap',[0 0 0]',...
    'interp',5); % the B-spline interpolation method

% START FUNCTION
j_disp(fname_log,['\n\n\n=========================================================================================================='])
j_disp(fname_log,['   Running: j_mri_moco_v8'])
j_disp(fname_log,['=========================================================================================================='])
j_disp(fname_log,['.. Started: ',datestr(now)])



% Check parameters
j_disp(fname_log,['\nCheck parameters:'])
j_disp(fname_log,['.. todo:              ',todo])
j_disp(fname_log,['.. fname_source:      ',fname_data])
j_disp(fname_log,['.. fname_target:      ',fname_target])
j_disp(fname_log,['.. suffix:            ',suffix])
j_disp(fname_log,['.. folder_mat:        ',folder_mat])
j_disp(fname_log,['.. split_data:        ',num2str(split_data)])
j_disp(fname_log,['.. fname_data_splitT: ',fname_data_splitT])
j_disp(fname_log,['.. cost_function:     ',cost_function_flirt])
j_disp(fname_log,['.. flirt_options:     ',flirt_options])
j_disp(fname_log,['.. Number of volumes: ',num2str(nt)])
j_disp(fname_log,['.. Number of slices:  ',num2str(nz)])
j_disp(fname_log,['.. Merge data back:   ',num2str(merge_back)])


% Create folder for mat files
if ~exist(output_path,'dir'), mkdir(output_path), end
if ~exist(folder_mat,'dir'), mkdir(folder_mat), end


% split into T dimension
if split_data
    j_disp(fname_log,['\nSplit along T dimension...'])
    cmd = [fsloutput,'fslsplit ',fname_data,' ',fname_data_splitT];
    j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
end
numT = j_numbering(nt,4,0);





% Generate schedule file for FLIRT
j_disp(fname_log,['\nCreate schedule file for FLIRT...'])

schedule_file = [folder_mat,'schedule_',dof];
switch (dof)
    case 'TxSx'
        fname_schedule = which('j_mri_schedule_TxSx.m');
        
    case 'TxSxKx'
        fname_schedule = which('j_mri_schedule_TxSxKx.m');
        
    case 'TxSxKy'
        fname_schedule = which('j_mri_schedule_TxSxKy.m');
        
    case 'TxSxKxKy'
        fname_schedule = which('j_mri_schedule_TxSxKxKy.m');
        
    case 'TxTyTzSxSySzKxKyKz'
        fname_schedule = which('j_mri_schedule_TxTyTzSxSySzKxKyKz.m');
        
    case 'Tx'
        fname_schedule = which('j_mri_schedule_Tx.m');
        
    otherwise
        schedule_file = [folder_mat,'schedule'];
        fname_schedule = dof;
        dof='UserTransfo';
end



% check if schedule file was found
if isempty(fname_schedule)
    error('Schedule file was not found. Thanks for playing with us.')
end
j_disp(fname_log,['.. Schedule file: ',fname_schedule])
unix(['cp ' fname_schedule ' ' schedule_file]);
j_disp(fname_log,['.. File created (locally): ',schedule_file])





% check if there is an indexation that tells which volume should be corrected
j_disp(fname_log,'\nCheck if there is an index that tells which volume to correct...')
if isfield(param,'index')
    if ~isempty(param.index)
        j_disp(fname_log,'Found!')
    else
        % no indexation. Create one that includes all the volumes.
        j_disp(fname_log,'No indexation found. Create one that includes all the volumes.')
        param.index = (1:nt);
    end
else
    % no indexation. Create one that includes all the volumes.
    j_disp(fname_log,'No indexation found. Create one that includes all the volumes.')
    param.index = (1:nt);
end
param.index=param.index(:);
j_disp(fname_log,['.. Index of volumes to correct: ',num2str(param.index')])



% volume-based or slice-by-slice motion correction?
j_disp(fname_log,'\nSlicewise motion correction?')
if slicewise
    % if slicewise, split target data along Z
    j_disp(fname_log,['Yes! Split target data along Z...'])
    fname_data_ref_splitZ = [output_path,'tmp_moco.target_splitZ'];
    cmd = [fsloutput,'fslsplit ',fname_target,' ',fname_data_ref_splitZ,' -z'];
    j_disp(fname_log,['>> ',cmd]);
    [status result] = unix(cmd);
    if status, error(result); end
    numZ = j_numbering(nz,4,0);
else
    j_disp(fname_log,['Nope!'])
end


% Generate Gaussian mask
j_disp(fname_log,'\nUse Mask?')
if mask
    fname_mask = [output_path 'tmp_moco.gaussian_mask_in'];
    if exist('ref_weight')
        if slicewise
            cmd=['fslsplit ' ref_weight ' ' fname_mask ' -z'];
            j_disp(fname_log,['>> ',cmd]);
            [status result] = unix(cmd);
            if status, error(result); end
            j_disp(fname_log,['.. File created: ',fname_mask])
            for iZ=1:dims(3)
                fslmask{iZ} = [' -inweight ' fname_mask '_' numZ{iZ} ' -refweight ' fname_mask '_' numZ{iZ}];
            end
        else
            fslmask=ref_weight;
        end
    else
        j_disp(fname_log,'Yes! Create mask...')
        %fname_mask_ref = [output_path 'tmp_moco.gaussian_mask_ref'];
        sigma=mask./scales(1:2);
        for iZ=1:dims(3)
            M_mask(:,:,iZ)=floor(gauss2d(dims, sigma, centerline(iZ,:))+1/2);
        end
        if slicewise
            for iZ=1:dims(3)
                save_avw_v2(M_mask(:,:,iZ),[fname_mask '_' numZ{iZ}],'f',scales, [fname_data_ref_splitZ '0000'],1);
                fslmask{iZ} = [' -inweight ' fname_mask '_' numZ{iZ} ' -refweight ' fname_mask '_' numZ{iZ}];
            end
        else
            save_avw_v2(M_mask,fname_mask,'f',scales, fname_data,1);
            fslmask = [' -refweight ' fname_mask];
        end
        j_disp(fname_log,['.. File created: ',fname_mask])
        
        disp('mask_preview :')
        S=num2str(floor(M_mask+0.5));
        S(:,2:3:end)=[];
        disp(S);
    end
else
    fslmask=cell(1,dims(3)); fslmask(1:dims(3))={''};
end


% MOTION CORRECTION

% Loop on T
j_disp(fname_log,['\n   Motion correction'])
j_disp(fname_log,['-----------------------------------------------'])
j_disp(fname_log,'Loop on iT...')

fail_mat = -ones(nt,nz);
for indice_index = 1:length(param.index)
    iT = param.index(indice_index);
    j_disp(fname_log,['\nVolume ',num2str(iT),'/',num2str(nt),':'])
    j_disp(fname_log,['--------------------'])
    
    % name of volume
    fname_data_splitT_num{iT} = [fname_data_splitT,numT{iT}];
    fname_data_splitT_moco_num{iT} = [fname_data_splitT,suffix,numT{iT}];
    
    % volume-based or slice-by-slice motion correction?
    j_disp(fname_log,'Slicewise motion correction?')
    if slicewise
        
        % SLICE-WISE MOTION CORRECTION
        % if slicewise, split data along Z
        j_disp(fname_log,['.. Yes!'])
        j_disp(fname_log,['Split data along Z...'])
        fname_data_splitT_splitZ = [fname_data_splitT_num{iT},'_splitZ'];
        cmd = [fsloutput,'fslsplit ',fname_data_splitT_num{iT},' ',fname_data_splitT_splitZ,' -z'];
        j_disp(fname_log,['>> ',cmd]);
        [status, result] = unix(cmd);
        if status, error(result); end
        
        
        
        % If slice regulation
        if strcmp(todo,'estimate') && strcmp(program,'ANTS')
            j_disp(fname_log,['Process with ANTS'])
            mat_tmp=[folder_mat,'mat.T',num2str(iT),'_tmp']; 
            if ~exist([folder_mat 'nifti_reg'],'dir'), mkdir([folder_mat 'nifti_reg']); end
            out= [folder_mat 'nifti_reg' filesep num2str(iT) '.nii'];
            cmd = ['isct_antsSliceRegularizedRegistration -p 2 --output [' mat_tmp ', ' out '] --transform Translation[0.1] --metric MeanSquares[ ' fname_target '.nii* , ' fname_data_splitT_num{iT} '.nii*  , 1 , 16 , Regular , 0.2 ] --iterations 20 -f 1 -s 2'];
            j_disp(fname_log,['>> ',cmd]);
            [status, result] = unix(cmd); if status, error(result); end
            unix(['rm ' out ' ' mat_tmp 'W* ' mat_tmp 'I*'])
            mat_tmp=[mat_tmp 'TxTy_poly.csv'];
        end
        
        
        % loop on Z
        j_disp(fname_log,['Loop on Z...'])
        for iZ = 1:nz
            % motion correction
            % TODO: add more options!x
            fname_data_splitT_splitZ_num{iT,iZ} = [fname_data_splitT_splitZ,numZ{iZ}];
            fname_data_splitT_splitZ_moco_num{iT,iZ} = [fname_data_splitT_splitZ_num{iT,iZ},suffix];
            fname_data_ref_splitZ_num{iZ} = [fname_data_ref_splitZ,numZ{iZ}];
            fname_mat{iT,iZ} = [folder_mat,'mat.T',num2str(iT),'_Z',num2str(iZ),'.txt'];
            switch(todo)
                
                case 'estimate'
                    switch (program)
                        case 'ANTS'
                            j_disp(fname_log,['convert ANTS matrix to FSL'])
                            % Open the text file.
                            fileID = fopen(mat_tmp,'r');
                            mocoArray = textscan(fileID, '%f%f%[^\n\r]', 'Delimiter', ',', 'HeaderLines' ,1, 'ReturnOnError', false);
                            fclose(fileID);
                            % Create transfo matrix
                            M=diag([1 1 1 1]); 
                            M(1,4)=-mocoArray{:, 1}(iZ);
                            M(2,4)=mocoArray{:, 2}(iZ);
                            
                            % save to FSL matrix file
                            sct_tools_matrix2txt(M,fname_mat{iT,iZ});
                            j_disp(fname_log,[fname_mat{iT,iZ} ' created']);
                            

                        case 'FLIRT'
                            j_disp(fname_log,['Process with FLIRT'])
                            cmd = [fsloutput,'flirt -schedule ', schedule_file, ' -in ',fname_data_splitT_splitZ_num{iT,iZ},' -ref ',fname_data_ref_splitZ_num{iZ},' -omat ',fname_mat{iT,iZ},' -out ',fname_data_splitT_splitZ_moco_num{iT,iZ},' -cost ',cost_function_flirt,fslmask{iZ},' ',flirt_options];
                            j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
                        case 'SPM'
                            j_disp(fname_log,['Process with SPM'])
                            %put ".nii" extension on files name
                            fname_data_splitT_splitZ_num_ext{iT,iZ}=[fname_data_splitT_splitZ_num{iT,iZ},'.nii'];
                            fname_data_ref_splitZ_num_ext{iZ}=[fname_data_ref_splitZ_num{iZ},'.nii'];
                            
                            % open matrix file
                            fid = fopen(fname_mat{iT,iZ},'w');
                            
                            % read input and reference headers
                            fname_data_splitT_splitZ_header=spm_vol(fname_data_splitT_splitZ_num_ext{iT,iZ});
                            ref_header=spm_vol(fname_data_ref_splitZ_num_ext{iZ});
                            
                            % generate transformation matrix
                            j_disp(fname_log,[' File to registrate : ', fname_data_splitT_splitZ_num_ext{iT,iZ}])
                            j_disp(fname_log,[' Reference : ', fname_data_ref_splitZ_num_ext{iZ}])
                            
                            transfo=spm_coreg2(ref_header,fname_data_splitT_splitZ_header,options_spm_coreg);
                            matrix_transfo = spm_matrix(transfo);
                            for line=1:4
                                fprintf(fid,'%f %f %f %f\n',matrix_transfo(line,1),matrix_transfo(line,2),matrix_transfo(line,3),matrix_transfo(line,4));
                            end
                            fclose(fid);
                    end
                    
                    
                case 'apply'
                    % build file name of input matrix
                    
                    switch(program)
                        
                        case 'FLIRT'
                            j_disp(fname_log,['Process with FLIRT'])
                            cmd = [fsloutput,'flirt -in ',fname_data_splitT_splitZ_num{iT,iZ},' -ref ',fname_data_ref_splitZ_num{iZ},' -applyxfm -init ',fname_mat{iT,iZ},' -out ',fname_data_splitT_splitZ_moco_num{iT,iZ},' ',flirt_options];
                             j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
                        case 'ANTS'
                            j_disp(fname_log,['Process with FLIRT'])
                            cmd = [fsloutput,'flirt -in ',fname_data_splitT_splitZ_num{iT,iZ},' -ref ',fname_data_ref_splitZ_num{iZ},' -applyxfm -init ',fname_mat{iT,iZ},' -out ',fname_data_splitT_splitZ_moco_num{iT,iZ},' ',flirt_options];
                             j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
                        case 'SPM'
                            j_disp(fname_log,['Process with SPM'])
                            fname_data_splitT_splitZ_num_ext{iT,iZ}=[fname_data_splitT_splitZ_num{iT,iZ},'.nii'];
                            fname_data_ref_splitZ_num_ext{iZ}=[fname_data_ref_splitZ_num{iZ},'.nii'];
                            
                            M_transfo = textread(fname_mat{iT,iZ});
                            M_transfo = M_transfo(1:4,1:4);
                            fname_data_splitT_splitZ_num_ext_header=spm_vol(fname_data_splitT_splitZ_num_ext{iT,iZ});
                            fname_data_splitT_splitZ_num_ext_header.mat=M_transfo^(-1)*fname_data_splitT_splitZ_num_ext_header.mat;
                            spm_get_space(fname_data_splitT_splitZ_num_ext{iT,iZ},fname_data_splitT_splitZ_num_ext_header.mat);
                            
                            options_spm_reslice.output = [fname_data_splitT_splitZ_moco_num{iT,iZ},'.nii'];
                            spm_reslice2({fname_data_ref_splitZ_num_ext{iZ};fname_data_splitT_splitZ_num_ext{iT,iZ}},options_spm_reslice);
                            
                    end
                    
                    
                case 'estimate_and_apply'
                    
                    
                    switch(program)
                        
                        case 'FLIRT'
                            j_disp(fname_log,['Process with FLIRT'])
                            cmd = [fsloutput,'flirt -schedule ', schedule_file, ' -in ',fname_data_splitT_splitZ_num{iT,iZ},' -ref ',fname_data_ref_splitZ_num{iZ},' -out ',fname_data_splitT_splitZ_moco_num{iT,iZ},' -omat ',fname_mat{iT,iZ},' -cost ', cost_function_flirt,fslmask{iZ},' ',flirt_options];
                             j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
                        case 'SPM'
                            j_disp(fname_log,['Process with SPM'])
                            % put ".nii" extension on files name
                            fname_data_splitT_splitZ_num_ext{iT,iZ}=[fname_data_splitT_splitZ_num{iT,iZ},'.nii'];
                            fname_data_ref_splitZ_num_ext{iZ}=[fname_data_ref_splitZ_num{iZ},'.nii'];
                            
                            fname_data_splitT_splitZ_num_ext_header=spm_vol(fname_data_splitT_splitZ_num_ext{iT,iZ});
                            ref_header=spm_vol(fname_data_ref_splitZ_num_ext{iZ});
                            
                            if ~strcmp(cost_spm_coreg,'none')
                                % generate transformation matrix
                                j_disp(fname_log,['Create transformation matrix... : '])
                                j_disp(fname_log,['Source : ',fname_data_splitT_splitZ_num_ext{iT,iZ}])
                                j_disp(fname_log,['Reference : ', fname_data_ref_splitZ_num_ext{iZ}])
                                
                                transfo=spm_coreg2(ref_header,fname_data_splitT_splitZ_num_ext_header,options_spm_coreg);
                                M_transfo = spm_matrix(transfo);
                                
                                % open and write matrix file
                                fid = fopen(fname_mat{iT,iZ},'w');
                                for line=1:4
                                    fprintf(fid,'%f %f %f %f\n',M_transfo(line,1),M_transfo(line,2),M_transfo(line,3),M_transfo(line,4));
                                end
                                fclose(fid);
                                
                                % modify file header
                                fname_data_splitT_splitZ_num_ext_header.mat=M_transfo^(-1)*fname_data_splitT_splitZ_num_ext_header.mat;
                                spm_get_space(fname_data_splitT_splitZ_num_ext{iT,iZ},fname_data_splitT_splitZ_num_ext_header.mat);
                            end
                            
                            % apply transformation
                            
                            options_spm_reslice.output = [fname_data_splitT_splitZ_moco_num{iT,iZ},'.nii'];
                            spm_reslice2({fname_data_ref_splitZ_num_ext{iZ};fname_data_splitT_splitZ_num_ext{iT,iZ}},options_spm_reslice);
                            
                            
                    end
                    
            end
            
            if strcmp(program , 'FLIRT') || strcmp(program , 'ANTS')
                j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
            end
            
            % Check transformation absurdity
            M_transfo = textread(fname_mat{iT,iZ});
            M_transfo = M_transfo(1:4,1:4);
            if ( abs(M_transfo(1,4)) > 10 || abs(M_transfo(2,4)) > 10 || abs(M_transfo(3,4) > 10) || abs(M_transfo(4,4) > 10) )
                nb_fails = nb_fails + 1;
                j_disp(fname_log,['failure #',num2str(nb_fails), ' this tranformation matrix is absurd, try others parameters (SPM, cost_function...) '])
                msgbox(['failure #',num2str(nb_fails), ' : (ref: ',fname_data_ref_splitZ_num{iZ},', in: ',fname_data_splitT_splitZ_num{iT,iZ},') this tranformation matrix is absurd, try others parameters (SPM, cost_function...) '], 'Correction failure','warn')
                fail_mat(iT,iZ) = 1;
            else
                fail_mat(iT,iZ) = 0;
            end
        end	% iZ
        
        if ~strcmp(todo,'estimate')
            % merge into Z dimension
            j_disp(fname_log,['Concatenate along Z...'])
            cmd = [fsloutput,'fslmerge -z ',fname_data_splitT_moco_num{iT}];
            for iZ = 1:nz
                cmd = strcat(cmd,[' ',fname_data_splitT_splitZ_moco_num{iT,iZ}]);
            end
            j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
            
            % remove tmp files
            j_disp(fname_log,['Remove temporary splited files...'])
            cmd = ['rm -f ' fname_data_splitT_num{iT} '*'];
            j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
        end
        
    else
        
        % Volume-based motion correction
        % =======================================================
        j_disp(fname_log,['.. Nope! Volume-based motion correction'])
        fname_mat{iT} = [folder_mat,'mat.T',numT{iT},'.txt'];
        switch(todo)
            
            case 'estimate'
                
                switch (program)
                    
                    case 'FLIRT'
                        j_disp(fname_log,['Process with FLIRT'])
                        cmd = [fsloutput,'flirt -schedule ', schedule_file, ' -in ',fname_data_splitT_num{iT},' -ref ',fname_target,' -omat ',fname_mat{iT},' -cost ',cost_function_flirt,fslmask{1},' ',flirt_options];
                        
                        
                    case 'SPM'
                        j_disp(fname_log,['Process with SPM'])
                        %put ".nii" extension on files name
                        fname_data_splitT_num_ext{iT}=[fname_data_splitT_num{iT},'.nii'];
                        fname_target_ext=[fname_target,'.nii'];
                        
                        
                        % read input and reference headers
                        fname_data_splitT_header=spm_vol(fname_data_splitT_num_ext{iT});
                        ref_header=spm_vol(fname_target_ext);
                        
                        % generate transformation matrix
                        j_disp(fname_log,['Create transformation matrix... : '])
                        j_disp(fname_log,['Source : ',fname_data_splitT_num_ext{iT}])
                        j_disp(fname_log,['Reference : ', fname_target_ext])
                        transfo=spm_coreg2(ref_header,fname_data_splitT_header,options_spm_coreg);
                        matrix_transfo = spm_matrix(transfo);
                        
                        % open and write matrix file
                        fid = fopen(fname_mat{iT},'w');
                        for line=1:4
                            fprintf(fid,'%f %f %f %f\n',matrix_transfo(line,1),matrix_transfo(line,2),matrix_transfo(line,3),matrix_transfo(line,4));
                        end
                        fclose(fid);
                end
                
                
            case 'apply'
                % build file name of input matrix
                fname_mat{iT} = [folder_mat,'mat.T',numT{iT},'.txt'];
                switch(program)
                    
                    case 'FLIRT'
                        j_disp(fname_log,['Process with FLIRT'])
                        cmd = [fsloutput,'flirt -in ',fname_data_splitT_num{iT},' -ref ',fname_target,' -applyxfm -init ',fname_mat{iT},' -out ',fname_data_splitT_moco_num{iT},' ',flirt_options];
                    case 'SPM'
                        j_disp(fname_log,['Process with SPM'])
                        fname_data_splitT_num_ext{iT}=[fname_data_splitT_num{iT},'.nii'];
                        fname_target_ext=[fname_target,'.nii'];
                        
                        M_transfo = textread(fname_mat{iT});
                        M_transfo = M_transfo(1:4,1:4);
                        fname_data_splitT_num_ext_header=spm_vol(fname_data_splitT_num_ext{iT});
                        fname_data_splitT_num_ext_header.mat=M_transfo^(-1)*fname_data_splitT_num_ext_header.mat;
                        spm_get_space(fname_data_splitT_num_ext{iT},fname_data_splitT_num_ext_header.mat);
                        
                        options_spm_reslice.output = [fname_data_splitT_moco_num{iT},'.nii'];
                        spm_reslice2({fname_target_ext;fname_data_splitT_num_ext{iT}},options_spm_reslice);
                        
                end
                
            case 'estimate_and_apply'
                
                switch (program)
                    
                    case 'FLIRT'
                        j_disp(fname_log,['Process with FLIRT'])
                        cmd = [fsloutput,'flirt -schedule ', schedule_file, ' -in ',fname_data_splitT_num{iT},' -ref ',fname_target,' -out ',fname_data_splitT_moco_num{iT},' -omat ',fname_mat{iT},' -cost ', cost_function_flirt,fslmask{1},' ',flirt_options];
                        
                        
                    case 'SPM'
                        j_disp(fname_log,['Process with SPM'])
                        % put ".nii" extension on files name
                        fname_data_splitT_num_ext{iT}=[fname_data_splitT_num{iT},'.nii'];
                        fname_target_ext=[fname_target,'.nii'];
                        
                        
                        
                        fname_data_splitT_header=spm_vol(fname_data_splitT_num_ext{iT});
                        ref_header=spm_vol(fname_target_ext);
                        
                        if ~strcmp(cost_spm_coreg,'none')
                            % generate transformation matrix
                            j_disp(fname_log,['Create transformation matrix... : '])
                            j_disp(fname_log,['Source : ',fname_data_splitT_num_ext{iT}])
                            j_disp(fname_log,['Reference : ', fname_target_ext])
                            
                            transfo=spm_coreg2(ref_header,fname_data_splitT_header,options_spm_coreg);
                            M_transfo = spm_matrix(transfo);
                            
                            
                            % open and write matrix file
                            fid = fopen(fname_mat{iT},'w');
                            transfo=spm_coreg2(ref_header,fname_data_splitT_header,options_spm_coreg);
                            matrix_transfo = spm_matrix(transfo); % Permet de convertir les donn�es de transfo en matrice de transfo
                            for line=1:4
                                fprintf(fid,'%f %f %f %f\n',matrix_transfo(line,1),matrix_transfo(line,2),matrix_transfo(line,3),matrix_transfo(line,4));
                            end
                            fclose(fid);
                            % modify file header
                            fname_data_splitT_header.mat=M_transfo^(-1)*fname_data_splitT_header.mat;
                            spm_get_space(fname_data_splitT_num_ext{iT},fname_data_splitT_header.mat);
                            
                        end
                        
                        % apply transformation
                        
                        options_spm_reslice.output = [fname_data_splitT_moco_num{iT},'.nii'];
                        spm_reslice2({fname_target_ext;fname_data_splitT_num_ext{iT}},options_spm_reslice);
                end
        end
       
        
        % Check transformation absurdity
        M_transfo = textread(fname_mat{iT});
        M_transfo = M_transfo(1:4,1:4);
        if ( abs(M_transfo(1,4)) > 10 || abs(M_transfo(2,4)) > 10 || abs(M_transfo(3,4) > 10) || abs(M_transfo(4,4) > 10) )
            nb_fails = nb_fails + 1;
            j_disp(fname_log,['failure #',num2str(nb_fails), ' this tranformation matrix is absurd, try others parameters (SPM, cost_function...) '])
            msgbox(['failure #',num2str(nb_fails), ' : (ref: ',fname_target,', in: ',fname_data_splitT_num{iT},') this tranformation matrix is absurd, try others parameters (SPM, cost_function...) '], 'Correction failure','warn')
        end
    end
    
    % END OF MOCO
    
    
end % loop on T

% replace failed transformation matrix to the closest good one
if slicewise
    
    % failed matrix index
    [fT,fZ] = find(fail_mat==1);
    % good matrix index
    [gT,gZ] = find(fail_mat==0);
    
    for iT = 1:length(fT)
        j_disp(fname_log,['Replace failed matrix T',num2str(fT(iT)), ' Z', num2str(fZ(iT))]);
        
        % rename failed matrix
        unix(['mv ' fname_mat{fT(iT),fZ(iT)} ' ' fname_mat{fT(iT),fZ(iT)} '_failed']);
        % find good slice number fZ(iT)
        good_Zindex = find(gZ == fZ(iT));
        good_index = gT(good_Zindex)
        % find
        [dummy,I] = min(abs(good_index-fT(iT)));
        cmd = ['cp ' fname_mat{good_index(I),fZ(iT)} ' ' fname_mat{fT(iT),fZ(iT)}];
        j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
        
        
    end
end


% merge data
if ~strcmp(todo,'estimate')
    if merge_back
        j_disp(fname_log,'\n\nMerge data back along T...')
        cmd = [fsloutput,'fslmerge -t ',fname_data_moco];
        for indice_index = 1:length(param.index)
            cmd = cat(2,cmd,[' ',fname_data_splitT_moco_num{param.index(indice_index)}]);
        end
        j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
        j_disp(fname_log,['.. File created: ',fname_data_moco])
        
        % Delete temp files
        if delete_tmp_files
            j_disp(fname_log,'\nDelete temporary files...')
            cmd = ['rm -rf ' output_path ' tmp_moco.*'];
            j_disp(fname_log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
        end
        
    end
end





% end
j_disp(fname_log,['\n.. Ended: ',datestr(now)])
j_disp(fname_log,['==========================================================================================================\n'])

function mat = gauss2d(dims, sigma, center)
[R,C] = ndgrid(1:dims(1), 1:dims(2));
mat = gaussC(R,C, sigma, center);

function val = gaussC(x, y, sigma, center)
xc = center(1);
yc = center(2);
exponent = ((x-xc).^2./(2*sigma(1)) + (y-yc).^2./(2*sigma(2)));
val       = (exp(-exponent));

