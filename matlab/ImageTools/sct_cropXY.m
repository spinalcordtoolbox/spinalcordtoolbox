function sct_cropXY(data_file,varargin)
%sct_cropXY(data_file [,method,param])
% EXAMPLES:
% sct_cropXY data_file.nii
% sct_cropXY data_file.nii centerline 20
% sct_cropXY data_file.nii autobox
% sct_cropXY data_file.nii autobox 30
% sct_cropXY('data_file.nii', 'autobox', '30')
%
% METHODS
% 1- centerline (default)
% 2- box (not functional)
% 3- manual (not functional)
% 4- autobox --> crop at the center of the image
%
% PARAM
% margin around the spinal cord 30 --> crop around 3cm in X and Y

dbstop if error

[data_file,data_path]=sct_tool_remove_extension(data_file,0);
fname_data = [data_path,data_file];

if isempty(varargin), crop_method='centerline'; else crop_method=varargin{1}; end
if length(varargin)>1, crop_margin=str2num(varargin{2}); else crop_margin=30; end


[~, dims] = read_avw(fname_data);

% Find which SHELL is running
disp(['\nFind which SHELL is running...'])
[status result] = unix('echo $0');
if ~isempty(findstr(result,'bash'))
    sct.dmri.shell = 'bash';
elseif ~isempty(findstr(result,'tsh'))
    sct.dmri.shell = 'tsh';
elseif ~isempty(findstr(result,'tcsh'))
    sct.dmri.shell = 'tsh';
else
    disp(['.. Failed to identify shell. Using default.'])
    sct.dmri.shell = 'tsh';
end
disp(['.. Running: ',sct.dmri.shell])
% FSL output
if strcmp(sct.dmri.shell,'bash')
    fsloutput = ['export FSLOUTPUTTYPE=NIFTI; ']; % if running BASH
elseif strcmp(sct.dmri.shell,'tsh') || strcmp(sct.dmri.shell,'tcsh')
    fsloutput = ['setenv FSLOUTPUTTYPE NIFTI; ']; % if you're running C-SHELL
else
    error('Check SHELL field.')
end



switch (crop_method)
    
    case 'manual'
        
        disp(['\n\n   Crop data'])
        disp(['-----------------------------------------------'])
        
        disp(['.. Cropping method: ',crop_method])
        
        % display stuff
        % split the data into Z dimension
        j_progress('Split the data into Z dimension ...............')
        
        fname_data_splitZ = [data_path,'tmp.dmri.data_splitZ'];
        cmd = [fsloutput,'fslsplit ',fname_data,' ',fname_data_splitZ,' -z'];
        [status result] = unix(cmd); % run UNIX command
        if status, error(result); end % check error
        j_progress(1)
        % split the mask into Z dimension
        j_progress('Split the cropping mask into Z dimension ......')
        fname_mask = [data_path,sct.dmri.crop.file_crop];
        fname_mask_splitZ = [data_path,'tmp.dmri.mask_splitZ'];
        cmd = [fsloutput,'fslsplit ',fname_mask,' ',fname_mask_splitZ,' -z'];
        [status result] = unix(cmd); % run UNIX command
        if status, error(result); end % check error
        j_progress(1)
        % Crop each slice individually
        j_progress('Crop each slice individually ..................')
        numZ = j_numbering(dims(3),4,0);
        for iZ = 1:dims(3)
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
            save([data_path 'tmp.dmri.crop_box.mat'],'minX','maxX','minY','maxY','minZ','maxZ');
            disp(['... File created: ','tmp.dmri.crop_box.mat'])
            
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
        fname_data_crop = [data_path,data_file,'_crop'];
        cmd = [fsloutput,'fslmerge -z ',fname_data_crop,' ',fname_data_crop_splitZ];
        [status result] = unix(cmd);
        if status, error(result); end
        j_progress(1)
        % delete temp files
        delete([data_path,'tmp.dmri.*'])
        
        
        
    case 'box'
        
        disp(['\n\n   Crop data'])
        disp(['-----------------------------------------------'])
        
        disp(['.. Cropping method: ',crop_method])
        disp(['... Crop size: ',sct.dmri.crop.size])
        j_progress('Crop data .....................................')
        
        
        fname_datacrop = [data_path,data_file,'_crop'];
        cmd = [fsloutput,'fslroi ',fname_data,' ',fname_datacrop,' ',sct.dmri.crop.size];
        
        [status result] = unix(cmd); % run UNIX command
        if status, error(result); end % check error
        
        % save box coordonates
        box = strread(sct.dmri.crop.size);
        minX = box(1); maxX = box(2)-box(1); minY = box(3); maxY = box(4)-box(3); minZ = box(5); maxZ = box(6)-box(5);
        save([data_path 'tmp.dmri.crop_box.mat'],'minX','maxX','minY','maxY','minZ','maxZ');
        disp(['... File created: ','tmp.dmri.crop_box.mat'])
        
        
    case 'autobox'
        
        disp(['\n\n   Crop data'])
        disp(['-----------------------------------------------'])
        
        disp(['.. Cropping method: ',crop_method])
        
        if crop_margin < 3, margin=15; else margin = crop_margin; end % crop size around centerline
        
        fname_datacrop = [data_path,data_file,'_crop'];
        cmd = [fsloutput,'fslroi ',fname_data,' ',fname_datacrop,' ',num2str(dims(1)/2-margin),' ',num2str(2*margin),' ',num2str(dims(2)/2-margin),' ',num2str(2*margin),' 0 -1'];
        disp(['>> ',cmd]);
        [status result] = unix(cmd); % run UNIX command
        if status, error(result); end % check error
   
        
        
    case 'centerline'
        disp(['\n\n   Crop data'])
        disp(['-----------------------------------------------'])
        disp(['.. Cropping method: ',crop_method])
        
        % extract one image (to reslice center_line from anat to dw images)
        cmd = [fsloutput,'fslroi ',fname_data,' ','tmp.dmri.crop.',data_file, '_1',' ','1 1'];
        disp(['>> ',cmd]);
        [status result] = unix(cmd);
        if status, error(result); end
        
        file_data_1 = ['tmp.dmri.crop.',data_file, '_1'];
        disp(['... File created: ',file_data_1])
        
        
        % reslice centerline to dmri space
        if exist([data_path,'anat/Spinal_Cord_Segmentation/centerline.nii.gz']) || exist([data_path,'anat/Spinal_Cord_Segmentation/centerline.nii'])
            
            
            % convert centerline.nii.gz to centerline.nii
            if exist([data_path,'anat/Spinal_Cord_Segmentation/centerline.nii.gz'])
                cmd = ['fslchfiletype NIFTI ',data_path,'anat/Spinal_Cord_Segmentation/centerline'];
                disp(['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
            end
            
            % reslice centerline in file_data space
            resflags = struct(...
                'mask',1,... % don't mask anything
                'mean',0,... % write mean image
                'which',1,... % write everything else
                'wrap',[0 0 0]',...
                'interp',1,... % linear interp?
                'output',['tmp.dmri.crop.centerline_resliced.nii']);
            spm_reslice2({[file_data_1,'.nii'];[data_path,'anat/Spinal_Cord_Segmentation/centerline.nii']},resflags);
            
            sct.dmri.centerline_file = 'tmp.dmri.crop.centerline_resliced';
            disp(['... File created: ',sct.dmri.centerline_file])
        end
        
        
        j_progress('Crop data .....................................')
        
        fname_datacrop = [data_path,data_file,'_crop'];
        
        centerline = sct_get_centerline([file_data_1 '.nii']);
        
        if crop_margin < 3, margin=15; else margin = crop_margin; end % crop size around centerline
        minX = min(centerline(:,1))- margin;
        maxX = max(centerline(:,1))+ margin;
        minY = min(centerline(:,2))- margin;
        maxY = max(centerline(:,2))+ margin;
        minZ = 0;
        maxZ = dims(3);
        
        % prevent a crop bigger than the image data
        if minX<0, minX=0; end, if maxX>dims(1), maxX=dims(1); end
        if minY<0, minY=0; end, if maxY>dims(2), maxY=dims(2); end
        if minZ<0, minZ=0; end, if maxZ>dims(3), maxZ=dims(3); end
        
        % save box coordonates
        save([data_path 'tmp.dmri.crop_box.mat'],'minX','maxX','minY','maxY','minZ','maxZ');
        disp(['... File created: ','tmp.dmri.crop_box.mat'])
        
        % compute centerline_crop
        centerline(:,1)=centerline(:,1)-minX;
        centerline(:,2)=centerline(:,2)-minY;
        
        % perform cropping with whole spine min/max postions
        cmd = [fsloutput,'fslroi ',fname_data,' ',fname_datacrop,' ',num2str(minX),' ',num2str(maxX-minX),' ',num2str(minY),' ',num2str(maxY-minY),' ',num2str(minZ),' ',num2str(maxZ-minZ)];
        disp(['>> ',cmd]); [status result] = unix(cmd); % run UNIX command
        if status, error(result); end % check error
        
        % delete temp files
        disp(['\nDelete temporary files...'])
        cmd = ['rm -rf tmp.dmri.crop.*'];
        disp(['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
        
    case 'none'
        disp(['\n\n   Crop data'])
        disp(['-----------------------------------------------'])
        disp(['.. No croping'])
        
    otherwise
        disp(['\n\n   Crop data'])
        disp(['-----------------------------------------------'])
        error(['croping method ' crop_method ' isn''t correct'])
end