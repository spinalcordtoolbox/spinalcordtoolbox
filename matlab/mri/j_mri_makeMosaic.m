function j_mri_makeMosaic()
% =========================================================================
% 
% Convert single-file dicom to mosaic dicoms
% 
% INPUT
% 
% 
% OUTPUT
% 
% 
%   Example
%   j_mri_makeMosaic
%
% TODO
%
% Julien Cohen-Adad <jcohen@nmr.mgh.harvard.edu>
% 2011-10-08: Created
% 
% =========================================================================

% PARAMETERS


% INITIALIZATION
dbstop if error; % debug if error
if ~exist('opt'), opt = []; end
if ~isfield(opt,'fname_log'), opt.fname_log = 'log_j_mri_makeMosaic.txt'; end


% START FUNCTION
j_disp(opt.fname_log,['\n\n\n=========================================================================================================='])
j_disp(opt.fname_log,['   Running: j_mri_makeMosaic'])
j_disp(opt.fname_log,['=========================================================================================================='])
j_disp(opt.fname_log,['.. Started: ',datestr(now)])


N = 6120; 
I = zeros(100,100,N);
for count = 1:N

    if count <10
        I(:,:,count) = dicomread(['485000-000006-00000' num2str(count) '.dcm']);
    elseif count < 100
        I(:,:,count) = dicomread(['485000-000006-0000' num2str(count) '.dcm']);
    elseif count < 1000
        I(:,:,count) = dicomread(['485000-000006-000' num2str(count) '.dcm']);
    else
        I(:,:,count) = dicomread(['485000-000006-00' num2str(count) '.dcm']);        
    end
end

I = reshape(I,[100,100,60,102]);
I = I(:,:,:,3:102);

save_avw(double(I),'1x_siemens.nii','f',[2 2 2 2]);


figure; 
for count = 1:30
    subplot(5,6,count); imagesc(I(:,:,count)); 
end



% END FUNCTION
j_disp(opt.fname_log,['\n.. Ended: ',datestr(now)])
j_disp(opt.fname_log,['==========================================================================================================\n'])
