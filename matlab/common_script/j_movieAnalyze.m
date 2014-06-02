% =========================================================================
% FUNCTION
% j_movieAnalyze.m
%
% Make .avi movie from an MRI functional series
%
% DEPENDANCES
%
% COMMENTS
% Julien Cohen-Adad 2006-09-11
% =========================================================================
function j_movieAnalyze()


% initialization
avi_name = 'spine_subj1_run1.avi'
load fname_100;

% read functional volumes
[data,header,fname] = j_analyze_read('Enter functional series',fname,0,1);

% make a movie of components variation
load j_blueRed;
fig=figure('Name','Physiological variations','Position',[1 29 1276 957]);
set(fig,'DoubleBuffer','on');
mov = avifile(avi_name,'fps',10)
for it=5:1:20
    for iz=1:3
        subplot(2,2,iz)
        h = imagesc(data(:,:,iz,it),[-150 150]); axis square; colormap(j_blueRed); colorbar;
        title(['Slice number = ',num2str(iz)],'HorizontalAlignment','center');
        set(h,'EraseMode','xor');
    end
    subplot(2,2,4)
    t = text(0.1,0.5,['time = ',num2str(it*0.25)],'FontSize',50); axis off;
    F = getframe(fig);
    mov = addframe(mov,F);
    delete(t)
end
mov = close(mov);


% % find standard deviation for each voxel on ROI
% x_std = std(x_roi,0,2);
% clear x_roi;
% 
% % find mean of last standard deviations
% x_std2 = mean(x_std);
% x_std2std = std(x_std,0,1);
% clear x_std;





% % =========================================================================
% % find baseline
% % =========================================================================
% mri_series = j_analyze_read('Please select fMRI series');
% 
% % mask with roi
% for iTc=1:size(mri_series,4)
%     tc = mri_series(:,:,:,iTc);
%     tc_roi(:,iTc) = tc(roi);
% end
% clear mri_series tc;
% 
% % perform mean for each time series within mask
% tc_mean = mean(tc_roi,2);
% % tc_std = std(tc_roi,0,2);
% 
% % perform mean between all time series
% tc_mean2 = mean(tc_mean,1);
% % tc_std2 = std(tc_mean,0,1);
% clear tc_mean;
% 
