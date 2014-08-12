function sct_moco_spline(fname_mat, varargin)
% sct_moco_spline(fname_mat, fname_log(optional) )
if ~isempty(varargin), log_spline = varargin{1}; else log_spline = 'log_sct_moco_spline'; end

j_disp(log_spline,['\nSmoothing Patient Motion...'])
% LOAD MATRIX
list=dir(fname_mat);
path=[fileparts(fname_mat) filesep];
list=sort_nat({list.name});
j_progress('loading matrix...')
for imat=1:length(list), j_progress(imat/length(list)); M_tmp{imat}=load([path list{imat}]); X(imat)=M_tmp{imat}(1,4); Y(imat)=M_tmp{imat}(2,4); end
j_progress('elapsed')

% Plot subject movement
h=figure; plot(X,'+')
g=figure; plot(Y,'+')
% Search for abrupt moves
ind_ab=1;
ind_ab=[ind_ab find(abs(diff(Y))>2) find(abs(diff(X))>2) length(X)]; ind_ab=sort(ind_ab); ind_ab(diff(ind_ab)<15)=[];

% GENERATE SPLINE
for iab=2:length(ind_ab)
    j_disp(log_spline,['Generate motion splines...'])
    X(ind_ab(iab-1):ind_ab(iab))=spline(X(ind_ab(iab-1):ind_ab(iab))); Y(ind_ab(iab-1):ind_ab(iab))=spline(Y(ind_ab(iab-1):ind_ab(iab)));
end
figure(h), hold on; plot(X,'r+'); hold off; legend('raw moco', 'smooth moco', 'Location', 'NorthEast' ); grid on; ylabel( 'Displacement (mm)' ); xlabel('volume #');
figure(g); hold on; plot(Y,'r+'); hold off; legend('raw moco', 'smooth moco', 'Location', 'NorthEast' ); grid on; ylabel( 'Displacement (mm)' ); xlabel('volume #');

j_disp(log_spline,['...done!'])
% SAVE MATRIX
j_progress('\nSave Matrix...')
% move old matrix
if ~exist([path 'old'],'dir'); mkdir([path 'old']); end
unix(['mv ' fname_mat ' ' path 'old/'])
for imat=1:length(list),
    j_progress(imat/length(list))
    % update matrix
    M_tmp{imat}(1,4)=X(imat); M_tmp{imat}(2,4)=Y(imat);
    % write matrix
    fid = fopen([path list{imat}],'w');
    fprintf(fid,'%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n',[M_tmp{imat}(1,1:4), M_tmp{imat}(2,1:4), M_tmp{imat}(3,1:4), M_tmp{imat}(4,1:4)]);
    fclose(fid);
    
end


function M_motion_t = spline(M_motion_t)

%% Fit: 'sct_moco_spline'.
[xData, yData] = prepareCurveData( [], M_motion_t );

% Set up fittype and options.
ft = fittype( 'smoothingspline' );
opts = fitoptions( ft );
opts.SmoothingParam = 1e-06;

% Fit model to data.
[fitresult, gof] = fit( xData, yData, ft, opts );
M_motion_t = feval(fitresult,1:length(M_motion_t));




