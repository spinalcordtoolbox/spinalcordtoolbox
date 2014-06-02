function sct_moco_spline_slice_regulation(fname_mat, varargin)
% sct_moco_spline(fname_mat, fname_log(optional) )
if ~isempty(varargin), log_spline = varargin; else log_spline = 'log_sct_moco_spline'; end

if ~isempty(dir([fname_mat '*T*Z*.txt']))
    slicewise=1;
else
    slicewise=0;
end

j_disp(log_spline,['\nSmoothing Patient Motion...'])
% LOAD MATRIX
list=dir(fname_mat); list = list( cellfun(@(list) ~isempty(strfind(list,'T')), {list(:).name}) );
list=sort_nat({list.name});
j_progress('loading matrix...')
for imat=1:length(list), j_progress(imat/length(list)); M_tmp{imat}=load(list{imat}); X(imat)=M_tmp{imat}(1,4); Y(imat)=M_tmp{imat}(2,4); end
j_progress('elapsed')

% GENERATE SPLINE
j_disp(log_spline,['Generate motion splines...'])
X=spline(X); Y=spline(Y);
j_disp(log_spline,['...done!'])

% SAVE MATRIX
j_progress('\nSave Matrix...')
% move old matrix
if ~exist('old','dir'); mkdir('old'); end
unix(['mv ' fname_mat ' old/'])
for imat=1:length(list),
    j_progress(imat/length(list))
    % update matrix
    M_tmp{imat}(1,4)=X(imat); M_tmp{imat}(2,4)=Y(imat);
    % write matrix
    fid = fopen(list{imat},'w');
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

% Plot fit with data.
figure( 'Name', 'smooth moco' );
h = plot( fitresult, xData, yData );
legend( h, 'raw moco', 'smooth moco', 'Location', 'NorthEast' );
% Label axes
ylabel( 'A' );
grid on


