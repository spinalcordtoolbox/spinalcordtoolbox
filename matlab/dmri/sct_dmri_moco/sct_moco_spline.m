function sct_moco_spline(fname_mat, varargin)
% sct_moco_spline(fname_mat, fname_log(optional) )
% sct_moco_spline('mat.*')
dbstop if error
if ~isempty(varargin), log_spline = varargin{1}; else log_spline = 'log_sct_moco_spline'; end
if length(varargin)>1, ind_ab = varargin{2}; else ind_ab = []; end


j_disp(log_spline,['\nSmoothing Patient Motion...'])
% LOAD MATRIX
[list, path]=sct_tools_ls(fname_mat);

Z_index=cellfun(@(x) cell2mat(textscan(x,'%*[mat.T]%*u%*[_Z]%u%*[.txt]')),list);
T=cellfun(@(x) cell2mat(textscan(x,'%*[mat.T]%u%*[_Z]%*u%*[.txt]')),list); T=single(T);
j_progress('loading matrix...')
for imat=1:length(list), j_progress(imat/length(list)); M_tmp{imat}=load([path list{imat}]); X(imat)=M_tmp{imat}(1,4); Y(imat)=M_tmp{imat}(2,4); end
j_progress('elapsed')

color=jet(max(Z_index));
% Plot subject movement
figure(28); hold off;
for iZ=unique(Z_index)
    subplot(2,1,1); plot(T(Z_index==iZ),X(Z_index==iZ),'+','Color',color(iZ,:)); ylim([min(X)-0.5 max(X)+0.5]); hold on
    if iZ==1, hold off; end
    subplot(2,1,2); plot(T(Z_index==iZ),Y(Z_index==iZ),'+','Color',color(iZ,:)); ylim([min(Y)-0.5 max(Y)+0.5]); hold on
    
   % TZ=T(Z_index==iZ);
    % abrupt motion detection
    %installPottslab
%     u=minL1Potts(Y(Z_index==iZ), 10, 'samples',T(Z_index==iZ));
%     v=minL1Potts(X(Z_index==iZ), 10, 'samples',T(Z_index==iZ));
%[ind_ab TZ(find(diff(u))) TZ(find(diff(v)))];
end
drawnow;

%% Get abrupt motion volume #
ind_ab = inputdlg('Enter space-separated numbers:',...
             'Volume# before abrupt motion (starting at 1)',[1 150]);
if isempty(ind_ab), ind_ab=[]; else ind_ab = str2num(ind_ab{:}); end
j_disp(log_spline,['Abrupt motion on Volume #: ' num2str(ind_ab)])
ind_ab=[0 ind_ab max(T)];




%% GENERATE SPLINE
msgbox({'Use the slider (figure 28, bottom) to calibrate the smoothness of the regularization along time' 'Press any key when are done..'})

hsl = uicontrol('Style','slider','Min',-10,'Max',0,...
                'SliderStep',[1 1]./10,'Value',-2,...
                'Position',[20 20 200 20]);
set(hsl,'Callback',@(hObject,eventdata) GenerateSplines(X,Y,T,Z_index,ind_ab,10^(get(hObject,'Value')),color ))


pause

[Xout, Yout]=GenerateSplines(X,Y,T,Z_index,ind_ab,10^(get(hsl,'Value')),color)
j_disp(log_spline,['...done!'])
%% SAVE MATRIX
j_progress('\nSave Matrix...')
% move old matrix
if ~exist([path 'old'],'dir'); mkdir([path 'old']); end
unix(['mv ' fname_mat ' ' path 'old/'])

% create new list
for iT=1:max(T)
    for iZ=unique(Z_index)
        mat_name=['mat.T' num2str(iT) '_Z' num2str(iZ) '.txt'];
        % update matrix
        M=diag([1 1 1 1]);
        M(1,4)=Xout(iZ,iT); M(2,4)=Yout(iZ,iT);
        % write matrix
        fid = fopen([path mat_name],'w');
        fprintf(fid,'%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n%f %f %f %f\n',[M(1,1:4), M(2,1:4), M(3,1:4), M(4,1:4)]);
        fclose(fid);
        
    end
end


function [Xout,Yout]=GenerateSplines(X,Y,T,Z_index,ind_ab,smoothness,color)

figure(28), hold off,
for iZ=unique(Z_index)
    Xtmp=X(Z_index==iZ); Ytmp=Y(Z_index==iZ); Ttmp=T(Z_index==iZ);
    for iab=2:length(ind_ab)
        disp(['Generate motion splines...'])
        Tpiece=ind_ab(iab-1)+1:ind_ab(iab);
        
        index=Ttmp>ind_ab(iab-1) & Ttmp<=ind_ab(iab);% Piece index
        if length(find(index))>1
            Xfitresult=spline(Ttmp(index),Xtmp(index),smoothness); Yfitresult=spline(Ttmp(index),Ytmp(index),smoothness);
            Xout(iZ,Tpiece) = feval(Xfitresult,Tpiece); Yout(iZ,Tpiece)=feval(Yfitresult,Tpiece);
        else
            [~,closestT_l]=min(abs(Ttmp-mean([ind_ab(iab), ind_ab(iab-1)])));
            Xout(iZ,Tpiece)=Xtmp(closestT_l);
            Yout(iZ,Tpiece)=Ytmp(closestT_l);
        end
    end
    % plot splines
    Ttotal=1:max(T);
    subplot(2,1,1); plot(T(Z_index==iZ),X(Z_index==iZ),'+','Color',color(iZ,:)); ylim([min(X)-0.5 max(X)+0.5]); hold on
    subplot(2,1,1); plot(Ttotal,Xout(iZ,:),'-','Color',color(iZ,:));  ylim([min(X)-0.5 max(X)+0.5]); legend('raw moco', 'smooth moco', 'Location', 'NorthEast' ); grid on; ylabel( 'X Displacement (mm)' ); xlabel('volume #');
    if iZ==1, hold off; end
    subplot(2,1,2); plot(T(Z_index==iZ),Y(Z_index==iZ),'+','Color',color(iZ,:)); ylim([min(Y)-0.5 max(Y)+0.5]); hold on
    subplot(2,1,2); plot(Ttotal,Yout(iZ,:),'-','Color',color(iZ,:));  ylim([min(Y)-0.5 max(Y)+0.5]); legend('raw moco', 'smooth moco', 'Location', 'NorthEast' ); grid on; ylabel( 'Y Displacement (mm)' ); xlabel('volume #');
end
drawnow;


function fitresult = spline(T,M_motion_t,smoothness)

%% Fit: 'sct_moco_spline'.
[xData, yData] = prepareCurveData( T, M_motion_t );

% Set up fittype and options.
ft = fittype( 'smoothingspline' );
opts = fitoptions( ft );
opts.SmoothingParam = smoothness;

% Fit model to data.
[fitresult, gof] = fit( xData, yData, ft, opts );
M_motion_t = feval(fitresult,T);




