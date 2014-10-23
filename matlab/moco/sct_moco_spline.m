function sct_moco_spline(fname_mat, varargin)
% ind_abrupt=sct_moco_spline(fname_mat, fname_log(optional) )
if ~isempty(varargin), log_spline = varargin{1}; else log_spline = 'log_sct_moco_spline'; end
if length(varargin)>1, ind_ab = varargin{2}; else ind_ab = []; end


j_disp(log_spline,['\nSmoothing Patient Motion...'])
% LOAD MATRIX
list=dir(fname_mat);
path=[fileparts(fname_mat) filesep];
list=sort_nat({list.name});
Z_index=cellfun(@(x) cell2mat(textscan(x,'%*[mat.T]%*u%*[_Z]%u%*[.txt]')),list);
T=cellfun(@(x) cell2mat(textscan(x,'%*[mat.T]%u%*[_Z]%*u%*[.txt]')),list); T=single(T);
j_progress('loading matrix...')
for imat=1:length(list), j_progress(imat/length(list)); M_tmp{imat}=load([path list{imat}]); X(imat)=M_tmp{imat}(1,4); Y(imat)=M_tmp{imat}(2,4); end
j_progress('elapsed')

color=jet(max(Z_index));
% Plot subject movement
figure(28); hold off; figure(29); hold off; 
for iZ=unique(Z_index)
    figure(28); plot(T(Z_index==iZ),X(Z_index==iZ),'+','Color',color(iZ,:)); hold on
    figure(29); plot(T(Z_index==iZ),Y(Z_index==iZ),'+','Color',color(iZ,:)); hold on
    
    TZ=T(Z_index==iZ);
    % abrupt motion detection
    u=minL1Potts(Y(Z_index==iZ), 10, 'samples',T(Z_index==iZ));
    v=minL1Potts(X(Z_index==iZ), 10, 'samples',T(Z_index==iZ));
    ind_ab=[ind_ab TZ(find(diff(u))) TZ(find(diff(v)))];
end
ind_ab=sort(ind_ab); ind_ab=ind_ab(~(diff(ind_ab)<15));


% if ~isempty(ind_ab)
%     ind_ab = input(['abrut motion detect around volume ' num2str(ind_ab) '.. specify exact volume # (before motion) as a matrix:']);
% end

ind_ab=[0 ind_ab max(T)];

% GENERATE SPLINE
for iZ=unique(Z_index)
    Xtmp=X(Z_index==iZ); Ytmp=Y(Z_index==iZ); Ttmp=T(Z_index==iZ);
    for iab=2:length(ind_ab)
        j_disp(log_spline,['Generate motion splines...'])
        Tpiece=ind_ab(iab-1)+1:ind_ab(iab);
        
        index=Ttmp>ind_ab(iab-1) &Ttmp<=ind_ab(iab);% Piece index
        if length(find(index))>1
        Xfitresult=spline(Ttmp(index),Xtmp(index)); Yfitresult=spline(Ttmp(index),Ytmp(index));
        Xout(iZ,Tpiece) = feval(Xfitresult,Tpiece); Yout(iZ,Tpiece)=feval(Yfitresult,Tpiece);
        else
            Xout(iZ,Tpiece)=(Xtmp(find(index)+1)-Xtmp(index))/(max(Tpiece)-min(Tpiece))*(Tpiece-Tpiece(1))+Xtmp(index);
            Yout(iZ,Tpiece)=(Ytmp(find(index)+1)-Ytmp(index))/(max(Tpiece)-min(Tpiece))*(Tpiece-Tpiece(1))+Ytmp(index);
        end
    end
    % plot splines
    Ttotal=1:max(T);
    figure(28), hold on; plot(Ttotal,Xout(iZ,:),'-','Color',color(iZ,:)); hold off; legend('raw moco', 'smooth moco', 'Location', 'NorthEast' ); grid on; ylabel( 'X Displacement (mm)' ); xlabel('volume #');
    figure(29); hold on; plot(Ttotal,Yout(iZ,:),'-','Color',color(iZ,:)); hold off; legend('raw moco', 'smooth moco', 'Location', 'NorthEast' ); grid on; ylabel( 'Y Displacement (mm)' ); xlabel('volume #');
end



j_disp(log_spline,['...done!'])
% SAVE MATRIX
j_progress('\nSave Matrix...')
% move old matrix
if ~exist([path 'old'],'dir'); mkdir([path 'old']); end
unix(['mv ' fname_mat ' ' path 'old/'])

% create new list
for iT=Ttotal
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



function fitresult = spline(T,M_motion_t,Teval)

%% Fit: 'sct_moco_spline'.
[xData, yData] = prepareCurveData( T, M_motion_t );

% Set up fittype and options.
ft = fittype( 'smoothingspline' );
opts = fitoptions( ft );
opts.SmoothingParam = 1e-06;

% Fit model to data.
[fitresult, gof] = fit( xData, yData, ft, opts );
M_motion_t = feval(fitresult,T);




