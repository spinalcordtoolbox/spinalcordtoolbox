% =========================================================================
% estime_frh_param.m
% 13/12/2005
% Guillaume Marrelec
%
% Estimation de l'intensité de la réponse hémodynamique par maximum de
% vraisemblance sur un modÚle paramétrique (réponse canonique SPM)
%
% INPUT
% t             time samples
% stimulus      onsets
% donnees       data to estimate
% degre         max degree of the polynome used for the derive
% nb_freq       max frequency of the sinus used for the derive
% nderivees     number of derivatives to use
% OUTPUT
% degh          HRF degree of freedom
% mh            HRF mean
% vh            HRF variance
% degs2         degree of freedom for the noise
% vs2           scale matrix for the noise
% degl          derive degree of freedom
% ml            derive mean
% Vl            derive variance
% y_frh         HRF estimated
% y_der         derive signal
% corr          correlation between the derived used and the protocol (onsets convoluated with the canonical HRF)
% D             concatenation of the 2 derive forms (polynom + sinus)
% =========================================================================
function [degh,mh,vh,degs2,vs2,degl,ml,y_frh,y_der,corr,D]=estime_frh_param(t,xh,donnees,varargin)


% ------------------
% --- Parametres ---
% ------------------

% initializations
if (nargin<3), help estime_frh_param;, return; end
if (nargin<4), argin=[]; else, argin=varargin{1}; end

if isfield(argin,'degre'), degre=argin.degre; else degre=0; end
if isfield(argin,'nb_freq'), nb_freq=argin.nb_freq; else nb_freq=5; end
if isfield(argin,'nderivees'), nderivees=argin.nderivees; else nderivees=0; end
if isfield(argin,'threshold_drift'), threshold_drift=argin.threshold_drift; else threshold_drift=1; end
if isfield(argin,'prevent_value'), prevent_value=argin.prevent_value; else prevent_value=5; end
if isfield(argin,'Dphysio'), Dphysio=argin.Dphysio; else Dphysio=[]; end
if isfield(argin,'perform_variance'), perform_variance=argin.perform_variance; else perform_variance=true; end
if isfield(argin,'start_correlation'), start_correlation=argin.start_correlation; else start_correlation=2; end

TR=t(2)-t(1);
nb_samples=length(t);

% --- Matrice de dessin experimental ---
% On convolue la frh initiale
% frh=spm_hrf(t(2)-t(1));
% b=GaussianBasis(t(2)-t(1));
% for index = 1:size(stimulus,2)
%     TX(:,index)=conv(stimulus(:,index),b(:,1));
% end
% % On ajoute le nombre de derivees demandees
% tmp=frh;
% n=size(stimulus,2);
% for ider=1:nderivees
%     dfrh=[diff(tmp);0];
%     dfrh=dfrh/(t(2)-t(1));
%     for index = 1:n
%         TX(:,ider*n+index)=conv(stimulus(:,index),b(:,ider+1));
%     end
%     tmp=dfrh;
% end
% On convolue la frh initiale
% frh=spm_hrf(t(2)-t(1));
% for index = 1:size(stimulus,2)
%     TX(:,index)=conv(stimulus(:,index),frh);
% end
% % On ajoute le nombre de derivees demandees
% tmp=frh;
% n=size(stimulus,2);
% for ider=1:nderivees
%     dfrh=[diff(tmp);0];
%     dfrh=dfrh/(t(2)-t(1));
%     if(ider==2)
%         dfrh=spm_scaled_hrf(t(2)-t(1));
%     end
%     for index = 1:n
%         TX(:,ider*n+index)=conv(stimulus(:,index),dfrh);
%     end
%     tmp=dfrh;
% end
% % xh equals the onsets convoluted with the canonical HRF
% xh=TX(1:size(donnees,1),:);    
% clear('TX','stimulus');
% % xh=stimulus;

% --- Matrice des derives ---
% composition des derives polynomiales et sinusoidales
D = create_base(xh,nb_samples,t,degre,nb_freq,Dphysio);

% correlation between paradigm and drifts
correlated_drifts = [];
for iX=1:size(xh,2)
    % normalize paradigm signal for correlation calculation
    coeff = sum(xh(:,iX).^2)/length(xh(:,iX));
    X_norm = xh(:,iX)/sqrt(coeff);
    % perform correlation between paradigm and drifts
    corr(1,iX) = zeros;
    for index=start_correlation:size(D,2)
        tmp2 = corrcoef(X_norm,D(:,index));
        corr(index,iX) = tmp2(1,2);
        % check correlation coefficients upon an arbitrary threshold
        if (abs(corr(index,iX)) > threshold_drift)
            correlated_drifts = cat(1,correlated_drifts,index);
        end 
    end
    clear X_norm;
end

% if specified, remove drifts that correlate with the paradigm
if ~isempty(correlated_drifts)
    j=1;
    for i=1:size(D,2)
        if i~=correlated_drifts(:)
            Dtemp(:,j)=D(:,i);
            j=j+1;
        end
    end
    clear D;
    D=Dtemp;
    clear Dtemp;
end

% orthonormalize xh+D basis
% XD = cat(2,xh,D(:,3:end));
% Dconst = D(:,1);
% Dcos1 = D(:,2);
% XDortho = orthonormalise(XD);
% clear D XD;
% D = cat(2,Dconst,Dcos1,XDortho(:,2:end));
% clear XDortho;

% prevent bad conditionning for inversion of D'*D
if (size(D,2) > size(D,1)-prevent_value)
    D = D(:,1:size(D,1)-prevent_value);
end

% composition du projecteur
% J=eye(nb_samples)-D*inv(D'*D)*D'; old version that takes 2x more memory
J = zeros([nb_samples nb_samples]);
J=-D*inv(D'*D)*D';
for i=1:nb_samples
    J(i,i)=J(i,i)+1;
end
deg=size(D,2);

% -----------------
% --- Inférence --- 
% -----------------


% --- Sur la FRH h et la variance du bruit s2 ---

M=xh'*J*xh;
invM=inv(M);
yJy=donnees'*J*donnees;

% Degrés de liberté
% degh=nb_samples-(degre+1);
degh=nb_samples-deg; % because here, deg already includes the constant drift
degs2=degh;

% Moyenne de h
mh=invM*xh'*J*donnees;

clear J;

% Parametre de s2
vs2=(yJy-mh'*M*mh)/degs2;
% NB: it's the same as: vs2_jul=(yJy-donnees'*J'*xh*invM*xh'*J*donnees)/degs2
% in the paper (Marrelec03, HBM)

% Variance de h
vh=vs2*invM;
% NB: this is the "scale matrix" again in the same paper. So now we can say
% that (mh|Y) is Student-t distributed, with degs2 degree of freedom, location
% parameter mh and scale matrix vh.

% Signal de la réponse reconstruit
for index = 1:size(mh)
    y_frh(index,:)=xh(:,index)*mh(index);
end

% --- Sur les coefficients de dérive ---

% K=eye(size(XX))-XX; old version that takes 2x more memory
K = zeros([nb_samples nb_samples]);
K=-xh*inv(xh'*xh)*xh';
for i=1:nb_samples
    K(i,i)=K(i,i)+1;
end

yKy=donnees'*K*donnees;

% Degrés de liberté
degl=degh;

% Moyenne de l
ml=inv(D'*K*D)*D'*K*donnees;

% % Variance de l
% if perform_variance
%     Vl=(yKy-ml'*(D'*K*D)*ml)*inv(D'*K*D)/degl;
% else
%     Vl=0;
% end

% Reconstruction du signal de dérive
y_der=(D*ml)';


