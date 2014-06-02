function [activation_chi2,activation_t, residu] = j_glm(data, regressors, mask)

% calculation of activation maps by linear regression
%
% [activation_chi2,activation_t, residu] = st_do_regression(data, mask, regressors)
%
% INPUTS
% data      	4D (3D+t) dataset
% regressors    matrix containing regressors in column
% (mask)      	mask of interest
%
% OUTPUTS
% activation_chi2   activation map (values distributed as a chi2-distribution) testing H0 of all the regressors except the last one.
% activation_t      activation map (values distributed as a Student t-distribution) testing H0 of the first regressor.
% residu            3D+t dataset containing the residu of the regression.
%
% COMMENTS
% Julien Cohen-Adad 2007-05-27
% based on st_do_regression, Vincent Perlbarg 02/07/05

tic
fprintf('regression in progress... ');
warning off

% Extracting data and regressors

[nx ny nz nt] = size(data);

if exist('mask')
    if mask == Inf
        nvox = nx*ny*nz;
    else
        nvox = sum(mask(:));
    end

    data_v = reshape(data,[nx*ny*nz,nt]);
    clear data
    if mask == Inf
        Y = reshape(data_v,[nvox,nt]);
    else
        mask=logical(mask);
        Y = reshape(data_v(mask(:),:),[nvox,nt]);
    end
    clear data_v
else
    Y = reshape(data,[nx*ny*nz,nt]);
end


Y = Y';
X = regressors;
clear regressors

% - betas calculation
beta = pinv(X'*X)*X'*Y;

% - residus calculation
res = Y - X*beta;

% - variance of the residus
var_res = sum(res.^2,1)/(nt - size(X,2));

% - variance of the betas
invXpX = pinv(X' * X);
diago = diag(invXpX)';
var_beta =  diago' * var_res;

% chi2 activation map
activation_chi2 = zeros([nx*ny*nz 1]);
if mask == Inf
    activation_chi2 = carte(beta, var_beta);
else
    activation_chi2(mask(:)) = carte(beta, var_beta);
end
activation_chi2 = reshape(activation_chi2,[nx ny nz]);
activation_chi2(isnan(activation_chi2))=0;

% T activation map
activation_t = zeros([nx*ny*nz 1]);
if mask == Inf
    activation_t = (beta(1,:)./sqrt(var_beta(1,:)));
else
    activation_t(mask(:)) = (beta(1,:)./sqrt(var_beta(1,:)));
end
activation_t = reshape(activation_t,[nx ny nz]);
activation_t(isnan(activation_t))=0;

if nargout==3
	residu = zeros([nx*ny*nz nt]);
	if mask == Inf
        residu = res';
	else
        residu(mask(:),:) = res';
	end
	residu = reshape(residu,[nx ny nz nt]);
end
fprintf(' elapsed time %s \n',num2str(toc));
warning on

function activation = carte(beta, var_beta)
nb_regresseurs = size(beta,1);
% carte d'activation
%activation = beta(1,:).^2 ./ var_beta(1,:) + beta(2,:).^2 ./ var_beta(2,:);
activation = sum(beta(1:nb_regresseurs-1,:).^2./abs(var_beta(1:nb_regresseurs-1,:)),1);
%activation = activation .* sign(-beta(1,:));
%activation = activation .* sign(beta(1,:));