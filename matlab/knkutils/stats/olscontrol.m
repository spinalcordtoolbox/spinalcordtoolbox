function [h,fit,residual] = olscontrol(regressors,data,nuisances)

% function [h,fit,residual] = olscontrol(regressors,data,nuisances)
%
% <regressors> is samples x regressors
% <data> is samples x 1
% <nuisances> is samples x nuisances
%
% fit <regressors> to <data> but control for <nuisances> (i.e. act as if
% <nuisances> were a part of the model).
% return:
%  <h> is regressors x 1 with the parameter estimates
%  <fit> is samples x 1 with the model fit that is due to <regressors>
%  <residual> is samples x 1 with <data> minus <fit>
%
% example:
% x = (0:.01:5)';
% regressors = sin(x);
% data = sin(x) + 0.4*x + 0.1*randn(size(x));
% nuisances = constructpolynomialmatrix(length(x),0:2);
% [h,fit,residual] = olscontrol(regressors,data,nuisances);
% figure; hold on;
% plot(x,data,'r-');
% plot(x,fit,'g-');
% plot(x,residual,'m-');

temp = projectionmatrix(nuisances);
h = olsmatrix(temp*regressors)*(temp*data);
fit = regressors*h;
residual = data - fit;
