function f = subspaceapprox(X,y)

% function f = subspaceapprox(X,y)
%
% <X> is samples x basis functions
% <y> is samples x 1
%
% calculate cod correlation between subspace <X> and vector <y>.  we do this by
% performing ordinary least-squares regression of <X> onto <y> and then
% calculating the R^2 of the fit and <y> (no mean subtraction).
%
% example:
% subspaceapprox(randn(100,99),randn(100,1)) > .99
%
% history:
% 2010/06/15 - switch to calccod.m.

f = calccod(X*(inv(X'*X)*(X'*y)),y,[],[],0);
