function fitnonlinearmodel_consolidate(dir0)

% function fitnonlinearmodel_consolidate(dir0)
%
% <dir0> is a directory containing results from fitnonlinearmodel.m
%
% load in all of the *.mat files in <dir0> and write 
% the consolidated results to <dir0>.mat.
%
% we assume that the full set of voxels have been analyzed
% (in one or more chunks).  only the primary outputs of
% fitnonlinearmodel.m are saved to the new file; the auxiliary
% outputs (that exist in individual .mat files) are not.
%
% note that we check to make sure that the total number of
% voxels that are found in the .mat files is equal to the total
% number of voxels specified in the original call to fitnonlinearmodel.m.
% this ensures that all voxels have been analyzed!
%
% example:
%
% % first, set up the problem
% x = randn(100,1);
% y = bsxfun(@plus,2*x + 3,randn(100,20));
% opt = struct( ...
%   'outputdir','test', ...
%   'stimulus',[x ones(100,1)], ...
%   'data',y, ...
%   'model',{{[1 1] [-Inf -Inf; Inf Inf] @(pp,dd) dd*pp'}});
%
% % next, do the fitting in chunks of 2
% for p=1:10
%   fitnonlinearmodel(opt,2,p);
% end
%
% % then, consolidate the results
% fitnonlinearmodel_consolidate('test');
%
% % check the output
% a = load('test.mat');
% a

% consolidate
file0 = consolidatematdir(dir0,{'opt'});  % 'opt' may be big. let's specifically exclude it.

% load
a = load(file0);

% what is the total number of voxels (so we can check)
totalnumvxs = a.results(1).totalnumvxs;

% assign to b, consolidating as we go
clear b;
varlist = {'params' 'trainperformance' 'testperformance' 'aggregatedtestperformance' 'testdata' 'modelpred' 'modelfit' 'numiters' 'resnorms'};
dimlist = [3 2 2 2 2 2 3 2 2];
for zz=1:length(varlist)
  if isfield(a.results(1),varlist{zz});
     
    % cat the results
    temp = cat(dimlist(zz),a.results.(varlist{zz}));

    % check if we have the full set of results.  note that sometimes outputs can be empty.
    assert(isempty(temp) || size(temp,dimlist(zz)) == totalnumvxs, ...
           'we did not find the full set of results!');
    
    % record
    b.(varlist{zz}) = temp;

  end
end

% save
save(file0,'-struct','b');
