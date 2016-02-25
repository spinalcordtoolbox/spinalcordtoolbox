function results = fitnonlinearmodel(opt,chunksize,chunknum)

% function results = fitnonlinearmodel(opt,chunksize,chunknum)
%
% <opt> is a struct with the following fields (or a .mat file with 'opt'):
%
%   *** OUTPUT DIRECTORY ***
%   <outputdir> (optional) is the directory to save results to
%
%   *** STIMULUS ***
%   <stimulus> is:
%     (1) a matrix with time points x components
%     (2) a cell vector of (1) indicating different runs
%     (3) a function that returns (1) or (2)
%
%   *** DATA ***
%   <data> is:
%     (1) a matrix with time points x voxels
%     (2) a cell vector of (1) indicating different runs
%     (3) a function that returns (1) or (2)
%     (4) a function that accepts a vector of voxel indices and returns (1) or (2)
%         corresponding to those voxels.  in this case, <vxs> must be supplied.
%   <vxs> (optional) is:
%     (1) a vector of all voxel indices that you wish to analyze.  (If you use
%         the chunking mechanism (<chunksize>, <chunknum>), then a subset of these
%         voxels are analyzed in any given function call.)  Note that we automatically
%         sort the voxel indices and ensure uniqueness.
%     (2) a .mat file with 'vxs' as (1)
%     this input matters only if <data> is of case (4).
%
%   *** MODEL ***
%   <model> is:
%     {X Y Z W} where
%       X is the initial seed (1 x P).
%       Y are the bounds (2 x P).  NaNs in the first row indicate parameters to fix.
%       Z is a function that accepts two arguments, parameters (1 x P) and 
%         stimuli (N x C), and outputs predicted responses (N x 1).
%       W (optional) is a function that transforms stimuli into a new form prior 
%         to model evaluation.
%    OR
%     {M1 M2 M3 ...} where M1 is of the form {X Y Z W} described above,
%       and the remaining Mi are of the form {F G H I} where
%       F is a function that takes fitted parameters (1 x P) from the previous model
%         and outputs an initial seed (1 x Pnew) for the current model
%       G are the bounds (2 x Pnew).  NaNs in the first row indicate parameters to fix.
%       H is a function that takes fitted parameters (1 x P) from the previous model
%         and outputs a function that accepts two arguments, parameters (1 x Pnew) and
%         stimuli (N x C), and outputs predicted responses (N x 1).
%       I (optional) is a function that takes fitted parameters (1 x P) from the 
%         previous model and outputs a function that transforms stimuli into a 
%         new form prior to model evaluation.
%    OR
%     M where M is a function that takes stimuli (N x C) and responses (N x 1) and
%       outputs an estimate of the linear weights (1 x C).  For example, simple
%       OLS regression is the case where M is @(X,y) (inv(X'*X)*X'*y)'.
%       This case is referred to as the linear-model case.
%
%   *** SEED ***
%   <seed> (optional) is:
%     (1) the initial seed (1 x P)
%     (2) several initial seeds to try (Q x P) in order to find the one that
%         produces the least error
%     (3) a function that accepts a single voxel index and returns (1) or (2).
%         in this case, <vxs> must be supplied.
%     If supplied, <seed> overrides the contents of X in <model>.
%     In the linear-model case, <seed> is not applicable and should be [].
%
%   *** OPTIMIZATION OPTIONS ***
%   <optimoptions> (optional) are optimization options in the form used by optimset.m.
%     Can also be a cell vector with option/value pairs, in which these are applied
%     after the default optimization options.  The default options are:
%       optimset('Display','iter','FunValCheck','on', ...
%                'MaxFunEvals',Inf,'MaxIter',Inf, ...
%                'TolFun',1e-6,'TolX',1e-6, ...
%                'OutputFcn',@(a,b,c) outputfcnsanitycheck(a,b,c,1e-6,10))
%     In particular, it may be useful to specify a specific optimization algorithm to use.
%     In the linear-model case, <optimoptions> is ignored.
%   <outputfcn> (optional) is a function suitable for use as an 'OutputFcn'.  If you
%     supply <outputfcn>, it will take precedence over any 'OutputFcn' in <optimoptions>.
%     The reason for <outputfcn> is that the data points being fitted will be passed as a
%     fourth argument to <outputfcn> (if <outputfcn> accepts four arguments).  This 
%     enables some useful functionality such as being able to visualize the model and
%     the data during the optimization.
%     In the linear-model case, <outputfcn> is ignored.
%
%   *** RESAMPLING SCHEMES ***
%   <wantresampleruns> (optional) is whether to resample at the level of runs (as opposed
%     to the level of individual data points).  If only one run of data is supplied, the
%     default is 0 (resample data points).  If more than one run of data is supplied, the
%     default is 1 (resample runs).
%   <resampling> (optional) is:
%     0 means fit fully (no bootstrapping nor cross-validation)
%     B or {B SEED GROUP} indicates to perform B bootstraps, using SEED as the random
%       number seed, and GROUP as the grouping to use.  GROUP should be a vector of
%       positive integers.  For example, [1 1 1 2 2 2] means to draw six bootstrap
%       samples in total, with three bootstrap samples from the first three cases and
%       three bootstrap samples from the second three cases.  If SEED is not provided,
%       the default is sum(100*clock).  If GROUP is not provided, the default is ones(1,D)
%       where D is the total number of runs or data points.
%     V where V is a matrix of dimensions (cross-validation schemes) x (runs or data
%       points).  Each row indicates a distinct cross-validation scheme, where 1 indicates
%       training, -1 indicates testing, and 0 indicates to not use.  For example, 
%       [1 1 -1 -1 0] specifies a scheme where the first two runs (or data points) are 
%       used for training and the second two runs (or data points) are used for testing.
%     Default: 0.
%
%   *** METRIC ***
%   <metric> (optional) determine how model performance is quantified.  <metric> should
%     be a function that accepts two column vectors (the first is the model; the second 
%     is the data) and outputs a number.  Default: @calccorrelation.
%
%   *** ADDITIONAL REGRESSORS ***
%   <maxpolydeg> (optional) is a non-negative integer with the maximum polynomial degree
%     to use for polynomial nuisance functions.  The polynomial nuisance functions are
%     constructed on a per-run basis.  <maxpolydeg> can be a vector to indicate different
%     degrees for different runs.  A special case is NaN which means to omit polynomials.
%     Default: NaN.
%   <wantremovepoly> (optional) is whether to project the polynomials out from both the
%     model and the data before computing <metric>.  Default: 1.
%   <extraregressors> (optional) is:
%     (1) a matrix with time points x regressors
%     (2) a cell vector of (1) indicating different runs
%     (3) a function that returns (1) or (2)
%     Note that a separate set of regressors must be supplied for each run.  The number
%     of regressors does not have to be the same across runs.
%   <wantremoveextra> (optional) is whether to project the extraregressors out from
%     both the model and the data before computing <metric>.  Default: 1.
%
%   *** OUTPUT-RELATED ***
%   <dontsave> (optional) is a string or a cell vector of strings indicating outputs
%     to omit when returning.  For example, you may want to omit 'testdata', 'modelpred',
%     'modelfit', 'numiters', and 'resnorms' since they may use a lot of memory.  
%     If [] or not supplied, then we use the default of {'modelfit' 'numiters' 'resnorms'}.
%     If {}, then we will return all outputs.  Note: <dontsave> can also refer to 
%     auxiliary variables that are saved to the .mat files when <outputdir> is used.
%   <dosave> (optional) is just like 'dontsave' except that the outputs specified here
%     are guaranteed to be returned.  (<dosave> takes precedence over <dontsave>.)
%     Default is {}.
%
% <chunksize> (optional) is the number of voxels to process in a single function call.
%   The default is to process all voxels.
% <chunknum> (optional) is the chunk number to process.  Default: 1.
%
% This function, fitnonlinearmodel.m, is essentially a wrapper around MATLAB's
% lsqcurvefit.m function for the purposes of fitting nonlinear (and linear) models
% to data.
%
% This function provides the following key benefits:
% - Deals with input and output issues (making it easy to process many individual
%   voxels and evaluate different models)
% - Deals with resampling (cross-validation and bootstrapping)
% - In the case of nonlinear models, makes it easy to evaluate multiple initial 
%   seeds (to avoid local minima)
% - In the case of nonlinear models, makes it easy to perform stepwise fitting of models
%
% Outputs:
% - 'params' is resampling cases x parameters x voxels.
%     These are the estimated parameters from each resampling scheme for each voxel.
% - 'trainperformance' is resampling cases x voxels.
%     This is the performance of the model on the training data under each resampling
%     scheme for each voxel.
% - 'testperformance' is resampling cases x voxels.
%     This is the performance of the model on the testing data under each resampling
%     scheme for each voxel.
% - 'aggregatedtestperformance' is 1 x voxels.
%     This is the performance of the model on the testing data, after aggregating
%     the data and model predictions across the resampling schemes.
% - 'testdata' is time points x voxels.
%     This is the aggregated testing data across the resampling schemes.
% - 'modelpred' is time points x voxels.
%     This is the aggregated model predictions across the resampling schemes.
% - 'modelfit' is resampling cases x time points x voxels.
%     This is the model fit for each resampling scheme.  Here, by "model fit"
%     we mean the fit for each of the original stimuli based on the parameters
%     estimated in a given resampling case; we do not mean the fit for each of the 
%     stimuli involved in the fitting.  (For example, if there are 100 stimuli and 
%     we are performing cross-validation, there will nevertheless be 100 time points
%     in 'modelfit'.)  Also, note that 'modelfit' is the raw fit; it is not adjusted
%     for <wantremovepoly> and <wantremoveextra>.
% - 'numiters' is a cell vector of dimensions 1 x voxels.  Each element is
%     is resampling cases x seeds x models.  These are the numbers of iterations
%     used in the optimizations.  Note that 'numiters' is [] in the linear-model case.
% - 'resnorms' is a cell vector of dimensions 1 x voxels.  Each element is
%     is resampling cases x seeds.  These are the residual norms obtained
%     in the optimizations.  This is useful for diagnosing multiple-seed issues.
%     Note that 'resnorms' is [] in the linear-model case.
%
% Notes:
% - Since we use %06d.mat to name output files, you should use no more than 999,999 chunks.
% - <chunksize> and <chunknum> can be strings (if so, they will be passed to str2double).
% - <stimulus> can actually have multiple frames in the third dimension.  This is handled
%   by making it such that the prediction for a given data point is calculated as the
%   average of the predicted responses for the individual stimulus frames associated with
%   that data point.
% - In the case of nonlinear models, to control the scale of the computations, in the 
%   optimization call we divide the data by its standard deviation and apply the exact 
%   same scaling to the model.  This has the effect of controlling the scale of the 
%   residuals.  This last-minute scaling should have no effect on the final parameter estimates.
%
% History:
% - 2014/05/01 - change the main loop to parfor; some cosmetic tweaks;
%                now, if no parameters are to be optimized, just return the initial seed
% - 2013/10/02 - implement the linear-model case
% - 2013/09/07 - fix bug (if polynomials or extra regressors were used in multiple runs,
%                then they were not getting fit properly).
% - 2013/09/07 - in fitnonlinearmodel_helper.m, convert to double in the call to lsqcurvefit;
%                and perform a speed-up (don't compute modelfit if unwanted)
% - 2013/09/04 - add totalnumvxs variable
% - 2013/09/03 - allow <dontsave> to refer to the auxiliary variables
% - 2013/09/02 - add 'modelfit' and adjust default for 'dontsave'; add 'dosave'
% - 2013/08/28 - new outputs 'resnorms' and 'numiters'; last-minute data scaling; 
%                tweak default handling of 'dontsave'
% - 2013/08/18 - Initial version.
%
% Example 1:
% 
% % first, a simple example
% x = randn(100,1);
% y = 2*x + 3 + randn(100,1);
% opt = struct( ...
%   'stimulus',[x ones(100,1)], ...
%   'data',y, ...
%   'model',{{[1 1] [-Inf -Inf; Inf Inf] @(pp,dd) dd*pp'}});
% results = fitnonlinearmodel(opt);
% 
% % now, try 100 bootstraps
% opt.resampling = 100;
% opt.optimoptions = {'Display' 'off'};  % turn off reporting
% results = fitnonlinearmodel(opt);
% 
% % now, try leave-one-out cross-validation
% opt.resampling = -(2*(eye(100) - 0.5));
% results = fitnonlinearmodel(opt);
% 
% Example 2:
% 
% % try a more complicated example.  we use 'outputfcn' to 
% % visualize the data and model during the optimization.
% x = (1:.1:10)';
% y = evalgaussian1d([5 1 4 0],x);
% y = y + randn(size(y));
% opt = struct( ...
%   'stimulus',x, ...
%   'data',y, ...
%   'model',{{[1 2 1 0] [repmat(-Inf,[1 4]); repmat(Inf,[1 4])] ...
%             @(pp,dd) evalgaussian1d(pp,dd)}}, ...
%   'outputfcn',@(a,b,c,d) pause2(.1) | outputfcnsanitycheck(a,b,c,1e-6,10) | outputfcnplot(a,b,c,1,d));
% results = fitnonlinearmodel(opt);
%
% Example 3:
%
% % same as the first example in Example 1, but now we use the
% % linear-model functionality
% x = randn(100,1);
% y = 2*x + 3 + randn(100,1);
% opt = struct( ...
%   'stimulus',[x ones(100,1)], ...
%   'data',y, ...
%   'model',@(X,y) (inv(X'*X)*X'*y)');
% results = fitnonlinearmodel(opt);

% internal notes:
% - replaces fitprf.m, fitprfstatic.m, fitprfmulti.m, and fitprfstaticmulti.m
% - some of the new features: opt struct format, fix projection matrix bug (must
%   compute projection matrix based on concatenated regressors), multiple initial
%   seeds are handled internally!, user must deal with model specifics like
%   the HRF and positive rectification, massive clean up of the logic (e.g.
%   runs and data points are treated as a single case), consolidation of 
%   the different functions, drop support for data trials (not worth the
%   implementation cost), drop support for NaN stimulus frames, hide the
%   myriad optimization options from the input level, drop run-separated metrics,
%   drop the stimulus transformation speed-up (it was getting implemented in a 
%   non-general way)
% - regularization is its own thing? own code module?

%%%%%%%%%%%%%%%%%%%%%%%%%%%% REPORT

fprintf('*** fitnonlinearmodel: started at %s. ***\n',datestr(now));
stime = clock;  % start time

%%%%%%%%%%%%%%%%%%%%%%%%%%%% SETUP

% deal with opt
if ischar(opt)
  opt = loadmulti(opt,'opt');
end

% is <data> of case (4)?
isvxscase = isa(opt.data,'function_handle') && nargin(opt.data) > 0;

% deal with outputdir
if ~isfield(opt,'outputdir') || isempty(opt.outputdir)
  opt.outputdir = [];
end
wantsave = ~isempty(opt.outputdir);  % should we save results to disk?

% deal with vxs
if isfield(opt,'vxs')
  if ischar(opt.vxs)
    vxsfull = loadmulti(opt.vxs,'vxs');
  else
    vxsfull = opt.vxs;
  end
  vxsfull = sort(union([],flatten(vxsfull)));
  totalnumvxs = length(vxsfull);
end

% deal with chunksize and chunknum
if ~exist('chunksize','var') || isempty(chunksize)
  chunksize = [];  % deal with this later
end
if ~exist('chunknum','var') || isempty(chunknum)
  chunknum = 1;
end
if ischar(chunksize)
  chunksize = str2double(chunksize);
end
if ischar(chunknum)
  chunknum = str2double(chunknum);
end

% deal with data (including load the data)
if isa(opt.data,'function_handle')
  fprintf('*** fitnonlinearmodel: loading data. ***\n');
  if nargin(opt.data) == 0
    data = feval(opt.data);
    if iscell(data)
      totalnumvxs = size(data{1},2);
    else
      totalnumvxs = size(data,2);
    end
  else  % note that in this case, vxs should have been specified,
        % so totalnumvxs should have already been calculated above.
    if isempty(chunksize)
      chunksize = length(vxsfull);
    end
    vxs = chunking(vxsfull,chunksize,chunknum);
    data = feval(opt.data,vxs);
  end
else
  data = opt.data;
  if iscell(data)
    totalnumvxs = size(data{1},2);
  else
    totalnumvxs = size(data,2);
  end
end
if ~iscell(data)
  data = {data};
end

% deal with chunksize
if isempty(chunksize)
  chunksize = totalnumvxs;  % default is all voxels
end

% if not isvxscase, then we may still need to do chunking
if ~isvxscase
  vxs = chunking(1:totalnumvxs,chunksize,chunknum);
  if ~isequal(vxs,1:totalnumvxs)
    for p=1:length(data)
      data{p} = data{p}(:,vxs);
    end
  end
end

% calculate the number of voxels to analyze in this function call
vnum = length(vxs);

% finally, since we have dealt with chunksize and chunknum, we can do some reporting
fprintf(['*** fitnonlinearmodel: outputdir = %s, chunksize = %d, chunknum = %d\n'], ...
  opt.outputdir,chunksize,chunknum);

% deal with model
if ~isa(opt.model,'function_handle') && ~iscell(opt.model{1})
  opt.model = {opt.model};
end
if ~isa(opt.model,'function_handle')
  for p=1:length(opt.model)
    if length(opt.model{p}) < 4 || isempty(opt.model{p}{4})
      if p==1
        opt.model{p}{4} = @identity;
      else
        opt.model{p}{4} = @(ss) @identity;
      end
    end
  end
end

% deal with seed
if ~isfield(opt,'seed') || isempty(opt.seed)
  opt.seed = [];
end

% deal with optimization options
if ~isfield(opt,'optimoptions') || isempty(opt.optimoptions)
  opt.optimoptions = {};
end
if iscell(opt.optimoptions)
  temp = optimset('Display','iter','FunValCheck','on', ...
                  'MaxFunEvals',Inf,'MaxIter',Inf, ...
                  'TolFun',1e-6,'TolX',1e-6, ...
                  'OutputFcn',@(a,b,c) outputfcnsanitycheck(a,b,c,1e-6,10));
  for p=1:length(opt.optimoptions)/2
    temp.(opt.optimoptions{(p-1)*2+1}) = opt.optimoptions{(p-1)*2+2};
  end
  opt.optimoptions = temp;
  clear temp;
end
if ~isfield(opt,'outputfcn') || isempty(opt.outputfcn)
  opt.outputfcn = [];
end

% deal with resampling schemes
if ~isfield(opt,'wantresampleruns') || isempty(opt.wantresampleruns)
  if length(data) == 1
    opt.wantresampleruns = 0;
  else
    opt.wantresampleruns = 1;
  end
end
if opt.wantresampleruns
  numdataunits = length(data);
else
  numdataunits = sum(cellfun(@(x) size(x,1),data));
end
if ~isfield(opt,'resampling') || isempty(opt.resampling)
  opt.resampling = 0;
end
if isequal(opt.resampling,0)
  resamplingmode = 'full';
elseif ~iscell(opt.resampling) && numel(opt.resampling) > 1
  resamplingmode = 'xval';
else
  resamplingmode = 'boot';
end
if isequal(resamplingmode,'boot')
  if ~iscell(opt.resampling)
    opt.resampling = {opt.resampling};
  end
  if length(opt.resampling) < 2 || isempty(opt.resampling{2})
    opt.resampling{2} = sum(100*clock);
  end
  if length(opt.resampling) < 3 || isempty(opt.resampling{3})
    opt.resampling{3} = ones(1,numdataunits);
  end
end

% deal with metric
if ~isfield(opt,'metric') || isempty(opt.metric)
  opt.metric = @calccorrelation;
end

% deal with additional regressors
if ~isfield(opt,'maxpolydeg') || isempty(opt.maxpolydeg)
  opt.maxpolydeg = NaN;
end
if length(opt.maxpolydeg) == 1
  opt.maxpolydeg = repmat(opt.maxpolydeg,[1 length(data)]);
end
if ~isfield(opt,'wantremovepoly') || isempty(opt.wantremovepoly)
  opt.wantremovepoly = 1;
end
if ~isfield(opt,'extraregressors') || isempty(opt.extraregressors)
  opt.extraregressors = [];
end
if ~isfield(opt,'wantremoveextra') || isempty(opt.wantremoveextra)
  opt.wantremoveextra = 1;
end
if ~isfield(opt,'dontsave') || (isempty(opt.dontsave) && ~iscell(opt.dontsave))
  opt.dontsave = {'modelfit' 'numiters' 'resnorms'};
end
if ~iscell(opt.dontsave)
  opt.dontsave = {opt.dontsave};
end
if ~isfield(opt,'dosave') || isempty(opt.dosave)
  opt.dosave = {};
end
if ~iscell(opt.dosave)
  opt.dosave = {opt.dosave};
end

% make outputdir if necessary
if wantsave
  mkdirquiet(opt.outputdir);
  opt.outputdir = subscript(matchfiles(opt.outputdir),1,1);
  outputfile = sprintf([opt.outputdir '/%06d.mat'],chunknum);
end

% set random number seed
if isequal(resamplingmode,'boot')
  setrandstate({opt.resampling{2}});
end

% calc
numtime = cellfun(@(x) size(x,1),data);

% save initial time
if wantsave
  saveexcept(outputfile,[{'data'} setdiff(opt.dontsave,opt.dosave)]);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%% LOAD SOME ITEMS

% deal with stimulus
if isa(opt.stimulus,'function_handle')
  fprintf('*** fitnonlinearmodel: loading stimulus. ***\n');
  stimulus = feval(opt.stimulus);
else
  stimulus = opt.stimulus;
end
if ~iscell(stimulus)
  stimulus = {stimulus};
end
stimulus = cellfun(@full,stimulus,'UniformOutput',0);

% deal with extraregressors
if isa(opt.extraregressors,'function_handle')
  fprintf('*** fitnonlinearmodel: loading extra regressors. ***\n');
  extraregressors = feval(opt.extraregressors);
else
  extraregressors = opt.extraregressors;
end
if isempty(extraregressors)
  extraregressors = repmat({[]},[1 length(data)]);
end
if ~iscell(extraregressors)
  extraregressors = {extraregressors};
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%% PRECOMPUTE SOME STUFF

% construct polynomial regressors matrix
polyregressors = {};
for p=1:length(data)
  if isnan(opt.maxpolydeg(p))
    polyregressors{p} = zeros(numtime(p),0);
  else
    polyregressors{p} = constructpolynomialmatrix(numtime(p),0:opt.maxpolydeg(p));
  end
end

% construct total regressors matrix for fitting purposes
% (i.e. both polynomials and extra regressors)
  % first, construct the run-wise regressors
tmatrix = {};
for p=1:length(data)
  tmatrix{p} = cat(2,polyregressors{p},extraregressors{p});
end
  % then, separate them using blkdiag
temp = blkdiag(tmatrix{:});
cnt = 0;
for p=1:length(data)
  tmatrix{p} = temp(cnt+(1:size(tmatrix{p},1)),:);
  cnt = cnt + size(tmatrix{p},1);
end
clear temp;

% construct special regressors matrix for the purposes of the <metric>
  % first, construct the run-wise regressors
smatrix = {};
for p=1:length(data)
  temp = [];
  if opt.wantremovepoly
    temp = cat(2,temp,polyregressors{p});
  end
  if opt.wantremoveextra
    temp = cat(2,temp,extraregressors{p});
  end
  smatrix{p} = temp;
end
  % then, separate them using blkdiag
temp = blkdiag(smatrix{:});
cnt = 0;
for p=1:length(data)
  smatrix{p} = temp(cnt+(1:size(smatrix{p},1)),:);
  cnt = cnt + size(smatrix{p},1);
end
clear temp;

% figure out trainfun and testfun for resampling
switch resamplingmode
case 'full'
  trainfun = {@(x) catcell(1,x)};
  testfun =  {@(x) []};
case 'xval'
  trainfun = {};
  testfun =  {};
  for p=1:size(opt.resampling,1)
    trainix = find(opt.resampling(p,:) == 1);
    testix =  find(opt.resampling(p,:) == -1);
    if opt.wantresampleruns
      trainfun{p} = @(x) catcell(1,x(trainix));
      testfun{p} =  @(x) catcell(1,x(testix));
    else
      trainfun{p} = @(x) subscript(catcell(1,x),{trainix ':' ':' ':' ':' ':'});  % HACKY
      testfun{p}  = @(x) subscript(catcell(1,x),{testix ':' ':' ':' ':' ':'});
    end
  end
case 'boot'
  trainfun = {};
  testfun = {};
  for p=1:opt.resampling{1}
    trainix = [];
    for b=1:max(opt.resampling{3})
      temp = opt.resampling{3}==b;
      trainix = [trainix subscript(find(temp),ceil(rand(1,sum(temp))*sum(temp)))];
    end
    testix = [];
    if opt.wantresampleruns
      trainfun{p} = @(x) catcell(1,x(trainix));
      testfun{p} =  @(x) catcell(1,x(testix));
    else
      trainfun{p} = @(x) subscript(catcell(1,x),{trainix ':' ':' ':' ':' ':'});  % HACKY
      testfun{p}  = @(x) subscript(catcell(1,x),{testix ':' ':' ':' ':' ':'});
    end
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%% PERFORM THE FITTING

% loop over voxels
clear results0;
parfor p=1:vnum

  % report
  fprintf('*** fitnonlinearmodel: processing voxel %d (%d of %d). ***\n',vxs(p),p,vnum);
  vtime = clock;  % start time for current voxel

  % get data and hack it in
  opt2 = opt;
  opt2.data = cellfun(@(x) x(:,p),data,'UniformOutput',0);

  % get seed and hack it in
  if ~isempty(opt2.seed)
    assert(~isa(opt2.model,'function_handle'));  % sanity check
    if isa(opt2.seed,'function_handle')
      seed0 = feval(opt2.seed,vxs(p));
    else
      seed0 = opt2.seed;
    end
    opt2.model{1}{1} = seed0;
  end
  
  % call helper function to do the actual work
  results0(p) = fitnonlinearmodel_helper(opt2,stimulus,tmatrix,smatrix,trainfun,testfun);

  % report
  fprintf('*** fitnonlinearmodel: voxel %d (%d of %d) took %.1f seconds. ***\n', ...
          vxs(p),p,vnum,etime(clock,vtime));

end

% consolidate results
results = struct;
results.params = cat(3,results0.params);
results.testdata = cat(2,results0.testdata);
results.modelpred = cat(2,results0.modelpred);
results.modelfit = cat(3,results0.modelfit);
results.trainperformance = cat(1,results0.trainperformance).';
results.testperformance  = cat(1,results0.testperformance).';
results.aggregatedtestperformance = cat(2,results0.aggregatedtestperformance);
results.numiters = cat(2,{results0.numiters});
results.resnorms = cat(2,{results0.resnorms});

% kill unwanted outputs
for p=1:length(opt.dontsave)

  % if member of dosave, keep it!
  if ismember(opt.dontsave{p},opt.dosave)

  % if not, then kill it (if it exists)!
  else
    if isfield(results,opt.dontsave{p})
      results = rmfield(results,opt.dontsave{p});
    end
  end

end

% save results
if wantsave
  save(outputfile,'-struct','results','-append');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%% REPORT

fprintf('*** fitnonlinearmodel: ended at %s (%.1f minutes). ***\n', ...
        datestr(now),etime(clock,stime)/60);
