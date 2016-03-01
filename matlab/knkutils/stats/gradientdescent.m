function [h,dc,numiters,Xmn,Xsd,Xbad,esterr,stoperr,vout] = ...
  gradientdescent(y,X,fittype,splitfrac,randomtype,stepsize,momentum,maxiters,fixediters,convergencecriterion,wantreport,vfun,manualsplit,ignorestoppingmean,wantgpu,nodc)

% function [h,dc,numiters,Xmn,Xsd,Xbad,esterr,stoperr,vout] = ...
%   gradientdescent(y,X,fittype,splitfrac,randomtype,stepsize,momentum,maxiters,fixediters,convergencecriterion,wantreport,vfun,manualsplit,ignorestoppingmean,wantgpu,nodc)
%
% <y> is the p x n data matrix.  rows indicate different data points; columns indicate different cases.
%   NaNs are okay -- on a case-by-case basis, we treat NaNs as missing data and just ignore them.
% <X> is the p x q design matrix.  rows indicate different data points; columns indicate different regressors.
%   NaNs are okay -- after z-scoring the regressors, we explicitly convert all NaNs to 0.
%   note that <X> should not include a constant regressor, as we deal with DC explicitly.
% <fittype> (optional) is
%   0 means gradient descent
%   1 means forward stagewise
%   2 means forward stepwise [find regressor max correlated with residuals, fully enter it in, and repeat]
%   3 means forward stepwise [orthogonalize all regressors wrt currently entered variables, 
%                             find orthogonalized regressor max correlated with residuals,
%                             enter regressor in and recalculate OLS fit using all entered variables]
%   default: 0.
% <splitfrac> (optional) is the fraction of data points (in (0,1)) to use as an early stopping set.
%   special case is 0 which indicates do not use early stopping.
%   note that if <manualsplit> is supplied, <splitfrac> is ignored.
%   default: 0.2.
% <randomtype> (optional) determines how we choose the stopping set.
%   1 means randomize
%   [2 X] means use X as the seed
%   [3 X P] means use X as the seed and take the Pth subset, where 1 <= P <= round(1/splitfrac).
%   note that in all of these cases, the same division into stopping set and estimation set is used for all cases.
%   note that if <manualsplit> is supplied, <randomtype> is ignored.
%   default: [2 0].
% <stepsize> (optional) is a positive number.  let S be <stepsize>*SD where SD is the std dev
%   of the data points in the estimation set.  then,
%   when <fittype> is 0, S is the scale factor for the unit-length-normalized gradient.
%   when <fittype> is 1, S is the step size.
%   when <fittype> is 2 or 3, S is ignored.
%   default: 0.001.
% <momentum> (optional) is a non-negative number indicating the scale factor for the previous gradient.
%   note that [grad] = N(N([current grad]) + [momentum]*[previous grad]), where N indicates unit-length normalization.
%   when <fittype> is 1 or 2 or 3, <momentum> is ignored.
%   default: 0.9.
% <maxiters> (optional) is a positive number indicating maximum number of training iterations.
%   if we reach <maxiters>, we simply stop the fitting process.
%   default: Inf.
% <fixediters> (optional) is 1 x 1 or 1 x n with positive integers that indicate fixed numbers
%   of training iterations to use.  if supplied, <maxiters> is ignored and we return
%   the solution after <fixediters> iterations, regardless of the error on the estimation set
%   or stopping set (if any).
%   default: [].
% <convergencecriterion> (optional) is [A B] where A is in (0,1) and B is a positive integer.
%   We stop if we see a series of max(B,round(A*[current iteration number])) iterations that do not
%   improve performance on the estimation set or that do not improve performance on the
%   stopping set (if there is one).  For example, if <convergencecriterion> is [0.1 30],
%   then after 10 iterations, we are looking for 30 iterations that do not result in improvement,
%   and after 1000 iterations, we are looking for 100 iterations that do not result in improvement.
%   default: [0.25 10].
% <wantreport> (optional) is number of iterations between status reports.
%   if +X, display in stdout.
%   if -X, display in a figure window and also output <esterr> and <stoperr>.
%     special case is -Inf in which case we just output <esterr> and <stoperr> (no figure window).
%   if 0, do not give status reports.
%   default: 0.
% <vfun> (optional) is a function that expects h (q x 1), dc (1 x 1), and xx
%   (the index of the current case) and outputs a 1 x z vector.
%   if supplied, <vout> will be calculated (see below).
%   if <nodc>, then the dc input will just be passed in as 0.
%   default: [].
% <manualsplit> (optional) is p x 1 with elements that are 0/1 indicating the data points to
%   use for the stopping set.  if this is supplied, <splitfrac> and <randomtype> are ignored,
%   and we assume that early stopping is desired.
%   default is [], which means do the normal thing (i.e. use <splitfrac> and <randomtype>).
% <ignorestoppingmean> (optional) is whether to ignore DC when evaluating stopping set error.
%   default: 0.
% <wantgpu> (optional) is whether to attempt to use the GPU.  default: 0.  we ultimately
%   use the GPU if <wantgpu> AND GPUok.m returns true AND <fittype>==0 or 1.  otherwise, we use the CPU.
%   note that when using the GPU, calculations are performed in single-precision.  but the
%   outputs of this function are always double.  due to bugs in the GPUmat implementation, in some bad
%   cases we may temporarily not use the GPU.
% <nodc> (optional) is whether to omit estimating an explicit DC for the model.
%   in this case, the mean of each channel is not subtracted.
%   default: 0.
%
% return:
%   <h> is q x n with the estimated parameters.
%     note that when <fittype> is 1 or 2 or 3, we automatically construct <h> as sparse.
%   <dc> is 1 x n with the estimated DCs.  when <nodc>, all entries in <dc> are returned as 0.
%   <numiters> is 1 x n with iteration numbers corresponding to the returned solutions
%   <Xmn> is 1 x q with the mean that was subtracted from each regressor (could be NaN if regressor is all NaNs).
%     when <nodc>, <Xmn> is returned as all zeros.
%   <Xsd> is 1 x q with the std dev divided from each regressor (could be NaN if regressor is all NaNs).
%     when <nodc>, <Xsd> is not the std dev but the scale factor that is divided from each regressor such
%     that the length of each regressor is sqrt(n) where n is the number of non-NaN entries.
%   <Xbad> is 1 x q with 0/1 indicating regressors that were set to 0 and ignored (i.e. Xsd==0 | isnan(Xsd))
%   <esterr> is 1 x v with the estimation error at each iteration.  provided only when <wantreport> is negative.
%   <stoperr> is 1 x v with the stopping error at each iteration.  provided only when <wantreport> is negative.
%   <vout> is z x n with the output given by <vfun>.  if <vfun> is [], then <vout> is returned as [].
%
% see also gradientdescent_wrapper.m.
%
% example:
% X = randn(200,10);           % make design matrix for estimation set
% h = randn(10,3);             % construct kernels
% y = X*h + 2*randn(200,3);    % simulate estimation data
% Xval = randn(100,10);        % make design matrix for validation set
% yval = Xval*h;               % simulate validation data
% X(rand(size(X))>.95) = NaN;  % corrupt design matrix
% y(rand(size(y))>.95) = NaN;  % corrupt estimation data
% [h0,dc,numiters,Xmn,Xsd,Xbad] = gradientdescent(y,X,[],[],[],[],[],[],[],[],1);
% calccod(Xval*h0+repmat(dc,[100 1]),yval,1)  % how well can we predict validation data?
% [h0ALT,dcALT,numitersALT] = gradientdescent_wrapper(y,X,[],[],[3 0 0]);  % let's try an averaging technique
% calccod(Xval*h0ALT+repmat(dcALT,[100 1]),yval,1)
%
% history:
% 2014/04/27 - fix minor bug (would have crashed)
% 2010/08/07 - add <nodc> input.

% internal note: what about predictions and p-values?
% what about elastic net?

% input
if ~exist('fittype','var') || isempty(fittype)
  fittype = 0;
end
if ~exist('splitfrac','var') || isempty(splitfrac)
  splitfrac = 0.2;
end
if ~exist('randomtype','var') || isempty(randomtype)
  randomtype = [2 0];
end
if ~exist('stepsize','var') || isempty(stepsize)
  stepsize = 0.001;
end
if ~exist('momentum','var') || isempty(momentum)
  momentum = 0.9;
end
if ~exist('maxiters','var') || isempty(maxiters)
  maxiters = Inf;
end
if ~exist('fixediters','var') || isempty(fixediters)
  fixediters = [];
end
if ~exist('convergencecriterion','var') || isempty(convergencecriterion)
  convergencecriterion = [0.25 10];
end
if ~exist('wantreport','var') || isempty(wantreport)
  wantreport = 0;
end
if ~exist('vfun','var') || isempty(vfun)
  vfun = [];
end
if ~exist('manualsplit','var') || isempty(manualsplit)
  manualsplit = [];
end
if ~exist('ignorestoppingmean','var') || isempty(ignorestoppingmean)
  ignorestoppingmean = 0;
end
if ~exist('wantgpu','var') || isempty(wantgpu)
  wantgpu = 0;
end
if ~exist('nodc','var') || isempty(nodc)
  nodc = 0;
end

% calc and deal with input
p = size(y,1);
n = size(y,2);
q = size(X,2);
if ~isempty(fixediters)
  fixediters = fillout(fixediters,[1 n]);
end

% initialize figure window
if wantreport < 0 && isfinite(wantreport)
  fig = figure;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SELECT STOPPING AND ESTIMATION SETS

% stopping set indices
if isempty(manualsplit)
  if splitfrac==0
    stoppingidx = [];
  else
    switch randomtype(1)
    case 1
      stoppingidx = picksubset(1:p,round(splitfrac*p),sum(100*clock));
    case 2
      stoppingidx = picksubset(1:p,round(splitfrac*p),randomtype(2));
    case 3
      stoppingidx = picksubset(1:p,[round(1/splitfrac) randomtype(3)],randomtype(2));
    end
  end
  stoppingidx = sort(stoppingidx);  % just to keep things tidy
  isearlystop = splitfrac > 0;
else
  stoppingidx = flatten(find(manualsplit));
  isearlystop = 1;
end

% estimation set indices
estimationidx = setdiff(1:p,stoppingidx);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PREPARE THE DESIGN AND DATA MATRICES

% divide into estimation and stopping sets
X0 = X(stoppingidx,:);
X = X(estimationidx,:);
y0 = y(stoppingidx,:);
y = y(estimationidx,:);

% z-score or normalize the length of the regressors (calculate from X, apply same transformation to X0)
indices = chunking(1:q,1000);
Xmn = zeros(1,q);
Xsd = zeros(1,q);
if nodc
  for p=1:length(indices)
    [X(:,indices{p}),d,Xsd(indices{p})] = unitlength(X(:,indices{p}),1,1,0);  % NOTE: this can blow up channels with low variance!
    if ~isempty(X0)
      X0(:,indices{p}) = unitlength(X0(:,indices{p}),1,1,0,Xsd(indices{p}));
    end
  end
else
  for p=1:length(indices)
    [X(:,indices{p}),Xmn(indices{p}),Xsd(indices{p})] = calczscore(X(:,indices{p}),1,[],[],0);  % NOTE: this can blow up channels with low variance!
    if ~isempty(X0)
      X0(:,indices{p}) = calczscore(X0(:,indices{p}),1,Xmn(indices{p}),Xsd(indices{p}),0);
    end
  end
end

% eliminate bad regressors (note that the bad regressors
% (i.e. those with no variance or those that were all NaN) become NaN 
% after the zscore operation.)
Xbad = Xsd==0 | isnan(Xsd);
X(:,Xbad) = [];
X0(:,Xbad) = [];
qt = size(X,2);  % qt means q "tight"

% get rid of any remaining NaNs.  the idea is that we don't want to let
% stray NaNs endanger entire data points.  putting in zeros means that
% these instances will not contribute positively or negatively to model fits (i.e. Xh).
X(isnan(X)) = 0;
X0(isnan(X0)) = 0;

% subtract mean of training set (if desired)
if nodc
  dc = zeros(1,n);  % just set to this
else
  dc = nanmean(y,1);
  y = bsxfun(@minus,y,dc);
  y0 = bsxfun(@minus,y0,dc);
end

% at this point, we have guaranteed that there are no NaNs in X or X0.
% however, there may be NaNs in y and y0.  we will deal with this below.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GPU stuff

% should we try to use the GPU?
% yes, if we have the right parameters AND if <wantgpu> AND if the current hostname is okay according to GPUok
gpuon = ismember(fittype,[0 1]) && wantgpu && GPUok;

% start GPU
if gpuon
  fprintf('*** attempting to start GPU.\n');
  try
    GPUstart;
    needed = numel(X)*4 + numel(X0)*4;
    memavail = GPUmem;
    fprintf('*** we are going to need at least %d bytes on the GPU (X is %s, X0 is %s).\n',needed,mat2str(size(X)),mat2str(size(X0)));
    fprintf('*** we have %d bytes available.\n',memavail);
    if memavail - needed < 200000000
      fprintf('*** this will leave us with less than 200 MB of GPU memory free, so not using GPU.\n');
      gpuon = 0;
      GPUstop;
    else
      fprintf('*** GPU successfully started.\n');
    end
  catch
    fprintf('*** there was an error in starting the GPU, so we will use the CPU.\n');
    gpuon = 0;
  end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% init
if ismember(fittype,[1 2 3])
  h = sparse(q,n);
else
  h = zeros(q,n);
end
numiters = zeros(1,n);
vout = [];
yok_old = NaN;
y0ok_old = NaN;
tempgpuoff = 0;

% loop over cases
starttime0 = clock;
for xx=1:n
  fprintf('starting %d of %d.\n',xx,n);
  starttime = clock;

  % initialize status reporting stuff
  if wantreport < 0 && isfinite(wantreport)
    figure(fig); clf;
    set(gcf,'Units','points','Position',[100 100 800 400]);
    if ~isearlystop
      subplot(1,2,1);
      xlabel('parameter');
      ylabel('magnitude');
      title(sprintf('case number %d',xx));
      subplot(1,2,2);
      xlabel('iteration');
      ylabel('estimation error');
    else
      subplot(1,3,1);
      xlabel('parameter');
      ylabel('magnitude');
      title(sprintf('case number %d',xx));
      subplot(1,3,2);
      xlabel('iteration');
      ylabel('estimation error');
      subplot(1,3,3);
      xlabel('iteration');
      ylabel('stopping error');
    end
  end

  % deal with NaNs in y and y0
  yok = ~isnan(y(:,xx));
  y0ok = ~isnan(y0(:,xx));
  
  % if any of the important dimensions are zero, we have to avoid the GPU temporarily
  if gpuon && (count(yok)==0 || count(y0ok)==0 || qt==0)
    gpuon = 0;
    tempgpuoff = 1;
  end
  
  %% PROBABLY HERE IS WHERE WE WOULD INSTAZOOM
  
  % extract out stuff.  note that there is one potential weirdness -- the zscoring
  % of regressors was performed using all of y, but we may have NaNs for some data
  % points in y, which means that the extracted subset of X may not actually
  % have zero mean and unit standard deviation.  this is probably okay.
  % note that in the non-GPU case, it may be wasteful in terms of memory to extract
  % out another copy of X.  in the GPU case, we have to allocate memory on the GPU anyway!
  y_ = GPUconv(y(yok,xx),gpuon);      % p1 x 1
  y0_ = GPUconv(y0(y0ok,xx),gpuon);   % p2 x 1
  if ~isequal(yok,yok_old)  % this kind of thing really hurts in the GPU case, so let's try to avoid it if possible
    clear X_;
    X_ = GPUconv(X(yok,:),gpuon);       % p1 x qt
  end
  if ~isequal(y0ok,y0ok_old)
    clear X0_;
    X0_ = GPUconv(X0(y0ok,:),gpuon);    % p2 x qt
  end

  % record
  yok_old = yok;
  y0ok_old = y0ok;

  % dimensions  
  dimq = size(X_,2);
  dimp1 = size(y_,1);
  dimp2 = size(y0_,1);

  % init
  esterr = [];                    % record of estimation set error
  esterr_min = Inf;               % minimum estimation error found so far
  stoperr = [];                   % record of stopping set error
  stoperr_min = Inf;              % minimum stopping error found so far
  estbadcnt = 0;                  % number of times estimation error has gone up
  stopbadcnt = 0;                 % number of times stopping error has gone up
  grad_prev = 0;                  % previous gradient (used only when fittype==0)
  h1 = []; h2 = []; h3 = [];      % plot handles
  h_ = GPUconv(zeros(qt,1),gpuon);  % kernel
  if gpuon
      % don't use syntax like zeros(dimp2,1,GPUsingle) because that breaks for zero dimensions
    grad = GPUsingle(zeros(1,dimq));  % current gradient (GPU version)
    esterr0GPU = GPUsingle(zeros(dimp1,1));  % estimation error (GPU version)
    stoperr0GPU = GPUsingle(zeros(dimp2,1));  % stopping error (GPU version)
  end
  
  % precompute
  A = (y_.'*X_).';   % qt x 1  (GPU)    [like this to avoid transposing X_]
  B = X_*h_;         % p1 x 1  (GPU)
  B0 = X0_*h_;       % p2 x 1  (GPU)
  finalstepsize = stepsize*std(GPUconv(y_,-gpuon));
  
  % report
  fprintf('*** number of data points: %d, number of parameters: %d\n',length(y_),length(h_));
  
  % do the main loop
  iter = 1;
  while 1
    
    % if not the first iteration, adjust kernel according to the gradient and recalculate B and B0
    if iter~=1
  
      % calculate gradient
      switch fittype
      case {0 1 2}
        % grad = X'(Xh-y) = X'(Xh) - X'y
        % grad = GPUconv(X_'*B - A,-gpuon);
        % THE FOLLOWING ACCOUNTS FOR A LOT OF EXECUTION TIME, SO LET'S BE REALLY UGLY
        if gpuon
          GPUmtimes(B.',X_,grad);  % [like this to avoid transposing X_]
          GPUminus(grad,A.',grad);
        else
          grad = B.'*X_ - A.';  % qt x 1   [like this to avoid transposing X_]
        end
      case 3
        % orthogonalize regressors wrt currently entered regressors and then find gradient
        grad = ((projectionmatrix(X_(:,h_~=0))*X_)'*(B-y_)).';
      end
  
      % adjust the kernel (h)
      switch fittype
      case 0
        if momentum > 0
          grad = unitlengthfast(unitlengthfast(grad) + momentum * grad_prev);
        else
          grad = unitlengthfast(grad);
        end
        h_ = h_ - finalstepsize*grad.';
        grad_prev = grad;  % record gradient for next iteration
        % THE FOLLOWING ACCOUNTS FOR A LOT OF EXECUTION TIME, SO LET'S BE REALLY UGLY
        if gpuon
          GPUmtimes(X_,h_,B);
          GPUmtimes(X0_,h_,B0);
        else
          B = X_*h_;
          B0 = X0_*h_;
        end
      case 1
        if gpuon
          ix = cublasIsamax(dimq,getPtr(abs(grad)),1);
          delta = finalstepsize*sign(grad(ix));
          h_(ix) = h_(ix) - delta;
          GPUminus(B,delta * X_(:,ix),B);  % CAN WE MAKE THIS FASTER?   (CONSIDER EXPLICIT INSTEAD OF :)
          GPUminus(B0,delta * X0_(:,ix),B0);
        else
          [d,ix] = max(abs(grad));
          delta = finalstepsize*sign(grad(ix));
          h_(ix) = h_(ix) - delta;
          B = B - delta * X_(:,ix);
          B0 = B0 - delta * X0_(:,ix);
        end
      case 2
        [d,ix] = max(abs(grad));
          % check that this regressor hasn't already been entered
        if h_(ix)~=0
          if wantreport > 0
            fprintf('halted because regressor to enter in has already been entered.\n');
          end
          break;
        end
          % fully regress X_(:,ix) onto the residuals
        tt = X_(:,ix)'*X_(:,ix);
        if tt==0  % weird, regressor has no variance
          if wantreport > 0
            fprintf('halted because regressor to enter in has no variance.\n');
          end
          break;
        end
        h_(ix) = -grad(ix)/tt;
        B = B + X_(:,ix)*h_(ix);
        B0 = B0 + X0_(:,ix)*h_(ix);
      case 3
        [d,ix] = max(abs(grad));
        h_([find(h_~=0)' ix]) = olsmatrix(X_(:,[find(h_~=0)' ix]))*y_;  % enter regressor in and recalculate OLS solution
        B = X_(:,h_~=0)*h_(h_~=0);
        B0 = X0_(:,h_~=0)*h_(h_~=0);
      end
      
    end

    % calculate and record error (ALSO TIME-CONSUMING, SO LET'S BE UGLY)
    if gpuon
      GPUminus(y_,B,esterr0GPU);
      esterr0 = esterr0GPU.'*esterr0GPU;  % scalars automatically get turned into CPU format
    else
      esterr0 = sum((y_-B).^2);
    end
    if ignorestoppingmean
      stoperr0 = sum((zeromean(GPUconv(y0_,-gpuon))-zeromean(GPUconv(B0,-gpuon))).^2);  % hack it because zeromean doesn't work on GPU
    else
      if gpuon
        GPUminus(y0_,B0,stoperr0GPU);
        stoperr0 = stoperr0GPU.'*stoperr0GPU;  % scalars automatically get turned into CPU format
      else
        stoperr0 = sum((y0_-B0).^2);  % GPU NOTE: we do not have to convert, because sum magically returns CPU format
      end
    end
    if wantreport < 0
      esterr(iter) = esterr0;  % SLOW
      stoperr(iter) = stoperr0;
    end
  
    % report
    if wantreport > 0
      if iter==1 || mod(iter,wantreport)==0
        if ~isearlystop
          fprintf('iter: %03d | est: %.3f (%s)\n',iter,esterr0,choose(esterr0 < esterr_min,'D','U'));
        else
          fprintf('iter: %03d | est: %.3f (%s) | stop: %.3f (%s)\n',iter,esterr0,choose(esterr0 < esterr_min,'D','U'), ...
            stoperr0,choose(stoperr0 < stoperr_min,'D','U'));
        end
      end
    end
    if wantreport < 0 && isfinite(wantreport)
      if iter==1 || mod(iter,wantreport)==0
        if ~isearlystop
          delete([h1 h2]);
          subplot(1,2,1); hold on;
          h1 = bar(GPUconv(h_,-gpuon));
          subplot(1,2,2); hold on;
          h2 = plot(esterr,'.-');
        else
          delete([h1 h2 h3]);
          subplot(1,3,1); hold on;
          h1 = bar(GPUconv(h_,-gpuon));
          subplot(1,3,2); hold on;
          h2 = plot(esterr,'.-');
          subplot(1,3,3); hold on;
          h3 = plot(stoperr,'.-');
        end
        drawnow;
      end
    end
  
    % do we consider this iteration to be the best yet?
    % (if fixed iter is supplied) OR (if no early stopping and estimation error is minimum so far) OR (if early stopping and stopping error is minimum so far)
    if ~isempty(fixediters) || (~isearlystop && esterr0 < esterr_min) || (isearlystop && stoperr0 < stoperr_min)
      numiters(xx) = iter;
      h_best = h_;  % (GPU)
    end
  
    % check estimation error against minimum (both cases)
    if esterr0 < esterr_min
      estbadcnt = 0;
      esterr_min = esterr0;
    else
      estbadcnt = estbadcnt + 1;
    end
  
    % check stopping error against minimum (only for early stopping)
    if isearlystop  % doing this only in early stopping case ensures stopbadcnt always remains 0 in the no-early-stopping case
      if stoperr0 < stoperr_min
        stopbadcnt = 0;
        stoperr_min = stoperr0;
      else
        stopbadcnt = stopbadcnt + 1;
      end
    end
    
    % do we stop?
    % if we are in the special fixed iters case and we have reached fixed iters, then stop.
    if ~isempty(fixediters) && iter==fixediters(xx)
      if wantreport > 0
        fprintf('halted because reached fixed iterations.\n');
      end
      break;
    end
    % if we are not in the special fixed iters case, see if the badcnt for estimation set has reached the criterion
    if isempty(fixediters) && estbadcnt == max(convergencecriterion(2),round(convergencecriterion(1)*iter))
      if wantreport > 0
        fprintf('halted because error on estimation set is not decreasing.\n');
      end
      break;
    end
    % if we are not in the special fixed iters case, see if the badcnt for stopping set has reached the criterion.
    if isempty(fixediters) && stopbadcnt == max(convergencecriterion(2),round(convergencecriterion(1)*iter))
      if wantreport > 0
        fprintf('halted because error on stopping set is not decreasing.\n');
      end
      break;
    end
    % if we are not in the special fixed iters case, see if we have reached maxiters.
    if isempty(fixediters) && iter==maxiters
      if wantreport > 0
        fprintf('halted because reached max iterations.\n');
      end
      break;
    end
  
    iter = iter + 1;
  end

  % explicitly come back to CPU
  h_best = GPUconv(h_best,-gpuon);
  
  % record the kernel and DC
  h(~Xbad,xx) = h_best./Xsd(~Xbad)';             % deal with the std-dev-division in the z-scoring process
  if ~nodc
    dc(xx) = dc(xx) - Xmn(~Xbad)*h(~Xbad,xx);      % deal with the mean-subtraction in the z-scoring process
  end
  
  % evaluate vfun
  if ~isempty(vfun)
    vout(:,xx) = feval(vfun,h(:,xx),dc(xx),xx).';
    fprintf('vout is %s\n',mat2str(vout(:,xx).',5));
    if xx==1
      vout = placematrix(zeros(size(vout,1),n),vout,[1 1]);  % quasi-dynamically resize
    end
  end

  % deal with temporary gpu off case
  if tempgpuoff
    tempgpuoff = 0;
    gpuon = 1;
    yok_old = NaN;  % start afresh to avoid any stupid caching problems
    y0ok_old = NaN;
  end

  % report
  fprintf('execution time: %d seconds (%d iterations; optimal iterations was %d)\n',round(etime(clock,starttime)),iter,numiters(xx));
end

% HRM, THIS MAY CAUSE PROBLEMS...
% % stop GPU
% if gpuon
%   fprintf('*** attempting to stop GPU.\n');
%   try
%     GPUstop;
%     fprintf('*** GPU successfully stopped.\n');
%   catch
%     fprintf('*** there was an error in stopping the GPU.\n');
%   end
% end

% final report
fprintf('TOTAL execution time: %d seconds\n',round(etime(clock,starttime0)));

% make sure these memory hogs are gone!
clear X_ X0_;
