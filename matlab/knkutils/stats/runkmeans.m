function [idx,c,err] = runkmeans(data,k,disttype,seed,maxiters,iterfun)

% function [idx,c,err] = runkmeans(data,k,disttype,seed,maxiters,iterfun)
%
% <data> is points x dimensions
% <k> is a positive integer
% <disttype> (optional) is
%   'sqdist' means squared Euclidean distance
%   'cosine' means one minus cosine of angle between vectors
%   default: 'sqdist'
% <seed> (optional) is k x dimensions with the seed to use
% <maxiters> (optional) is max number of iterations.  default: Inf.
%   special case is -Y which means stop when the error decrease is less
%   than Y percent.
% <iterfun> (optional) is a function to call at each iteration.
%   we automatically tack on "drawnow; pause;" after the call.
%   this function should accept <data>, <idx>, <c>, and <err>.
%   if not supplied, do nothing special.
%   special case is -1, which means to use a default function that
%   provides some visualization.  (this function assumes dimensions
%   can be interpreted as a square matrix.)
%   special case is -2, which means to use a default function that
%   provides some visualization.  (this function assumes that there are
%   two dimensions.)
%
% run k-means. return <idx> as points x 1 with positive integers
% indicating cluster assignment. return <c> as k x dimensions
% with the centers. return <err> as 1 x iterations with the errors
% (mean distance from points to their nearest centers). note that if
% <disttype> is 'sqdist', the errors have a constant offset and are
% therefore not accurate in an absolute sense; the constant offset 
% depends only on <data> so if <data> is constant, you can make 
% comparisons between different error values.
%
% note that we initialize centers to randomly selected
% points if <seed> is not supplied.
%
% example:
% data = [randn(1000,2); randn(1000,2)+3];
% [idx,c,err] = runkmeans(data,2);
% figure; hold on;
% scatter(data(:,1),data(:,2),'r.');
% scatter(c(:,1),c(:,2),'bo');
% figure; plot(err);

% CONSIDER merging here xval meancorr?  see fitrectdensity.m
%  but it seems xval always goes up!

% inputs
if ~exist('disttype','var') || isempty(disttype)
  disttype = 'sqdist';
end
if ~exist('seed','var') || isempty(seed)
  seed = [];
end
if ~exist('maxiters','var') || isempty(maxiters)
  maxiters = Inf;
end
if ~exist('iterfun','var') || isempty(iterfun)
  iterfun = [];
end

% calc
p = size(data,1);
d = size(data,2);
if isequal(disttype,'cosine')
  dataunit = unitlengthfast(data,2);
end

% check
assert(k<=p,'k must be less than or equal to number of points');

% initialize centers
if isempty(seed)
  perm = randperm(p);
  c = data(perm(1:k),:);  % randomly selected points
else
  c = seed;
end

% report
fprintf('runkmeans:\n');

% do it (batch stage only)
idx_old = NaN;
err = [];
iter = 1;
while 1

  % assign points to centers
  switch disttype
  case 'sqdist'
    % calculate squared Euclidean distances.  note that (x-y)'*(x-y) is x'x + y'y - 2x'y
    dists = repmat(sum(c.^2,2)',[p 1]) - 2*data*c';  % for speed, this omits repmat(sum(data.^2,2),[1 k]) since we don't care about absolute values
  case 'cosine'
    dists = 1 - dataunit*unitlengthfast(c,2)';
  end
  [mn,idx] = min(dists,[],2);  % dists is points x centers
  err(iter) = mean(mn);
  
  % report
  fprintf('iter %03d | error %.5f\n',iter,err(iter));

  % call iterfun
  if ~isempty(iterfun)
    if isequal(iterfun,-1)
      feval(@visfun,data,idx,c,err);
    elseif isequal(iterfun,-2)
      feval(@visfun2,data,idx,c,err);
    else
      feval(iterfun,data,idx,c,err); drawnow; pause;
    end
  end

  % if no assignment changes or we have reached maxiters, we're done
  if isequal(idx,idx_old) || iter==maxiters || (maxiters < 0 && iter >= 2 && (err(iter-1)-err(iter)) / err(iter-1) * 100 < -maxiters)
    break;
  end
  idx_old = idx;

  % re-calc centers
  isdone = 0;
  c = zeros(k,d);
  while ~isdone
  
    for zz=1:k
    
      % if at least one point is in the center, then we're okay
      if sum(idx==zz) > 0
        c(zz,:) = mean(data(idx==zz,1:d),1);  % 1:d instead of : vastly speeds things up!!!

      % otherwise, assign a random point to the center, and re-do
      else
        idx(ceil(rand*p)) = zz;
        break;
      end
      
      % are we done?
      if zz==k
        isdone = 1;
      end

    end

  end
  
  % increment
  iter = iter + 1;

end

%%%%%

function visfun(data,idx,c,err)

dim = sqrt(size(c,2));

if length(err)==1
  figure(1); clf;
  figure(2); clf;
  figure(3); clf;
end  

figure(1); imagesc(makeimagestack(permute(reshape(c,[],dim,dim),[2 3 1]),-1),[0 1]); axis equal tight; title('centers');
figure(2); plot(err,'ro-'); xlabel('iteration'); ylabel('error');
figure(3); mx = max(countinstances(idx)); imagesc(makeimagestack(repmat(reshape(countinstances(idx),1,1,[]),[dim dim]),-2),[0 1]);
  axis equal tight; cb = colorbar; set(cb,'YTick',round(linspace(0,mx,10))/mx,'YTickLabel',round(linspace(0,mx,10))); title('density');
drawnow; pause;

%%%%%

function visfun2(data,idx,c,err)

if length(err)==1
  figure(1); clf; hold on; scatter(data(:,1),data(:,2),'r.');
  figure(2); clf;
end

figure(1); for p=1:size(c,1), text(c(p,1),c(p,2),num2str(p));, end
figure(2); plot(err,'ro-'); xlabel('iteration'); ylabel('error');
drawnow; pause;
