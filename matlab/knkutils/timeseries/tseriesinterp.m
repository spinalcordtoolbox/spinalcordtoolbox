function m = tseriesinterp(m,trorig,trnew,dim,numsamples)

% function m = tseriesinterp(m,trorig,trnew,dim,numsamples)
%
% <m> is a matrix with time-series data along some dimension.
%   can also be a cell vector of things like that.
% <trorig> is the sampling time of <m> (e.g. 1 second)
% <trnew> is the new desired sampling time
% <dim> (optional) is the dimension of <m> with time-series data.
%   default to 2 if <m> is a row vector and to 1 otherwise.
% <numsamples> (optional) is the number of desired samples.
%   default to the number of samples that makes the duration of the new
%   data match or minimally exceed the duration of the original data.
%
% use interp1 to cubic-interpolate <m> (with extrapolation) such that
% the new version of <m> coincides with the original version of <m>
% at the first time point.
%
% example:
% x0 = 0:.1:10;
% y0 = sin(x0);
% y1 = tseriesinterp(y0,.1,.23);
% figure; hold on;
% plot(x0,y0,'r.-');
% plot(0:.23:.23*(length(y1)-1),y1,'go');

% internal constants
numchunks = 20;

% input
if ~exist('dim','var') || isempty(dim)
  dim = choose(isrowvector(m),2,1);
end
if ~exist('numsamples','var') || isempty(numsamples)
  numsamples = [];
end

% prep
if iscell(m)
  leaveascell = 1;
else
  leaveascell = 0;
  m = {m};
end

% do it
for p=1:length(m)

  % prep 2D
  msize = size(m{p});
  m{p} = reshape2D(m{p},dim);

  % calc
  if isempty(numsamples)
    numsamples = ceil((size(m{p},1)*trorig)/trnew);
  end

  % do it
  timeorig = linspacefixeddiff(0,trorig,size(m{p},1));
  timenew  = linspacefixeddiff(0,trnew,numsamples);

  % do in chunks
  chunks = chunking(1:size(m{p},2),ceil(size(m{p},2)/numchunks));
  temp = {};
  mtemp = m{p};
  parfor q=1:length(chunks)
    temp{q} = interp1(timeorig,mtemp(:,chunks{q}),timenew,'pchip','extrap');
  end
  m{p} = catcell(2,temp);
  clear temp mtemp;

  % prepare output
  msize(dim) = numsamples;
  m{p} = reshape2D_undo(m{p},dim,msize);

end

% prepare output
if ~leaveascell
  m = m{1};
end
