function f = maskmultiscalegaborfilters(f,gaus,indices,mask,frac)

% function f = maskmultiscalegaborfilters(f,gaus,indices,mask,frac)
%
% <f>,<gaus>,<indices> are from the output of applymultiscalegaborfilters.m
% <mask> is 1 x pixels with the binary mask (0 means invalid; 1 means valid)
% <frac> is the fraction in [0,1] below which we ignore a channel
%
% zero-out all channels in <f> whose spatial footprints are less than <frac>
% within <mask>.  the spatial footprint of a channel is defined to be
% a binary mask where 1 indicates pixels that are non-zero in the Gaussian
% envelope associated with the channel.
% 
% example:
% [f,gbrs,gaus,sds,indices,info,filters] = applymultiscalegaborfilters(randn(1,1024),8,-1,2,1,1,.01,1,0);
% mask = zeros(32,32);
% mask(1:16,1:16) = 1;
% f2 = maskmultiscalegaborfilters(f,gaus,indices,mask(:)',0.5);
% figure; imagesc(reshape(f,[8 8]));
% figure; imagesc(reshape(f2,[8 8]));

% calc
numsc = size(gaus,1);
numor = size(gaus,2);
numph = size(gaus,3);
res = sqrt(length(mask)); assert(isint(res));

% do it
offset = 0;
for p=1:numsc

  % construct the spatial footprint
  foot = double(gaus{p,1,1}~=0);

  % figure out which positions have less than frac within the mask
  bad = squish(filter2subsample(foot,reshape(double(mask),res,res),indices{p}),2)' < sum(foot(:)) * frac;  % 1 x n*n
  
  % zero-out bad channels
  ix = offset + (1:numph*numor*length(indices{p})^2);  % which channels to work on
  temp = reshape(f(:,ix),size(f,1),numph*numor,[]);    % extract channels
  temp(:,:,bad) = 0;                                   % zero-out bad channels
  f(:,ix) = reshape(temp,size(temp,1),[]);             % put them back in

  % increment
  offset = ix(end);

end
