function [f,gbrs,gaus,sds,indices,info,filters] = applymultiscalegaborfilters(images,cpfovs,bandwidths,spacings,numor,numph,thresh,scaling,mode)

% function [f,gbrs,gaus,sds,indices,info,filters] = applymultiscalegaborfilters(images,cpfovs,bandwidths,spacings,numor,numph,thresh,scaling,mode)
%
% <images> is images x pixels.  it is assumed that the images are square.
% <cpfovs> is a vector with numbers of cycles per field-of-view
% <bandwidths> is a cell vector of the same length as <cpfovs> with elements that are
%   +A where A is the number of cycles per 4 std dev of the Gaussian envelope
%   -B where B is the spatial frequency bandwidth in octave units (FWHM of amplitude spectrum)
%   [X Y] where X is like +A or -B and Y is a positive number.  the interpretation is
%     that X determines the std dev of the Gaussian along the minor axis (orthogonal to the orientation)
%     and Y is a scale factor on X that determines the std dev of the Gaussian along the major axis
%     (parallel to the orientation).
%   note that cases +A and -B imply an isotropic Gaussian envelope.
%   can be a non-cell matrix, in which case we use that for all <cpfovs> cases.
% <spacings> is a vector of the same length as <cpfovs> with the positive number of std devs
%   of the Gaussian envelope that should separate adjacent Gabors along each dimension.
%   (in the case of anisotropic Gaussian envelopes, we use the std dev along the minor axis.)
%   we calculate this number in terms of pixel units and then round down (to be conservative).
%   can be a positive scalar, in which case we use this value for all <cpfovs> cases.
%   special case is a cell vector of the same length as <cpfovs>,
%     where each element is the indices vector to use for a given case
%     (following the position conventions of filter2.m.)
%   another special case is [A B ... N] where N is a negative number -X.  this case means to use
%     X as usual for the last <cpfovs> case.  then, for the earlier <cpfovs> cases, we position
%     Gabors relative to the grid defined by the last <cpfovs> case.  specifically, suppose an
%     earlier case has a positive integer Y.  then, we position Gabors at the center of each
%     successive group of Y x Y Gabors of the last <cpfovs> case, rounding to the nearest pixel.
%     if an earlier case has a negative integer -Y, we do the same thing, except that the
%     successive groups of Y x Y Gabors are mutually exclusive.  we assume even divisibility.
% <numor> is the number of orientations (equally spaced between 0 and pi, starting at 0)
% <numph> is the number of phases, as follows:
%   +A means equally spaced between 0 and pi, starting at 0
%   -B means equally spaced between 0 and 2*pi, starting at 0
% <thresh> is a value.  points in the Gaussian envelopes less than <thresh> are set to 0,
%   and the Gabor is cropped tight to the non-zero values.  special case is -X which means do
%   the tight-cropping but do NOT impose the Gaussian envelope on the Gabor.
%   actually, to be more specific, we enforce square crops, and we use the crop from the
%   first orientation and phase combination (at each scale) for the remaining orientation and
%   phase combinations (at that scale).
% <scaling> is
%   [0 X] means put Gabor into range [-X X].  0 implicitly means [0 .5].
%     special case is [0 -X] which means empirically fit the pixels to that range.
%   [1 X1 X2 ...] means make each Gabor unit-length and then multiply by X1 X2 etc. for each <cpfovs> case.
%                 1 implicitly means [1 1 1 ...]
%   2 means scale each Gabor such that the (linear) response to the optimal grating
%           (100% contrast, with values in [-.5,.5]) is equal to 1.  thus, if a response
%           to an image is .3, the equivalent Michelson contrast is simply 30%.
% <mode> is
%   0 means dot-product [this is recommended]
%   1 means squared distance over each filter's extent, divided by number of pixels.
%     special case is -1 which means ignore mean in the X'X term when calculating distance.
%   2 means squared distance over entire image
%   [3 n sigma] means divisive normalization, as follows:
%       response = ((image*filter)+)^n / (contrast-energy^n + sigma^n)
%     where contrast-energy is sqrt of average squared deviation from mean (for pixels within extent of filter).
%     special case is [-3 n sigma] which means to calculate contrast-energy as the sqrt of average squared value of
%     the pixels within extent of the filter.  another special case is n==0, which entails:
%       response = image*filter + j*contrast-energy
%     where sigma has no effect.
%
% if <mode> is 0, compute dot-product of <images> with a set of Gabor filters.
% if <mode> is 1 or -1 or 2, compute squared distance between <images> and a set of Gabor filters.
%    the squared distance is suitable for use in radial basis functions.
% if <mode> is 3 or -3, compute divisively-normalized rectified dot-product.
% in all cases, return <f> as images x channels.  the channels are ordered according to:
%   [ph][or][position (down then right)][cpfovs]
% also, return the Gabor filters that we used in <gbrs>, a cell matrix of
%   dimensions length(<cpfovs>) x <numor> x <numph>.  note that we crop the Gabor
%   filters, so the size may differ across scales.
% also, return the Gaussian masks that we used in <gaus>, a cell matrix of
%   dimensions length(<cpfovs>) x <numor> x <numph>.  note that we crop the Gabor
%   filters, so the size may differ across scales.
% also, return the standard deviation of the Gabor filters in <sds>, a matrix
%   of dimensions length(<cpfovs>) x <numor> x <numph> x 1/2
% also, return the tiling indices in <indices>, a cell vector of dimensions 1 x length(<cpfovs>).
% also, return information about the Gabors in <info>, a matrix of dimensions 9 x channels.
%   the entries are x, y, or, 4*sd along major axis, 4*sd along minor axis, phase index, 
%                   resolution, cpfov index, orientation index
% also, return the filters in <filters>, a sparse matrix of dimensions pixels x channels.  do not
%   assign this output unless you actually need it since in some cases we do not need to compute it.
%
% Gabors are tiled according to <spacings>.  at each scale, we fit as many Gabors
% as we can along each dimension (where a Gabor "fits" if its center is within the 
% field-of-view, exclusive at both ends), and center the set of Gabors with respect
% to the image (if uneven (due to odd number of pixels), use less excess at the left/top).
% the exact same Gabor filter is used across different positions; the heavy lifting
% is performed by filter2subsample.m.
%
% note that the image is zero-padded to accommodate Gabors that extend beyond the image.
%
% note that Gabors are not forced to be zero-mean.  so, even-symmetric cases (e.g. phase==0)
% may have a small DC response.
%
% example:
% im = getsampleimage;
% f = applymultiscalegaborfilters(flatten(im),[100],-1,2,4,2,.01,1,0);
% dim = sqrt(size(f,2)/8);
% figure; imagesc(im); axis equal tight;
% figure; imagesc(makeimagestack(permute(reshape(f,[2*4 dim dim]),[2 3 1]),[],[],[2 4])); axis equal tight;
%
% *** see also applymultiscalegaussianfilters.m ***
% *** see also visualizemultiscalegaborfilters.m ***

% HM, to think about:
%  what about log-Gabors?
%  l1normalize the mask??

% constants
wantdebug = 0;

% calc
numsc = length(cpfovs);

% input
if ~iscell(bandwidths)
  bandwidths = repmat({bandwidths},[1 numsc]);
end
if ~iscell(spacings) && isscalar(spacings) && spacings > 0
  spacings = repmat(spacings,[1 numsc]);
end

% calc
res = sqrt(size(images,2)); assert(res==round(res));
ors = linspacecircular(0,pi,numor);
if numph > 0
  phs = linspacecircular(0,pi,numph);
else
  numph = -numph;
  phs = linspacecircular(0,2*pi,numph);
end

% debug mode stuff
if wantdebug
  figure; hold on; axis equal;
end

% do it
f = cast([],class(images)); gbrs = {}; gaus = {}; sds = []; indices = {}; info0 = {}; info = []; filters = sparse([]);
for p=numsc:-1:1  % START AT THE HIGH END AND WORK OUR WAY BACKWARDS
  for q=1:numor
    for r=1:numph

      % construct the Gabor.  since the Gabor might extend outside the image, we might have to construct the Gabor at a higher image resolution.
      if q==1 && r==1
        resfactor = 1;  % initialize in the first case.  subsequent cases use whatever the first case found.
      end
      while 1

        % construct the Gabor (centered wrt the image) (range is [-.5,.5])
        [gbr,gau,d,d,sd] = makegabor2d(res*resfactor,[],[],cpfovs(p)*resfactor,ors(q),phs(r),bandwidths{p});
        gbr = gbr/2;

        % find where the Gaussian mask falls below the threshold
        bad = gau < abs(thresh);
        
        % if in the first case AND
        % the Gaussian mask is valid for all rows or columns, this is bad (since we probably didn't cover the whole Gabor)
        if (q==1 && r==1) && (all(~all(bad,2)) || all(~all(bad,1)));
          resfactor = resfactor + 1;
        else
          break;
        end

      end

      % set portions of Gabor and Gaussian beyond the threshold to 0
      if thresh > 0
        gbr(bad) = 0;
        gau(bad) = 0;
      end
      
      % crop Gabor and Gaussian
      if q==1 && r==1  % figure out cropping (idxcrop) based on first case
        goodrows = find(~all(bad,2));
        goodcols = find(~all(bad,1));
        if length(goodrows) >= length(goodcols)
          idxcrop = goodrows;
        else
          idxcrop = goodcols;
        end
      end
      gbr = gbr(idxcrop,idxcrop);
      gau = gau(idxcrop,idxcrop);
      gbrsize = size(gbr,1);
      
      % undo the Gaussian mask
      if thresh < 0
        gbr = gbr ./ gau;
      end

      % scale Gabor
      switch scaling(1)
      case 0
        if length(scaling) > 1 && scaling(2)~=.5
          if scaling(2) > 0
            gbr = gbr * scaling(2)/.5;
          else
            gbr = gbr / max(abs(gbr(:))) * -scaling(2);
          end
        end
      case 1
        % make unit-length and then multiply by a constant
        gbr = unitlength(gbr);
        if length(scaling) > 1
          gbr = gbr * scaling(1+p);
        end
      case 2
      
% find optimal grating:
%         xxx = []; yyy = [];
%         projs = zeros(1,optimalph);
%         for zz=1:optimalph
%           grat = makegrating2d(gbrsize,cpfovs(p)*(gbrsize/res),ors(q),(zz-1)/optimalph * (2*pi),xxx,yyy)/2;  % range [-.5,.5]
%           projs(zz) = grat(:)'*gbr(:);
%         end
%         gbr = gbr/max(projs);

        % divide by the response to the optimal grating
        gbr = gbr/(flatten(makegrating2d(gbrsize,cpfovs(p)*(gbrsize/res),ors(q),phs(r))/2)*gbr(:));  % range of grating is [-.5,.5]
      end
      n = prod(size(gbr));
      
      % record for safe keeping
      gbrs{p,q,r} = gbr;
      gaus{p,q,r} = gau;
      sds(p,q,r,:) = sd;
      
      % at each scale, only have to compute these things once.  we have to do it here because
      % these things depend on the variables calculated earlier in this loop.
      if q==1 && r==1

        % if not a cell, we have to figure out indices of the subsampling grid
        if ~iscell(spacings)

          % the easy normal case (if the last element is positive, or if the last element is negative and this is that very last case)
          if spacings(end) > 0 || p==numsc
            sep = floor(abs(spacings(p))*sd(1));  % separated by this many pixels
            if mod(gbrsize,2)==0
              indices{p} = 1:sep:res-1;                                        % let's see how many we can fit (res-1 because the last output is an odd man out)
              indices{p} = floor(((res-1)-indices{p}(end))/2) + 1 : sep : res-1;  % evenly distribute excess (if odd, less excess at beginning)
            else
              indices{p} = 1:sep:res;                                          % let's see how many we can fit
              indices{p} = floor((res-indices{p}(end))/2) + 1 : sep : res;        % evenly distribute excess (if odd, less excess at beginning)
            end

          % the tricky special case
          else

            % get desired pixel-oriented centers
            adjustment = choose(mod(size(gbrs{numsc,1,1},1),2)==0,.5,0);  % when the original filter was even-sized, have to add .5 to get the pixel-oriented centers
            if spacings(p) > 0
              temp = colfilt(indices{numsc}+adjustment,[1 spacings(p)],'sliding',@mean);  % get centers of successive groups
              if mod(spacings(p),2)==0
                indices{p} = temp(spacings(p)/2-1+1:end-spacings(p)/2);
              else
                indices{p} = temp((spacings(p)-1)/2+1:end-(spacings(p)-1)/2);
              end
            else
              indices{p} = blkproc(indices{numsc}+adjustment,[1 -spacings(p)],@mean);  % get centers of successive groups
            end
            
            % perform rounding
            if mod(gbrsize,2)==0
              indices{p} = round(indices{p}-.5);  % if even, we have to adjust and then round
            else
              indices{p} = round(indices{p});  % if odd, we just have to round to nearest pixel
            end

          end

        % if a cell, then we have the indices already
        else
          indices{p} = spacings{p};
        end
        
        % report useful diagnostics
        fprintf('grid is %d x %d, gbrsize is %d (sd is %.10f), indices is %s\n',length(indices{p}),length(indices{p}),gbrsize,sd(1),mat2str(indices{p}));
        if wantdebug
          adjustment = choose(mod(gbrsize,2)==0,-(gbrsize/2-1),-(gbrsize-1)/2);  % see filter2subsample.m   UNSURE IF THIS IS RIGHT
          for c0=1:length(indices{p})
            for r0=1:length(indices{p})
              plotrectangle([indices{p}(r0)+adjustment-.5 indices{p}(r0)+adjustment+gbrsize-1+.5 ...
                             indices{p}(c0)+adjustment-.5 indices{p}(c0)+adjustment+gbrsize-1+.5],[getcolorchar(p) '-']);
            end
          end
        end

        % calculate image energy [posrect to deal with weird rounding cases that result in negative values]
        switch mode(1)
        case 1
          % calculate local image energy for each position in the grid
          imageenergy = posrect(squish(filter2subsample(ones(size(gbr)),permute(reshape(images.^2,[],res,res),[2 3 1]),indices{p}),2)');  % images x n*n
        case 2
          % calculate global image energy
          imageenergy = posrect(repmat(sum(images.^2,2),[1 length(indices{p})^2]));  % images x n*n
        case {-1 3}
          % calculate rms contrast-energy for each position in the grid
          imageenergy = posrect(squish(filter2subsample(ones(size(gbr)),permute(reshape(images.^2,[],res,res),[2 3 1]),indices{p}),2)');  % images x n*n
          imageenergy2 = posrect(squish(filter2subsample(ones(size(gbr)),permute(reshape(images,[],res,res),[2 3 1]),indices{p}),2)');  % images x n*n
          rmscon = sqrt((imageenergy + n*(imageenergy2/n).^2 - 2*(imageenergy2/n).*imageenergy2)/n);
        case -3
          % calculate rms contrast-energy (no mean subtraction) for each position in the grid
          imageenergy = posrect(squish(filter2subsample(ones(size(gbr)),permute(reshape(images.^2,[],res,res),[2 3 1]),indices{p}),2)');  % images x n*n
          rmscon = sqrt(imageenergy/n);
        end
        
        % init
        tempresults = zeros(numor,numph,size(images,1),length(indices{p})^2);  % or x ph x images x n*n
        tempfilters = {};

      end
      
      % figure out info0
      info0{p,q,r} = [];
      info0{p,q,r}(1,:) = flatten(repmat(indices{p} + choose(mod(gbrsize,2)==0,.5,0),[length(indices{p}) 1]));
      info0{p,q,r}(2,:) = flatten(repmat(indices{p}' + choose(mod(gbrsize,2)==0,.5,0),[1 length(indices{p})]));
      info0{p,q,r}(3,:) = ors(q);
      info0{p,q,r}(4,:) = 4*sd(end);   % +/- 2sd along major axis
      info0{p,q,r}(5,:) = 4*sd(1);   % +/- 2sd along major axis
      info0{p,q,r}(6,:) = r;
      info0{p,q,r}(7,:) = res;
      info0{p,q,r}(8,:) = p;
      info0{p,q,r}(9,:) = q;
        
      % ok, calculate filter output
      if nargout==7
        [filteroutput,filters0] = filter2subsample(gbr,permute(reshape(images,[],res,res),[2 3 1]),indices{p});
        tempfilters{q,r} = sparse(squish(filters0,2));
      else
        filteroutput = filter2subsample(gbr,permute(reshape(images,[],res,res),[2 3 1]),indices{p});
      end
      filteroutput = squish(filteroutput,2)';  % images x n*n
      
      % ok, finish up
      switch mode(1)
      case 0
        % just dot-product of image and filter
        tempresults(q,r,:,:) = filteroutput;
      case {1 2}
        % (x-y)'*(x-y) = (x'-y')*(x-y) = x'*x+y'*y-2*x'*y
        % squared distance between image and filter
        tempresults(q,r,:,:) = imageenergy + gbr(:)'*gbr(:) - 2*filteroutput;
      case -1
        % squared distance between image and filter
        tempresults(q,r,:,:) = rmscon.^2*n + gbr(:)'*gbr(:) - 2*filteroutput;
      case {3 -3}
        % div norm
        if mode(2)==0
          tempresults(q,r,:,:) = filteroutput + j*rmscon;
        else
          tempresults(q,r,:,:) = posrect(filteroutput).^mode(2) ./ (rmscon.^mode(2) + mode(3)^mode(2));
        end
      end
      if ismember(mode(1),[1 -1])
        tempresults(q,r,:,:) = tempresults(q,r,:,:)/n;
      end
      
    end
  
  end
  
  % take the temporary results, permute and reshape, and then add them to our final results
  f = [reshape(permute(tempresults,[3 2 1 4]),size(images,1),[]) f];  % NOTE: add to beginning
  info = [reshape(permute(reshape(cat(2,info0{p,:,:}),9,[],numor,numph),[1 4 3 2]),9,[]) info];  % NOTE: add to beginning
  if nargout==7
        % OLD TRY 1
        %    filters = [reshape(permute(reshape(cat(2,tempfilters{:}),res*res,[],numor,numph),[1 4 3 2]),res*res,[]) filters];  % NOTE: add to beginning
        % OLD TRY 2
        %    filters = [reshapepermute(cat(2,tempfilters{:}),{res*res [] numor numph},[1 4 3 2],{res*res []}) filters];  % NOTE: add to beginning
    tempfilters = cat(2,tempfilters{:});
    ix = flatten(permute(reshape(1:size(tempfilters,2),[],numor,numph),[3 2 1]));
    filters = [tempfilters(:,ix) filters];  % NOTE: add to beginning
  end

end
