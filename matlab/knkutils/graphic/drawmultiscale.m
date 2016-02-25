function im = drawmultiscale(res,numdistinct,szfactors,density,pinkcon,images,masks)

% function im = drawmultiscale(res,numdistinct,szfactors,density,pinkcon,images,masks)
%
% <res> is the desired number of pixels along one side of the final image
% <numdistinct> is the number of distinct images to generate
% <szfactors> is a vector of multiplicative factors for varying the object size.
%   for example, [2 1 0.5] means that there are three scales; the first scale
%   involves making the objects twice as large, the second scale involves leaving
%   the native resolution of the objects intact, and the third scale involves
%   halving the resolution of the objects.
% <density> is a multiplicative factor for object density (along one dimension).
%   for example, going from <density>==1 to <density>==2 means that there will 
%   be four times as many objects placed.  we automatically adjust the object
%   density according to the <szfactors>.  for instance, the number of objects
%   for a size factor of 1 will be four times as large as the number of objects
%   for a size factor of 2.  the point of <density> is a do a global post-hoc
%   adjustment to control the overall object density (for all scales).  play
%   with <density> until you achieve a density that you like.
% <pinkcon> is the desired RMS contrast for the pink-noise background
% <images> is N x N x 3 x Z where N x N is the object image resolution, 3 indicates the
%   RGB channels, and Z indicates distinct object images.  the values should be in
%   the range [0,1].
% <masks> is N x N x Z with the object masks with values in [0,1].  0 indicates the
%   object and 1 indicates the background.
%
% place objects at multiple scales on a pink-noise background.  return <im>, which
% has dimensions <res> x <res> x 3 x <numdistinct> and has values in [0,1].
%
% the pink-noise backgrounds are generated to have a mean of 0.5 and an RMS contrast
% of <pinkcon>.
%
% to achieve different object sizes, the objects provided in <images> are 
% resampled using imresize.m and 'lanczos3' interpolation.
%
% objects are positioned randomly with centers ranging uniformly over the field-of-view
% of the final image.  objects are placed using the masks supplied in <masks>.
% note that objects are placed sequentially, so object overlap can occur.
%
% example:
% figure; im = drawmultiscale(500,1,[2 0.5],2,0.03,rand(100,100,3,4),zeros(100,100,4)); image(im);

%%%%% SETUP

% calc
numim = size(images,4);
rawsz = sizefull(images,2);
dens = (density * 1./szfactors).^2;  % number of objects to place (at each scale)

% report
fprintf('density (the number of objects to place at each scale) is %s.\n',mat2str(dens));

%%%%% DEAL WITH PINK-NOISE BACKGROUND

% generate pink-noise background
im = generatepinknoise(res,[],numdistinct);  % res x res x numdistinct

% set RMS contrast and set DC to 0.5
im = im / std(flatten(im(:,:,1))) * pinkcon + 0.5;
  %std(flatten(temp(:,:,1)))
  %figure;hist(temp(:),100)
  %figure;imagesc(temp,[0 1]);

% truncate, repeat for the RGB channels, and reshape to res x res x 3 x numdistinct
im = permute(repmat(restrictrange(im,0,1),[1 1 1 3]),[1 2 4 3]);

%%%%% DO SOME PRECOMPUTATION

% for each scale
imsz = [];    % scales x 2
isalot = [];  % 1 x scales
resampledimages = {};
resampledmasks = {};
for pp=1:length(szfactors)

  % calc
  imsz(pp,:) = round(rawsz * szfactors(pp));  % what resolution to use for the objects at this scale
  isalot(pp) = ceil(dens(pp)) > numim;
  
  % report
  fprintf('at scale %d, the object resolution is %d x %d.\n',pp,imsz(pp,1),imsz(pp,2));

  % precompute to save time
  if isalot(pp)
    resampledimages{pp} = zeros(imsz(pp,1),imsz(pp,2),3,numim);
    resampledmasks{pp} = zeros(imsz(pp,1),imsz(pp,2),numim);
    for zz=1:numim
      resampledimages{pp}(:,:,:,zz) = restrictrange(imresize(images(:,:,:,zz),imsz(pp,:),'lanczos3'),0,1);
      resampledmasks{pp}(:,:,zz) = restrictrange(imresize(masks(:,:,zz),imsz(pp,:),'lanczos3'),0,1);
    end
  end

end

%%%%% PLACE OBJECTS

% for each distinct image
for rep=1:numdistinct

  % for each scale
  for pp=1:length(szfactors)
    
    % for each object at this scale
    for qq=1:ceil(dens(pp))

      % handle weird fractional case
      if qq > dens(pp) && rand > (dens(pp)-floor(dens(pp)))
        continue;
      end
      
      % calc
      wh = randintrange(1,numim);  % which object to use
      
      % choose a random location for the center of the object (within the field-of-view of the image)
      rowpos = 0.5+rand*res;
      colpos = 0.5+rand*res;
      
      % figure out upper-left corner (nearest whole-pixel indices)
      rowpos = round(rowpos - imsz(pp,1)/2 + 0.5);
      colpos = round(colpos - imsz(pp,2)/2 + 0.5);
      
      % place the image and the mask
      if isalot(pp)
        tempim = placematrix(zeros(res,res,3),resampledimages{pp}(:,:,:,wh),[rowpos colpos]);
        tempmask = placematrix(ones(res,res),resampledmasks{pp}(:,:,wh),[rowpos colpos]);
      else
        tempim = placematrix(zeros(res,res,3), ...
                   restrictrange(imresize(images(:,:,:,wh),imsz(pp,:),'lanczos3'),0,1),[rowpos colpos]);
        tempmask = placematrix(ones(res,res), ...
                     restrictrange(imresize(masks(:,:,wh),imsz(pp,:),'lanczos3'),0,1),[rowpos colpos]);
      end
      
      % place object, mixing with the pink-noise background
      im(:,:,:,rep) = im(:,:,:,rep) .* repmat(tempmask,[1 1 3]) + tempim .* repmat(1-tempmask,[1 1 3]);
      
    end

  end

end
