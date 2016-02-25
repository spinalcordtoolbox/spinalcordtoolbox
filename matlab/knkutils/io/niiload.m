function data = niiload(files,voxelrange,finaldatatype,wanttranspose)

% function data = niiload(files,voxelrange,finaldatatype,wanttranspose)
%
% <files> is a pattern (see matchfiles.m) that matches one or more NIFTI files.
%   each NIFTI file should contain data from a 4D matrix, and the 4D matrices
%   should be consistent in the first three dimensions at least.  we interpret
%   the dimensions as X x Y x Z x T.
% <voxelrange> (optional) is a vector of indices referring to the first three dimensions.
%   these indices do not have to be in any particular order and can include repeats.
%   we return only this section of the data.  special case is [] which means to 
%   return all of the data.  default: [].
% <finaldatatype> (optional) is the final datatype that is desired (achieved through cast.m).
%   [] means to not change the format.
% <wanttranspose> (optional) is whether to perform a transpose at the end (so you get T x voxels).
%   default: 0.
%
% if we match only one file, return <data> as voxels x T.
% if we match multiple files, return <data> as a cell vector of matrices that are voxels x T.
%
% example:
% x1 = uint8(255*rand(100,100,20,10));
% x2 = uint8(255*rand(100,100,20,10));
% save_nii(make_nii(x1),'test1.nii');
% save_nii(make_nii(x2),'test2.nii');
% data = niiload('test*.nii',50:60);
% isequal(data{1},squeeze(x1(50:60,1,1,:)))
% isequal(data{2},squeeze(x2(50:60,1,1,:)))

% input
if ~exist('voxelrange','var') || isempty(voxelrange)
  voxelrange = [];
end
if ~exist('finaldatatype','var') || isempty(finaldatatype)
  finaldatatype = [];
end
if ~exist('wanttranspose','var') || isempty(wanttranspose)
  wanttranspose = 0;
end

% do it
files = matchfiles(files);
if isempty(files)
  error('<files> does not match any files');
end

% figure out xyzsize
temp = load_nii_hdr(files{1});
xyzsize = temp.dime.dim(2:4);

% deal with voxelrange
if isempty(voxelrange)
  voxelrange = 1:prod(xyzsize);
end

% do it
data = {};
for p=1:length(files)
  fprintf('loading data from %s.\n',files{p});
  
  % figure out subscripts and which slices we really need
  [ii1,ii2,ii3] = ind2sub(xyzsize,voxelrange);
  wh = sort(union([],flatten(ii3)));
  
  % load the data
  if isequal(wh,1:xyzsize(3))
    aaa = load_untouch_nii(files{p});  % faster
  else
    aaa = load_untouch_nii(files{p},[],[],[],[],[],wh);
  end
  
  % figure out which voxels we need relative to what we loaded
  iii = flatten(bsxfun(@plus,(1:prod(xyzsize(1:2)))',(wh-1)*prod(xyzsize(1:2))));
  ix = calcposition(iii,voxelrange);

  % subset through data
  data{p} = subscript(squish(aaa.img,3),{ix ':'});

  % convert if necessary
  if ~isempty(finaldatatype)
    data{p} = cast(data{p},finaldatatype);
  end

  % transpose if necessary
  if wanttranspose
    data{p} = data{p}.';
  end

end

% don't embed single cases
if length(data)==1
  data = data{1};
end
