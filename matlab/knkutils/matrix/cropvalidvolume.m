function [newvol,voloffset] = cropvalidvolume(vol,fun)

% function [newvol,voloffset] = cropvalidvolume(vol,fun)
%
% <vol> is a 3D matrix
% <fun> is a function that computes a boolean value for 
%   each entry in a 3D matrix.  the boolean value indicates
%   which entries are valid.
%
% return <newvol> as the minimal subvolume of <vol> that contains
% all the valid entries according to <fun>.  return <voloffset> 
% as a 3-element vector of non-negative integers indicating the
% offset associated with <newvol>.  for example, [10 0 2] means
% that <newvol> starts at the (11,1,3)th entry in the original
% <vol>.
%
% example:
% vol = placematrix2(NaN*zeros(20,20,20),randn(10,10,10),[3 5 7]);
% [newvol,voloffset] = cropvalidvolume(vol,@(x) ~isnan(x));
% figure; imagesc(makeimagestack(vol));
% figure; imagesc(makeimagestack(newvol));
% voloffset

% where are the bad values?
bad = ~feval(fun,vol);

% indices of good rows, columns, and depths
rowindices = find(~all(all(bad,2),3));
colindices = find(~all(all(bad,1),3));
depindices = find(~all(all(bad,1),2));

% do it
newvol = vol(rowindices(1):rowindices(end),colindices(1):colindices(end),depindices(1):depindices(end));
voloffset = [rowindices(1)-1 colindices(1)-1 depindices(1)-1];
