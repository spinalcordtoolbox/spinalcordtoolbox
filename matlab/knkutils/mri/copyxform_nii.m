function copyxform_nii(source,dest,rot,flipfirst,flipsecond,flipthird)

% function copyxform_nii(source,dest,rot,flipfirst,flipsecond,flipthird)
%
% <source> is a .nii file
% <dest> is a wildcard matching one or more .nii files
% <rot> is the number of in-plane CCW rotations to apply
% <flipfirst> is whether to flip the first dimension
% <flipsecond> is whether to flip the second dimension
% <flipthird> is whether to flip the third dimension
%
% for each file in <dest>:
%   propagate the transformation-related fields (see code
%   for details) from <source>.  also, rotate CCW
%   and flip the dimensions of the data matrix.
%   finally, save the result back to <dest>.
%
% the usefulness of this function is if you need to massage
% the transformation-related aspects of the files in <dest>.

[h,filetype,fileprefix,machine] = load_nii_hdr(source);
dest = matchfiles(dest);
for p=1:length(dest)
  h2 = load_untouch_nii(dest{p});
  h2.hdr.hist.qform_code = h.hist.qform_code;
  h2.hdr.hist.sform_code = h.hist.sform_code;
  h2.hdr.hist.quatern_b = h.hist.quatern_b;
  h2.hdr.hist.quatern_c = h.hist.quatern_c;
  h2.hdr.hist.quatern_d = h.hist.quatern_d;
  h2.hdr.hist.qoffset_x = h.hist.qoffset_x;
  h2.hdr.hist.qoffset_y = h.hist.qoffset_y;
  h2.hdr.hist.qoffset_z = h.hist.qoffset_z;
  h2.hdr.hist.srow_x     = h.hist.srow_x;
  h2.hdr.hist.srow_y     = h.hist.srow_y;
  h2.hdr.hist.srow_z     = h.hist.srow_z;
  h2.hdr.hist.magic      = h.hist.magic;
  h2.img = flipdims(rotatematrix(h2.img,1,2,rot),[flipfirst flipsecond flipthird]);
  save_untouch_nii(h2,dest{p});
end

%  h2.hdr.dime.xyzt_units = h.dime.xyzt_units;
%  h2.hdr.dime = h.dime;
%  h2.hdr.hist = h.hist;
%   fid = fopen(dest{p},'a');
%   fseek(fid,0,'bof');
%   save_nii_hdr(h2,fid);
%   fclose(fid);
