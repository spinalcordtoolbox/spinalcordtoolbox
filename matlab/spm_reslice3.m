function spm_reslice3(P,output,flags)
% Rigid body reslicing of images
% FORMAT spm_reslice(P,flags)
%
% MOD version 2 : option to define output name (flags.output)
%
% P      - matrix or cell array of filenames {one string per row}
%          All operations are performed relative to the first image.
%          ie. Coregistration is to the first image, and resampling
%          of images is into the space of the first image.
%
% output - name for resliced images [default: 'resliced']
%
% flags  - a structure containing various options.  The fields are:
%
%         mask   - mask output images (true/false) [default: true]
%                  To avoid artifactual movement-related variance the
%                  realigned set of images can be internally masked, within
%                  the set (i.e. if any image has a zero value at a voxel
%                  than all images have zero values at that voxel). Zero
%                  values occur when regions 'outside' the image are moved
%                  'inside' the image during realignment.
%
%         mean   - write mean image (true/false) [default: true]
%                  The average of all the realigned scans is written to
%                  an image file with 'mean' prefix.
%
%         interp - the B-spline interpolation method [default: 1]
%                  Non-finite values result in Fourier interpolation. Note
%                  that Fourier interpolation only works for purely rigid
%                  body transformations. Voxel sizes must all be identical
%                  and isotropic.
%
%         which  - values of 0, 1 or 2 are allowed [default: 2]
%                  0   - don't create any resliced images.
%                        Useful if you only want a mean resliced image.
%                  1   - don't reslice the first image.
%                        The first image is not actually moved, so it may
%                        not be necessary to resample it.
%                  2   - reslice all the images.
%                  If which is a 2-element vector, flags.mean will be set
%                  to flags.which(2).
%
%         wrap   - three values of either 0 or 1, representing wrapping in
%                  each of the dimensions. For fMRI, [1 1 0] would be used.
%                  For PET, it would be [0 0 0]. [default: [0 0 0]]
%
%         
%
%__________________________________________________________________________
%
% The spatially realigned images are written to the original subdirectory
% with the same (prefixed) filename. They are all aligned with the first.
%
% Inputs:
% A series of images conforming to SPM data format (see 'Data Format'). The
% relative displacement of the images is stored in their header.
%
% Outputs:
% The routine uses information in their headers and writes the realigned 
% image files to the same subdirectory with a prefix.
%__________________________________________________________________________
% Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging

% John Ashburner
% $Id: spm_reslice.m 4179 2011-01-28 13:57:20Z volkmar $

%__________________________________________________________________________
%
% The headers of the images contain a 4x4 affine transformation matrix 'M',
% usually affected bu the `realignment' and `coregistration' modules.
% What these matrices contain is a mapping from the voxel coordinates
% (x0,y0,z0) (where the first voxel is at coordinate (1,1,1)), to 
% coordinates in millimeters (x1,y1,z1).
%
% x1 = M(1,1)*x0 + M(1,2)*y0 + M(1,3)*z0 + M(1,4)
% y1 = M(2,1)*x0 + M(2,2)*y0 + M(2,3)*z0 + M(2,4)
% z1 = M(3,1)*x0 + M(3,2)*y0 + M(3,3)*z0 + M(3,4)
%
% Assuming that image1 has a transformation matrix M1, and image2 has a
% transformation matrix M2, the mapping from image1 to image2 is: M2\M1
% (ie. from the coordinate system of image1 into millimeters, followed
% by a mapping from millimeters into the space of image2).
%
% Several spatial transformations (realignment, coregistration,
% normalisation) can be combined into a single operation (without the
% necessity of resampling the images several times).
%
%__________________________________________________________________________
% Refs:
%
% Friston KJ, Williams SR, Howard R Frackowiak RSJ and Turner R (1995)
% Movement-related effect in fMRI time-series.  Mag. Res. Med. 35:346-355
%
% W. F. Eddy, M. Fitzgerald and D. C. Noll (1996) Improved Image
% Registration by Using Fourier Interpolation. Mag. Res. Med. 36(6):923-931
%
% R. W. Cox and A. Jesmanowicz (1999)  Real-Time 3D Image Registration
% for Functional MRI. Mag. Res. Med. 42(6):1014-1018
%__________________________________________________________________________


def_flags        = spm_get_defaults('realign.write');
if nargin < 2
    flags = def_flags;
else
    fnms = fieldnames(def_flags);
    for i=1:length(fnms)
        if ~isfield(flags,fnms{i})
            flags.(fnms{i}) = def_flags.(fnms{i});
        end
    end
end

if numel(flags.which) == 2
    flags.mean  = flags.which(2);
    flags.which = flags.which(1);
elseif ~isfield(flags,'mean')
    flags.mean  = 1; 
end

if ~nargin || isempty(P), P = spm_select([2 Inf],'image'); end
if iscellstr(P), P = char(P);    end;
if ischar(P),    P = spm_vol(P); end;
reslice_images(P,flags);


%==========================================================================
function reslice_images(P,flags)
% Reslices images volume by volume
% FORMAT reslice_images(P,flags)
% See main function for a description of the input parameters

if ~isfinite(flags.interp), % Use Fourier method
    % Check for non-rigid transformations in the matrixes
    for i=1:numel(P)
        pp = P(1).mat\P(i).mat;
        if any(abs(svd(pp(1:3,1:3))-1)>1e-7)
            fprintf('\n  Zooms  or shears  appear to  be needed');
            fprintf('\n  (probably due to non-isotropic voxels).');
            fprintf('\n  These  can not yet be  done  using  the');
            fprintf('\n  Fourier reslicing method.  Switching to');
            fprintf('\n  7th degree B-spline interpolation instead.\n\n');
            flags.interp = 7;
            break
        end
    end
end

if flags.mask || flags.mean
    spm_progress_bar('Init',P(1).dim(3),'Computing available voxels','planes completed');
    x1    = repmat((1:P(1).dim(1))',1,P(1).dim(2));
    x2    = repmat( 1:P(1).dim(2)  ,P(1).dim(1),1);
    if flags.mean
        Count    = zeros(P(1).dim(1:3));
        Integral = zeros(P(1).dim(1:3));
    end
    if flags.mask, msk = cell(P(1).dim(3),1);  end;
    for x3 = 1:P(1).dim(3)
        tmp = zeros(P(1).dim(1:2));
        for i = 1:numel(P)
            tmp = tmp + getmask(inv(P(1).mat\P(i).mat),x1,x2,x3,P(i).dim(1:3),flags.wrap);
        end
        if flags.mask, msk{x3} = find(tmp ~= numel(P)); end;
        if flags.mean, Count(:,:,x3) = tmp; end;
        spm_progress_bar('Set',x3);
    end
end

nread = numel(P);
if ~flags.mean
    if flags.which == 1, nread = nread - 1; end;
    if flags.which == 0, nread = 0; end;
end
spm_progress_bar('Init',nread,'Reslicing','volumes completed');

[x1,x2] = ndgrid(1:P(1).dim(1),1:P(1).dim(2));
nread   = 0;
d       = [flags.interp*[1 1 1]' flags.wrap(:)];

for i = 1:numel(P)

    if (i>1 && flags.which==1) || flags.which==2
        write_vol = 1;
    else
        write_vol = 0;
    end
    if write_vol || flags.mean
        read_vol = 1;
    else
        read_vol = 0;
    end

    if read_vol
        if ~isfinite(flags.interp)
            v = abs(kspace3d(spm_bsplinc(P(i),[0 0 0 ; 0 0 0]'),P(1).mat\P(i).mat));
            for x3 = 1:P(1).dim(3)
                if flags.mean
                    Integral(:,:,x3) = ...
                        Integral(:,:,x3) + ...
                        nan2zero(v(:,:,x3) .* ...
                        getmask(inv(P(1).mat\P(i).mat),x1,x2,x3,P(i).dim(1:3),flags.wrap));
                end
                if flags.mask
                    tmp = v(:,:,x3); tmp(msk{x3}) = NaN; v(:,:,x3) = tmp;
                end
            end
        else
            C = spm_bsplinc(P(i), d);
            v = zeros(P(1).dim);
            for x3 = 1:P(1).dim(3)
                [tmp,y1,y2,y3] = getmask(inv(P(1).mat\P(i).mat),x1,x2,x3,P(i).dim(1:3),flags.wrap);
                v(:,:,x3)      = spm_bsplins(C, y1,y2,y3, d);
                % v(~tmp)      = 0;

                if flags.mean
                    Integral(:,:,x3) = Integral(:,:,x3) + nan2zero(v(:,:,x3));
                end
                if flags.mask
                    tmp = v(:,:,x3); tmp(msk{x3}) = NaN; v(:,:,x3) = tmp;
                end
            end
        end
        if write_vol
            VO         = P(i);
            [pth,nm,xt,vr] = spm_fileparts(deblank(P(i).fname));
            VO.fname   = output;
            VO.dim     = P(1).dim(1:3);
            VO.dt      = P(i).dt;
            VO.pinfo   = P(i).pinfo;
            VO.mat     = P(1).mat;
            VO.descrip = 'spm - realigned';
            VO = spm_write_vol(VO,v);
        end

        nread = nread + 1;
    end
    spm_progress_bar('Set',nread);
end

if flags.mean
    % Write integral image (16 bit signed)
    %----------------------------------------------------------------------
    Integral    = Integral./Count;
    PO          = P(1);
    PO          = rmfield(PO,'pinfo');
    [pth,nm,xt] = spm_fileparts(deblank(P(1).fname));
    PO.fname    = fullfile(pth,['mean' nm xt]);
    PO.pinfo    = [max(max(max(Integral)))/32767 0 0]';
    PO.descrip  = 'spm - mean image';
    PO.dt       = [spm_type('int16') spm_platform('bigend')];
    spm_write_vol(PO,Integral);
end

spm_figure('Clear','Interactive');


%==========================================================================
function v = kspace3d(v,M)
% 3D rigid body transformation performed as shears in 1D Fourier space.
% FORMAT v1 = kspace3d(v,M)
% Inputs:
% v - the image stored as a 3D array.
% M - the rigid body transformation matrix.
% Output:
% v - the transformed image.
%
% The routine is based on the excellent papers:
% R. W. Cox and A. Jesmanowicz (1999)
% Real-Time 3D Image Registration for Functional MRI
% Magnetic Resonance in Medicine 42(6):1014-1018
%
% W. F. Eddy, M. Fitzgerald and D. C. Noll (1996)
% Improved Image Registration by Using Fourier Interpolation
% Magnetic Resonance in Medicine 36(6):923-931
%__________________________________________________________________________

[S0,S1,S2,S3] = shear_decomp(M);

d  = [size(v) 1 1 1];
g = 2.^ceil(log2(d));
if any(g~=d)
    tmp = v;
    v   = zeros(g);
    v(1:d(1),1:d(2),1:d(3)) = tmp;
    clear tmp;
end

% XY-shear
tmp1 = -sqrt(-1)*2*pi*([0:((g(3)-1)/2) 0 (-g(3)/2+1):-1])/g(3);
for j=1:g(2)
    t        = reshape( exp((j*S3(3,2) + S3(3,1)*(1:g(1)) + S3(3,4)).'*tmp1) ,[g(1) 1 g(3)]);
    v(:,j,:) = real(ifft(fft(v(:,j,:),[],3).*t,[],3));
end

% XZ-shear
tmp1 = -sqrt(-1)*2*pi*([0:((g(2)-1)/2) 0 (-g(2)/2+1):-1])/g(2);
for k=1:g(3)
    t        = exp( (k*S2(2,3) + S2(2,1)*(1:g(1)) + S2(2,4)).'*tmp1);
    v(:,:,k) = real(ifft(fft(v(:,:,k),[],2).*t,[],2));
end

% YZ-shear
tmp1 = -sqrt(-1)*2*pi*([0:((g(1)-1)/2) 0 (-g(1)/2+1):-1])/g(1);
for k=1:g(3)
    t        = exp( tmp1.'*(k*S1(1,3) + S1(1,2)*(1:g(2)) + S1(1,4)));
    v(:,:,k) = real(ifft(fft(v(:,:,k),[],1).*t,[],1));
end

% XY-shear
tmp1 = -sqrt(-1)*2*pi*([0:((g(3)-1)/2) 0 (-g(3)/2+1):-1])/g(3);
for j=1:g(2)
    t        = reshape( exp( (j*S0(3,2) + S0(3,1)*(1:g(1)) + S0(3,4)).'*tmp1) ,[g(1) 1 g(3)]);
    v(:,j,:) = real(ifft(fft(v(:,j,:),[],3).*t,[],3));
end

if any(g~=d), v = v(1:d(1),1:d(2),1:d(3)); end


%==========================================================================
function [S0,S1,S2,S3] = shear_decomp(A)
% Decompose rotation and translation matrix A into shears S0, S1, S2 and
% S3, such that A = S0*S1*S2*S3. The original procedure is documented in:
% R. W. Cox and A. Jesmanowicz (1999)
% Real-Time 3D Image Registration for Functional MRI
% Magnetic Resonance in Medicine 42(6):1014-1018

A0 = A(1:3,1:3);
if any(abs(svd(A0)-1)>1e-7), error('Can''t decompose matrix'); end

t  = A0(2,3); if t==0, t=eps; end
a0 = pinv(A0([1 2],[2 3])')*[(A0(3,2)-(A0(2,2)-1)/t) (A0(3,3)-1)]';
S0 = [1 0 0; 0 1 0; a0(1) a0(2) 1];
A1 = S0\A0;  a1 = pinv(A1([2 3],[2 3])')*A1(1,[2 3])';  S1 = [1 a1(1) a1(2); 0 1 0; 0 0 1];
A2 = S1\A1;  a2 = pinv(A2([1 3],[1 3])')*A2(2,[1 3])';  S2 = [1 0 0; a2(1) 1 a2(2); 0 0 1];
A3 = S2\A2;  a3 = pinv(A3([1 2],[1 2])')*A3(3,[1 2])';  S3 = [1 0 0; 0 1 0; a3(1) a3(2) 1];

s3 = A(3,4)-a0(1)*A(1,4)-a0(2)*A(2,4);
s1 = A(1,4)-a1(1)*A(2,4);
s2 = A(2,4);
S0 = [[S0 [0  0 s3]'];[0 0 0 1]];
S1 = [[S1 [s1 0  0]'];[0 0 0 1]];
S2 = [[S2 [0 s2  0]'];[0 0 0 1]];
S3 = [[S3 [0  0  0]'];[0 0 0 1]];


%==========================================================================
function [Mask,y1,y2,y3] = getmask(M,x1,x2,x3,dim,wrp)
tiny = 5e-2; % From spm_vol_utils.c
y1   = M(1,1)*x1+M(1,2)*x2+(M(1,3)*x3+M(1,4));
y2   = M(2,1)*x1+M(2,2)*x2+(M(2,3)*x3+M(2,4));
y3   = M(3,1)*x1+M(3,2)*x2+(M(3,3)*x3+M(3,4));
Mask = true(size(y1));
if ~wrp(1), Mask = Mask & (y1 >= (1-tiny) & y1 <= (dim(1)+tiny)); end
if ~wrp(2), Mask = Mask & (y2 >= (1-tiny) & y2 <= (dim(2)+tiny)); end
if ~wrp(3), Mask = Mask & (y3 >= (1-tiny) & y3 <= (dim(3)+tiny)); end


%==========================================================================
function vo = nan2zero(vi)
vo = vi;
vo(~isfinite(vo)) = 0;
