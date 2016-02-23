function [imaVOL,scaninfo] = loadminc(filename)
%function [imaVOL,scaninfo] = loadminc(filename)
%
% Function to load minc format input file. 
% This function use the netcdf MATLAB utility
%
% Matlab library function for MIA_gui utility. 
% University of Debrecen, PET Center/LB 2010
if nargin == 0
     [FileName, FilePath] = uigetfile('*.mnc','Select minc file');
     filename = [FilePath,FileName];
     if FileName == 0;
          imaVOL = [];scaninfo = [];
          return;
     end
end

ncid=netcdf.open(filename,'NC_NOWRITE');
scaninfo.filename = filename;

%[ndims nvars natts dimm] = netcdf.inq(ncid);
%[varname, xtype, dimids, atts] = netcdf.inqVar(ncid,netcdf.inqVarID(ncid,'xspace'));
% for i=1:atts
%attname = netcdf.inqattname(ncid,netcdf.inqVarID(ncid,varname),i-1)
%attval = netcdf.getAtt(ncid,netcdf.inqVarID(ncid,varname),attname)
%end%

pixsizex = netcdf.getAtt(ncid,netcdf.inqVarID(ncid,'xspace'),'step');
pixsizey = netcdf.getAtt(ncid,netcdf.inqVarID(ncid,'yspace'),'step');
pixsizez = netcdf.getAtt(ncid,netcdf.inqVarID(ncid,'zspace'),'step');
x_start = netcdf.getAtt(ncid,netcdf.inqVarID(ncid,'xspace'),'start');
if isempty(x_start)
    x_start = 0;
end
y_start = netcdf.getAtt(ncid,netcdf.inqVarID(ncid,'yspace'),'start');
if isempty(y_start)
    y_start = 0;
end
z_start = netcdf.getAtt(ncid,netcdf.inqVarID(ncid,'zspace'),'start');
if isempty(z_start)
    z_start = 0;
end

scaninfo.pixsize = abs([pixsizex pixsizey pixsizez]); % abs: Strange could happen
scaninfo.space_start = ([x_start y_start z_start]);

varid = netcdf.inqVarID(ncid,'image-max');
slice_max = netcdf.getVar(ncid,varid,'float');
scaninfo.mag = slice_max;
maxx = max(slice_max(:));
if maxx == round(maxx)
   precision = 'short';
   scaninfo.float = 0;
else
   precision = 'float';
   scaninfo.float = 1;
end


varid = netcdf.inqVarID(ncid,'image');
volume = netcdf.getVar(ncid,varid,precision);

varid = netcdf.inqVarID(ncid,'image-min');
slice_min = netcdf.getVar(ncid,varid,precision);
scaninfo.min = slice_min;
scaninfo.num_of_slice  = size(volume,3);

netcdf.close(ncid);

volume = double(volume);
imsize = size(volume);
% permute the slice image dim. This is for the permut command in thex for
% loop
imaVOL = zeros(imsize([2,1,3]));
slice_min = double(slice_min);
slice_max = double(slice_max);

if length(slice_min) >1 % ha minden slice-hoz el van tárolva a max-min érték 
    for i=1: size(volume,3)
        currentslice = volume(:,:,i);
        imaVOL(:,:,i) = permute( ((currentslice - min(currentslice(:))) / ( max(currentslice(:))- min(currentslice(:)) )...
            *(slice_max(i)- slice_min(i))) - slice_min(i),[2 1]);
    end
else
    imaVOL = permute( ( (volume - min(volume(:))) / ( max(volume(:))- min(volume(:)) )*...
        ((slice_max- slice_min))) - slice_min,[2 1 3]);
end
if strcmp(precision,'short')
    imaVOL = int32(imaVOL);
end

scaninfo.imfm = [size(volume,1) size(volume,2)];
scaninfo.Frames = 1;
scaninfo.start_times = [];
scaninfo.tissue_ts = [];
scaninfo.frame_lengths = [];
scaninfo.FileType    = 'mnc';


% tmp
%imaVOL(imaVOL>0.4)=0.4;




