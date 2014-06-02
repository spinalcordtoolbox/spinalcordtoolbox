function save_avw_v2(img,fname,vtype,vsize, varargin)
% SAVE_AVW(img,fname,vtype,vsize,(nifti_source),(copy dims?)) 
%
%  Create and save an analyse header (.hdr) and image (.img) file
%   for either a 2D or 3D or 4D array (automatically determined).
%  fname is the filename (must be inside single quotes)
%   
%  vtype is 1 character: 'b'=unsigned byte, 's'=short, 'i'=int, 'f'=float
%                        'd'=double or 'c'=complex
%  vsize is a vector [x y z tr] containing the voxel sizes in mm and
%  the tr in seconds  (defaults: [1 1 1 3])
%
%
%  OPTIONAL : nifti_source from which header will be copied
%
%  See also: READ_AVW
%

%% Save a temp volume in Analyze format
tmpname = tempname;

   if ((~isreal(img)) & (vtype~='c')),
     disp('WARNING:: Overwriting type - saving as complex');
     save_avw_complex(img,tmpname,vsize);
   else
     if (vtype=='c'),
       save_avw_complex(img,tmpname,vsize);
     else
       save_avw_hdr(img,tmpname,vtype,vsize);
       % determine endianness of header
       [dims,scales,bpp,endian,datatype]=read_avw_hdr(tmpname);
       save_avw_img(img,tmpname,vtype,endian);
     end
   end
         
%% Convert volume from NIFTI_PAIR format to user default
tmp=sprintf('sh -c ". ${FSLDIR}/etc/fslconf/fsl.sh; export FSLOUTPUTTYPE=NIFTI; $FSLDIR/bin/fslmaths %s %s"\n',tmpname,fname);
system(tmp);

if ~isempty(varargin)
    if length(varargin)<2 || ~varargin{2}, copydim = ' -d'; else copydim=''; end 
    cmd = ['fslcpgeom ' varargin{1} ' ' fname copydim]; [status result] = unix(cmd); if status, disp(result); end
end
% cross platform compatible deleting of files
delete([tmpname,'.hdr']);
delete([tmpname,'.img']);

