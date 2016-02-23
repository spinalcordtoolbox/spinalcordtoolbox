% sct_tuto
%% add spinalcordtoolbox scripts to your Matlab path
pathtool

%% load an MRI data
input_fname='t2.nii.gz';
nii=load_nii(input_fname);
%% get a slice:
img=squeeze(nii.img(floor(end/2),:,:)); % rigth click on squeeze --> help
%% display
imagesc(img,[0 1000])
colormap gray
colorbar
axis image

%% process image
nii.img(floor(end/2),:,:)=0;

%% save image
[basename, path, ext]=sct_tool_remove_extension(input_fname,0)
output_fname=[basename, '_processed']
save_nii(nii,output_fname) % OR save_nii_v2(nii.img,output_fname,input_fname)

%% display image
cmd=['fslview ' output_fname]
unix(cmd)

%% parser
function(varargin)
p=inputParser
addOptional('file','fds.nii',@isstr)
addOptional('page',1,@isnumeric)
addOptional('file','fds.nii',@isstr)
addOptional('file','fds.nii',@isstr)

parse(p,varargin{:})

in=p.Results;

in.file

