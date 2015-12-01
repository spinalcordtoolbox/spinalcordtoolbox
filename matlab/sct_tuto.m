% sct_intro
%% add spinalcordtoolbox scripts to your Matlab path
pathtool

%%
fname='t2.nii.gz';
nii=load_nii(fname);
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
[basename, path, ext]=sct_tool_remove_extension(fname,0)
output_fname=[basename, '_processed']
save_nii(nii,output_fname)

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

