function files = sct_sliceandrename(fname, varargin)
% files = splitandrename(fname, (uppest_level) )
%
% fname : ./../file.nii.gz
% files = {'fileC1', 'fileC2'...}
% uppest_level facultative (default = 1)
%



log = 'log';
if ~isempty(varargin), uppest_level = varargin{1}; else uppest_level=1; end
% read file parts
[path, file, ext] = fileparts(fname);
[~,file,ext2] = fileparts(file);
ext = [ext2 ext]; % .nii.gz has two extension!
if isempty(path), path = '.'; end
if isempty(ext), ext = '.nii.gz'; end
ext = '.nii.gz';
path = [path filesep];


[~,dim] = read_avw(fname);
% split by Z
cmd = ['fslsplit ' fname ' ' path file 'C -z'];
j_disp(log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end

%rename
numT = j_numbering(dim(3),4,0);
for i_slice = 1:dim(3)
    files{i_slice} = [file 'C' num2str(i_slice)];
    cmd = ['mv ' path file 'C' numT{dim(3)+1 - i_slice} ext ' ' file 'C' num2str(uppest_level+i_slice-1) ext ];
    j_disp(log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end
    
end

end
