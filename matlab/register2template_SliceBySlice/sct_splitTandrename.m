function files = sct_splitTandrename(fname)
% files = splitandrename(fname)
% EXAMPLE:
% fname : ./../file.nii.gz
% files = {'fileT1', 'fileT2'...}
%



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
cmd = ['fslsplit ' fname ' ' path file 'T -t'];
sct_unix(cmd);

%rename
numT = j_numbering(dim(4),4,0);
for iT = 1:dim(4)
    files{iT} = [file 'T' numT{iT}];
end

end
