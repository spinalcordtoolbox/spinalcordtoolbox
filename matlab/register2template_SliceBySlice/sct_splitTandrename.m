function files = sct_splitTandrename(fname)
% files = splitandrename(fname, (uppest_level) )
%
% fname : ./../file.nii.gz
% files = {'fileT1', 'fileT2'...}
%



log = 'log';
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
j_disp(log,['>> ',cmd]); [status result] = unix(cmd); if status, error(result); end

%rename
numT = j_numbering(dim(4),4,0);
for iT = 1:dim(4)
    files{iT} = [file 'T' numT{iT}];
end

end
