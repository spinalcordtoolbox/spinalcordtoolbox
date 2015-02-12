function files = sct_sliceandrename(fname, dir, varargin)
% files = splitandrename(fname, 'x'/'y'/'z' or 't',(uppest_level) )
%
% fname : ./../file.nii.gz
% files = {'fileC1', 'fileC2'...}
% uppest_level facultative (default = 1)
%



log = 'log';
if ~isempty(varargin), uppest_level = varargin{1}; else uppest_level=1; end
% read file parts
[file, path, ext]=sct_tool_remove_extension(fname,0);


[~,dim] = read_avw(fname);
% split by Z
cmd = ['fslsplit ' fname ' ' path file '_' dir ' -' dir];
j_disp(log,['>> ',cmd]); [status, result] = unix(cmd); if status, error(result); end

%rename
if strcmp(dir,'x'), dirn=1; end
if strcmp(dir,'y'), dirn=2; end
if strcmp(dir,'z'), dirn=3; end
if strcmp(dir,'t'), dirn=4; end

numT = j_numbering(dim(dirn),4,0);
outputext='.nii.gz';
for i_slice = 1:dim(dirn)
    files{i_slice} = [path file '_' dir numT{i_slice} outputext];    
end

end
