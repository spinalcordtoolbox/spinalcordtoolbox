function j_changeFileExtension(ext_old,ext_new)
% =========================================================================
% 
% Change file extension of all files in current directory.
% N.B. Case sensitive!!! I.e., "IMA" is not the same as "ima"
% 
% 
% INPUT
% ext_old		string. Old extension.
% ext_new		string. New extension.
% 
% OUTPUT
% -
% 
%   Example
%   
%
% Author: Julien Cohen-Adad <jcohen@nmr.mgh.harvard.edu>
% 2011-10-07
% 
% =========================================================================

path_name = pwd;

disp(' ')
disp(['Old extension: ',ext_old])
disp(['New extension: ',ext_new])

list_fname = dir(path_name);
nb_files = size(list_fname,1);
disp(['Number of files: ',num2str(nb_files)])

j_progress('Change file names...')
for iFile=1:nb_files
	fname_old = list_fname(iFile).name;
	[path_name,file_name,ext] = fileparts(fname_old);
	if strcmp(ext(2:end),ext_old)
		fname_new = cat(2,path_name,file_name,'.',ext_new);
% 		cmd = ['mv ',fname_old,' ',fname_new];
% 		[status result] = unix(cmd);
		movefile(fname_old,fname_new);
	end	
	j_progress(iFile/nb_files)
end

disp('Done!')
disp(' ')