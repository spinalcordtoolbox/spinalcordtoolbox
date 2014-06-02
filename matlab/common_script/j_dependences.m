% =========================================================================
% FUNCTION
% j_dependences.m
%
% Find dependences in my scripts.
%
% INPUT
% (file_name)		array		name of matlab function
% 
% OUTPUT
% (-)
%
% Julien Cohen-Adad <jcohen@nmr.mgh.harvard.edu>  2011-06-08
% =========================================================================
function j_dependences(file_name)


global			PATH_SCRIPT
global			PATH_WRITE
global			DEPENDENCES

% user initialization
prefixe			= 'j_';
PATH_SCRIPT		= '/Users/julien/Matlab/script';
PATH_WRITE		= '/Users/julien/Desktop/';
bCopy_files		= 1; % copy dependent files to a specified folder

% default initialization
DEPENDENCES		= {};

% select m-file
if ~exist('file_name')
    opt.output          = 'array';
    opt.ext_filter      = 'm';
    opt.file_selection  = 'matlab';
    fname = j_getfiles(opt);
    if ~fname, return; end
end

% get full path name of the function
fname = which(file_name);

% find dependences
% [a b c] = fileparts(fname);
% dependences.father = b;
dependences = find_dependences_recursively(fname,prefixe);

if isempty(dependences)
	disp('Prefixe was not contained within one file. Exit program')
	return
end

% write temp file to retrieve fields recursively
% fid = fopen('tmp.txt','w');
% write_field_recursively(fid,dependences);
% fclose(fid);

% % remove duplicates
% cDependences_tmp = textread('tmp.txt','%s');
% cDependences{1} = cDependences_tmp{1};
% for i=1:size(cDependences_tmp,1)
% 	found_one = 0;
%     size_cDependences = size(cDependences,2);
%     for j=1:size_cDependences
%         if strcmp(cDependences{j},cDependences_tmp{i})
%             found_one = 1;
%         end
%     end
%     if ~found_one
%         cDependences{size_cDependences+1} = cDependences_tmp{i};
%     end
% end

% display dependences
fprintf('\n\nDEPENDENCES OF %s.m',DEPENDENCES{1})
fprintf('\n------------------------------------------------')
for i=2:size(DEPENDENCES,2)
    fprintf('\n%s',DEPENDENCES{i})
end
fprintf('\n\n')

% copy files
if bCopy_files
	fprintf('Copy files ...')
	copy_files(DEPENDENCES);
	fprintf('\nDone\n\n');
end

% delete temp files
% delete('tmp.txt');








% =========================================================================
% =========================================================================

function dependences = find_dependences_recursively(fname,prefixe)

global			PATH_SCRIPT
global			DEPENDENCES

% select m-file
if ~exist('fname')
    opt.output          = 'array';
    opt.ext_filter      = '*.m';
    opt.file_selection  = 'matlab';
    fname = j_getfiles(opt);
end

% check for property files in selected m_file
test_string = textread(fname,'%s');
j=1;
for i=1:size(test_string,1)
    if length(test_string{i})>length(prefixe)-1
		% find 'j_' string for the selected line
		string_found = findstr(test_string{i},prefixe);
		nb_strings = length(string_found);
% 		string_identified = 0;
		% identify the whole function name for each identifies string
		nb_chars = length(test_string{i});
		for iString = 1:nb_strings
			% scan the rest of the line, starting from the specified index
			for iChar = 1:nb_chars-string_found(iString)-1
				% stops scanning when found typical character such as '(' or ';'
				test_char = test_string{i}(string_found(iString)+iChar+1);
				if strcmp(test_char,'(') | strcmp(test_char,'.') | strcmp(test_char,';') | strcmp(test_char,',') | strcmp(test_char,' ');
					% save the identified function
					dependences_tmp{j} = test_string{i}(string_found(iString):string_found(iString)+iChar);
					j=j+1;
					% quit loop
					break;
				end
			end
		end
		
%         if strcmp(test_string{i}(1:length(prefixe)),prefixe)
% 			k_ref = length(test_string{i});
%             for k=1:length(test_string{i}-1)
%                 if strcmp(test_string{i}(k),'(') | strcmp(test_string{i}(k),'.') | strcmp(test_string{i}(k),';') | strcmp(test_string{i}(k),',');
%                     k_ref = k-1;
%                     break;
%                 end
%             end
%             dependences_tmp{j} = test_string{i}(1:k_ref);
%             j=j+1;
%         end
    end
end

% check if the prefixe was nether found on the file
if exist('dependences_tmp')
	dependences{1} = dependences_tmp{1};
else
	dependences = {};
	return
end

% remove duplicates
for i=1:size(dependences_tmp,2)
	found_one = 0;
    size_dependences = size(dependences,2);
    for j=1:size_dependences
        if strcmp(dependences{j},dependences_tmp{i})
            found_one = 1;
        end
    end
    if ~found_one
        dependences{size_dependences+1} = dependences_tmp{i};
    end
end
clear dependences_tmp

% remove fields already identified (in previous recursive loops)
dependences_tmp = {};
if ~isempty(DEPENDENCES)
	k = 1;
	for i=1:size(dependences,2)
		same_field = 0;
		for j=1:size(DEPENDENCES,2)
			if strcmp(DEPENDENCES{j},dependences{i})
				same_field = 1;
			end
		end
		if ~same_field
			dependences_tmp{k} = dependences{i};
			k = k+1;
		end
	end
	dependences = dependences_tmp;
	clear dependences_tmp
end

% save identified dependences in global variable
size_DEPENDENCES = size(DEPENDENCES,2);
for i=1:size(dependences,2)
	DEPENDENCES{size_DEPENDENCES+i} = dependences{i};
end

% remove first field (if it is the father's field)
[a b c] = fileparts(fname);
dependences_father = b;
dependences_tmp = dependences;
dependences = {};
if ~size(dependences_tmp,2)
	dependences = {};
elseif strcmp(dependences_tmp{1},dependences_father)
	for i=2:size(dependences_tmp,2)
		dependences{i-1}.father = dependences_tmp{i};
	end
else
	for i=1:size(dependences_tmp,2)
		dependences{i}.father = dependences_tmp{i};
	end
end
clear dependences_tmp

% get dependences in dependent files (recursive function)
for i=1:size(dependences,2)
	fname_sub = []; % JULIEN 2009-11-07
    % build file name
    dir_script = j_dir(PATH_SCRIPT,'.m');
	for j=1:size(dir_script,2)
        if ~isempty(findstr(dir_script{j},strcat(dependences{i}.father,'.m')))
            fname_sub = dir_script{j};
            break
        end
	end
	if ~isempty(fname_sub)
		dependences{i}.son = find_dependences_recursively(fname_sub,prefixe);
	end
end





% =========================================================================
% =========================================================================

function write_field_recursively(fid,dependences)

fprintf(fid,'%s\n',dependences.father);
for i=1:size(dependences.son,2)
    write_field_recursively(fid,dependences.son{i});
end




% =========================================================================
% =========================================================================

function copy_files(cDependences)

global			PATH_SCRIPT;
global			PATH_WRITE;

iFileFound = 1;
for i=1:size(cDependences,2)
    % build file name
    dir_script = j_dir(PATH_SCRIPT,'.m');
	fileFound = 0;
	for j=1:size(dir_script,2)
        if ~isempty(findstr(dir_script{j},strcat(cDependences{i},'.m')))
            fname_copy{iFileFound} = dir_script{j};
			fileFound = 1;
            break
        end
	end
	if fileFound
		iFileFound = iFileFound + 1;
	else
		fprintf(['\nWarning: File "',cDependences{i},'.m" not found.'])
	end
end

% create folder within specified path
folder_write = strcat(PATH_WRITE,filesep,cDependences{1},filesep);
if ~isdir(folder_write), mkdir(folder_write); end

% copy files to the specified folder
for i=1:size(fname_copy,2)
	copyfile(fname_copy{i},folder_write);
end




