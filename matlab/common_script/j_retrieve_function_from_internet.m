% Retrieve function from internet when each line is fucked up with stuff
% like 0002, 0003, 0004, etc.
function j_retrieve_function_from_internet

fname		= '/Users/julien/Desktop/write_fsf.txt';
fname_write	= '/Users/julien/mri/spinal_co2/write_fsf.m';
nb_char		= 6; % nb char to remove for each line


% load file containing code
txt = textread(fname,'%s','delimiter','\n');
nb_lines = size(txt,1);

% remove first n character
for iLine = 1:nb_lines
	txt_new{iLine,1} = txt{iLine}(nb_char:end);
end

% write new file
fid = fopen(fname_write,'w');
for iLine = 1:nb_lines
	fprintf(fid,'%s\n',txt_new{iLine,1});
end
fclose(fid);
