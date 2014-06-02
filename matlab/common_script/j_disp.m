function j_disp(fname,txt)



% open log file
fid = fopen(fname,'a');

% write stuff in file
fprintf(fid,[txt,'\n']);

% close file
fclose(fid);

% disp on the command window with two carriage returns
fprintf([txt,'\n'])