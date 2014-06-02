function sct_merge_text_files(text_files, output,transpose)
% sct_merge_text_files(text_files, output)
dbstop if error


% =========================================================================
% DON'T CHANGE BELOW
% =========================================================================

if transpose
    for i_seq=1:length(text_files)
        unix(['sct_dmri_transpose_bvecs.py ' text_files{i_seq}]);
        text_files{i_seq} = strrep(text_files{i_seq},'.bvec','_t.bvec');
    end
end


% =========================================================================
% MERGE BVEC FILE AND GENERATE SCHEME FILE
% =========================================================================

copyfile(text_files{1},output)
output_fid = fopen(output,'r+');
first_file=textscan(output_fid,'%s','delimiter','\n','CommentStyle','#');
Nb_pt=length(first_file{1});

for i_seq=2:length(text_files)
    fid=fopen(text_files{i_seq},'r');
    text=textscan(fid,'%s','delimiter','\n','CommentStyle','#');
    text_files{i_seq}
    length(text{1})
    Nb_pt=Nb_pt+length(text{1});
    for i_line=1:length(text{1})
        % write bvecs
        fprintf(output_fid, '%s\n',text{1}{i_line});
    end
end

fclose all;

disp(['Total number of lines : ' num2str(Nb_pt)])
