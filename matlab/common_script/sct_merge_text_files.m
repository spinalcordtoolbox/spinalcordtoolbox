
function sct_tools_merge_text_files(list_text, output,transpose)
% sct_tools_merge_text_files({'fsems_D9d3_X01.scheme','fsems_D15d8_X01.scheme'}, output, transpose?)
dbstop if error

% =========================================================================
% DON'T CHANGE BELOW
% =========================================================================

if exist('transpose','var') && transpose
    for i_seq=1:length(list_text)
        unix(['sct_dmri_transpose_bvecs.py ' list_text{i_seq}]);
        list_text{i_seq} = strrep(list_text{i_seq},'.bvec','_t.bvec');
    end
end


% =========================================================================
% MERGE BVEC FILE AND GENERATE SCHEME FILE
% =========================================================================

copyfile(list_text{1},output)
output_fid = fopen(output,'r+');
first_file=textscan(output_fid,'%s','delimiter','\n','CommentStyle','#');
Nb_pt=length(first_file{1});

for i_seq=2:length(list_text)
    fid=fopen(list_text{i_seq},'r');
    text=textscan(fid,'%s','delimiter','\n','CommentStyle','#');
    list_text{i_seq}
    length(text{1})
    Nb_pt=Nb_pt+length(text{1});
    for i_line=1:length(text{1})
        % write bvecs
        fprintf(output_fid, '%s\n',text{1}{i_line});
    end
end

fclose all;

disp(['Total number of lines : ' num2str(Nb_pt)])
