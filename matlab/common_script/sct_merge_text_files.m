
function sct_tools_merge_text_files(text_files, output,transpose)
% sct_tools_merge_text_files('fsems_*.bvec', output, transpose?)
dbstop if error
list_text=sct_tools_ls(text_files);
if isempty(list_text), error('no files found, check file name'); end
% =========================================================================
% DON'T CHANGE BELOW
% =========================================================================

if exist('transpose','var') & transpose
    for i_seq=1:length(list_text)
        unix(['sct_dmri_transpose_bvecs.py ' list_text{i_seq}]);
        list_text{i_seq} = strrep(list_text{i_seq},'.bvec','_t.bvec');
    end
end


% =========================================================================
% MERGE BVEC FILE AND GENERATE SCHEME FILE
% =========================================================================

copyfile(list_text{1},output)
output_fid = fopen(output,'a+');
first_file=txt2mat(output);
Nb_pt=size(first_file,1);

for i_seq=2:length(list_text)
    text=txt2mat(list_text{i_seq});
    Nb_pt=Nb_pt+size(text,1);
    for i_line=1:size(text,1)
        for i_column=1:size(text,2)
            % write bvecs
            fprintf(output_fid, '%f   ',text(i_line,i_column));
        end
        fprintf(output_fid, '\n');
    end
end

fclose all;

disp(['Total number of lines : ' num2str(Nb_pt)])
