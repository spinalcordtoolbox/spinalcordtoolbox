function sct_tools_matrix2txt(M,output)
% sct_matrix2txt(Matrix,fname_output)
output_fid = fopen(output,'w+');
for i_line=1:size(M,1)
    for i_column=1:size(M,2)
    % write bvecs
    fprintf(output_fid, '%d ',M(i_line,i_column));
    end
    fprintf(output_fid, '\n',M(i_line,i_column));
end
fclose(output_fid);
