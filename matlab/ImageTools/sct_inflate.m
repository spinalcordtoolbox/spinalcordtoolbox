function sct_inflate(fname)
% sct_inflate(fname)
unix(['fslmaths ' fname ' -s 0.1 -bin ' sct_tool_remove_extension(fname,1) '_inflate'])
