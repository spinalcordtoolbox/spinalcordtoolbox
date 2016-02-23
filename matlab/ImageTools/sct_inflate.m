function sct_inflate(fname)
% sct_inflate(fname)
unix(['fslmaths ' fname ' -kernel 2D -fmean -bin ' sct_tool_remove_extension(fname,1) '_inflate'])
