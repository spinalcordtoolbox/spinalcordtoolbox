function tmp_folder=sct_tempdir
% tmp_folder=sct_tempdir
tmp_folder = ['tmp_' datestr(now,'yymmdd_hhMMSS')];
mkdir(tmp_folder)