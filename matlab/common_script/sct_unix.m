function sct_unix(cmd)
disp(['>> ' cmd]);
status=unix(cmd);
if status, disp('ERROR!!!!!!!!!!'); end