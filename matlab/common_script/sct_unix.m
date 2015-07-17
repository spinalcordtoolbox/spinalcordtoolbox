function sct_unix(cmd)
disp(['<strong> >> ' cmd '</strong>']);
status=unix(cmd);
if status, disp('ERROR!!!!!!!!!!'); pause; end