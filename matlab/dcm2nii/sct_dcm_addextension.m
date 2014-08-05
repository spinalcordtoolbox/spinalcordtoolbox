function sct_dcm_addextension(dcmdir)
% sct_dcm_addextension('MR*')
% add .dcm at the end of all dicoms
unix(['for f in ' dcmdir ' ; do  mv "$f" "$f.dcm"; done']);