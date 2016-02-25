function [hrf,tr,label] = getsamplehrf(wh,takemean)

% function [hrf,tr,label] = getsamplehrf(wh,takemean)
%
% <wh> (optional) is a vector of indices referring to cases.
%   default is [] which means to return all cases.
% <takemean> (optional) is whether to take the mean HRF across
%   voxels, normalize this mean to be unit-length, and return
%   only this result.  default: 1.
%
% return:
%   <hrf> as a cell vector of things that are time x voxels.
%     these are hemodynamic response functions (HRFs).  if
%     <takemean>, the dimensions of each thing will be time x 1.
%   <tr> as [A B C ...] where A, B, C are TRs in seconds
%   <label> as {A B C ...} where A, B, C are text labels
%
% notes:
% - the first point of the HRFs is always 0 and coincides with trial onset.
% - for cases 1-7, we use DCT basis functions (with some frequency cutoff) to fit the HRF of each voxel.
% - for cases 8-18, we use delta basis functions to fit the HRF of each voxel.
% - we use polynomials to model signal drift.
% - each HRF is unit-length-normalized and sign-flipped (based on the integral over some initial time range).
% - we consider HRFs of only those voxels that have a cross-validation R^2 > 16% (see fitprf.m).
%
% cases:
%  1. S1, 20100430, 2.5 mm x 2.5 mm x 2.5 mm, 3 s (15 static grayscale frames, 100 ms ON / 100 ms OFF), min ISI 3 s
%  2. S1, 20100707, "                       , "                                                       , "
%  3. S2, 20100901, "                       , "                                                       , "
%  4. S3, 20100903, "                       , 1 s (10 static grayscale frames, 100 ms ON / 0 ms OFF)  , "
%  5. S2, 20100923, "                       , 2 s (10 static grayscale frames, 100 ms ON / 100 ms OFF), min ISI 4 s
%  6. S2, 20100928, "                       , "                                                       , "
%  7. S2, 20101014, "                       , 4 s (40 static grayscale frames, 100 ms ON / 0 ms OFF)  , "
%  8. S3, 20101026, "                       , 4 s (8 static color frames, 400 ms ON / 100 ms OFF)     , "
%  9. S1, 20101103, "                       , 3 s (30 static grayscale frames, 100 ms ON / 0 ms OFF)  , min ISI 5 s
% 10. S2, 20101118, "                       , "                                                       , "
% 11. S4, 20101126, "                       , "                                                       , "
% 12. S3, 20110113, "                       , "                                                       , "
% 13. S2, 20110121, "                       , 2.9 s (88 grayscale frames, 33 ms ON / 0 ms OFF)        , min ISI 5.1 s
% 14. S5, 20110125, "                       , 3 s (30 static grayscale frames, 100 ms ON / 0 ms OFF)  , min ISI 5 s
% 15. S4, 20110209, "                       , "                                                       , "
% 16. S5, 20110215, "                       , "                                                       , "
% 17. S6, 20110216, "                       , "                                                       , "
% 18. S4, 20110222, "                       , "                                                       , "
%
% history:
% - 2013/06/23 - store data on external server; download when necessary.
%
% example:
% [hrf,tr,label] = getsamplehrf;
% figure; hold all;
% set(gca,'ColorOrder',jet(length(hrf)));
% for p=1:length(hrf)
%   plot(0:tr(p):tr(p)*(size(hrf{p},1)-1),hrf{p},'.-');
% end
% straightline(0,'h','k-');
% legend(label);
% xlabel('time from trial onset (s)'); ylabel('response (a.u.)');

% load
file0 = strrep(which('getsamplehrf'),'getsamplehrf.m','getsamplehrf.mat');
if ~exist(file0,'file')
  fprintf('Downloading %s (please be patient).\n',file0);
  urlwrite('http://kendrickkay.net/knkutils/getsamplehrf.mat',file0);
  fprintf('Downloading is done!\n');
end
aa = load(file0);

% deal with inputs
if ~exist('wh','var') || isempty(wh)
  wh = 1:length(aa.hrf);
end
if ~exist('takemean','var') || isempty(takemean)
  takemean = 1;
end

% do it
hrf = aa.hrf(wh);
if takemean
  hrf = cellfun(@(x)unitlength(mean(x,2)),hrf,'UniformOutput',0);
end
tr = catcell(2,aa.tr(wh));
label = aa.label(wh);
