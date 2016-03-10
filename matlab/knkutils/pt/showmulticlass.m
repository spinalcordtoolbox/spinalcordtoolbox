function [images,maskimages] = showmulticlass(outfile,offset,movieflip,frameduration,fixationinfo,fixationsize, ...
  triggerfun,ptonparams,soafun,skiptrials,images,setnum,isseq,grayval,iscolor, ...
  numrep,con,existingfile,dres,triggerkey,framefiles,trialparams,eyelinkfile,maskimages,specialoverlay, ...
  stimulusdir)

% function [images,maskimages] = showmulticlass(outfile,offset,movieflip,frameduration,fixationinfo,fixationsize, ...
%   triggerfun,ptonparams,soafun,skiptrials,images,setnum,isseq,grayval,iscolor, ...
%   numrep,con,existingfile,dres,triggerkey,framefiles,trialparams,eyelinkfile,maskimages,specialoverlay, ...
%   stimulusdir)
%
% <outfile> is the .mat file to save results to
% <offset> is horizontal and vertical offset for display purposes (see ptviewmovie.m)
% <movieflip> is flip to apply to movie (see ptviewmovie.m)
% <frameduration> is number of monitor refreshes for one movie frame (see ptviewmovie.m)
% <fixationinfo> is
%   {A B C} where A is the base color (uint8 1x3) for the fixation dot (see fixationcolor in ptviewmovie.m)
%                 B is alpha for the fixation dot in the default case
%                 C is alpha for the fixation dot when flips happen
%   {D E} where D is a set of colors (uint8 Nx3) (see the negative-integers case for fixationcolor in ptviewmovie.m)
%               E is an alpha value in [0,1]
%   {F} where F is the {A B C D E F G H} case of <fixationorder> in ptviewmovie.m
%     note that F should really only be {A B C D E F [] H} since we will add G if <existingfile> is supplied.
%   {G} where G is the {A C X} case of <fixationorder> in ptviewmovie.m.  actually, these
%     are just a subset of the inputs; the idea is that this function fills in the rest 
%     of the inputs for you.  if you supply <existingfile>, we will re-use the appropriate
%     values from that file.
% <fixationsize> is size in pixels for fixation dot or an entire alpha image for the fixation (see ptviewmovie.m)
% <triggerfun> is the trigger function to call when starting the movie (see ptviewmovie.m)
% <ptonparams> is a cell vector with parameters for pton.m
% <soafun> is a function that returns a (presumably stochastic) stimulus-onset asynchrony for the 
%   fixation dot (in number of movie frames).  the output must be a positive integer.
%   <soafun> is ignored when <fixationinfo> is {F} or {G}.
% <skiptrials> is number of trials to skip at the beginning
% <images> (optional) is a speed-up (no dependencies).  <images> can be reused only if <setnum> 
%   stays within [1 2 3 7 8] or [4 5] or [6] or [9] or [10 11] or [12 13] or [14] or [15] or
%                [16 17] or [18 19] or [20 21] or [22 23 23.5 24 25 26] or [27 28] or [29] or [30 31] or [32 33 35 36 37] or
%                [34] or [38 39 40 41  48] or [42 43 44] or [45] or [46 47] or [49 55] or [50] or [51 52] or 
%                [53 54] or [56 57 58] or [59 60 61] or [62 63 64 65] or [66] or [109 110] or [67 68 69 70 71 72] or
%                [73 74 75 76 77] or [78 79 80 81] or [82 83 84 85 86 87  88] or [89 90 91 92 93 94] or 
%                [95 96 97 98 99 100] or [101 102 103 104 105 106] or [107] or [108] or [111]
% <setnum> (optional) is
%   1 means the original 31 stimulus classes [15 frames, 3s / 3s]
%   2 means the horizontally-modulated random space stimuli plus small-scale checkerboard and letters [15 frames, 3s / 3s]
%   3 means the vertically-modulated random space stimuli plus photos and white noise [15 frames, 3s / 3s]
%   4 means the horizontally-modulated random space stimuli (WN case) plus the first four discs [10 frames, 1s / 3s]
%   5 means the vertically-modulated random space stimuli (WN case) plus the last three discs [10 frames, 1s / 3s]
%   6 means the original 31 stimulus classes plus the two classes we added (for version 2) [10 frames, 2s / 4s]
%   7 is like 2 except we add the first two discs instead of the calibration classes [15 frames, 3s / 3s]
%   8 is like 3 except we add the last two discs instead of the calibration classes [15 frames, 3s / 3s]
%   9 is the HRF stimuli [9 frames, .9s / .1s]
%   10 is like 4 except for a new experimental design [40 frames, 4s / 4s]
%   11 is like 5 except for a new experimental design [40 frames, 4s / 4s]
%   12 is like 10 except we change the texture to large-scale and multi-orientation and a new design [30 frames, 3s / 5s]
%   13 is like 11 except we change the texture to large-scale and multi-orientation and a new design [30 frames, 3s / 5s]
%   14 means iso color stimuli [8 frames, 4s / 4s]
%   15 means lum color stimuli "
%   16 is like 4 but with black letters on a light gray background and a new design: [15 frames, 3s / 5s]
%   17 is like 5 but with black letters on a light gray background plus a checkerboard and a new design: [15 frames, 3s / 5s]
%   18 is like 12 except that we scale texture with eccentricity and have contrast modulations instead of discs [30 frames, 3s / 5s]
%   19 is like 13 except that we scale texture with eccentricity and have contrast modulations instead of discs [30 frames, 3s / 5s]
%   20 is horizontal modulation, large-scale with noise degradation + contrast modulations + 5x5 grabbag [30 frames, 3s / 5s]
%   21 is vertical modulation, large-scale with noise degradation + contrast modulations + 5x5 grabbag [30 frames, 3s / 5s]
%   22 is like 20 except that 8 contrast modulations + 13x2 size invariance [30 frames, 3s / 5s]
%   23 is like 21 except that 8 contrast modulations + 13x2 size invariance  [30 frames, 3s / 5s]
%   23.5 is like 22 and 23 except that we include just the size invariance for objects (13 trials successive, no null)
%   24 is like 22 except that we omit contrast modulations and size invariance [30 frames, 3s / 5s]
%   25 is like 23 except that we omit contrast modulations and size invariance [30 frames, 3s / 5s]
%   26 is like 22 except that we do weird fixed and slow design (full, left, right) [30 frames, 3s / 25s]
%   27 is horizontal modulation, large-scale movement [8*11=88 frames (30 fps), 2.9s / 5.1s]
%   28 is vertical modulation, large-scale movement [8*11=88 frames (30 fps), 2.9s / 5.1s]
%   29 is carrier exploration [white noise, zebra [4 2 1 .5], pink noise, multiscale letters [dense 1 2 3 4], multiscale letters dense 3 [2 5 20]] [60 frames, 3s / 5s]
%   30 is horizontal modulation, slow color multiscale letters [9 frames, 3s / 5s]
%   31 is vertical modulation, slow color multiscale letters [9 frames, 3s / 5s]
%   32 is like 30 but denser and 800 x 800 resolution
%   33 is like 31 but denser and 800 x 800 resolution
%   34 is a weird random mixture of 20-23 that excludes contrast stimuli
%   35 is like 32 but 4/4 design (repeat the 9 frames into 12 randomly, ensuring no repeats)
%   36 is like 33 but 4/4 design (repeat the 9 frames into 12 randomly, ensuring no repeats) 
%   37 is special letter temporal summation thing
%   38 is the monster part 1
%   39 is the monster part 2
%   40 is the monster part 3
%   41 is the monster part 4
%   42 is monsterB (3/5 design, 9 frames, 800x800 resolution, 41 stim + 8 rest randomized).
%      you must use a PsychToolbox calibration for the third argument in <ptonparams>.
%   43 is like 42 except we use sparse3 instead of sparse2 for the fourth set
%   [44 N] is like 43 except that we do a single-frame strategy and N (positive integer) is used to 
%     cycle through sets of 9.  for a fixed N, the stimulus display is deterministic in what stimulus frames
%     are shown but the ordering of trials is still random.
%   45 is the subadd
%   46 is the monstertest part 1
%   47 is the monstertest part 2
%   48 is the monstersub (subset of stimuli from monster). summation of zebra, horizontal grating, vertical grating at horizontal line through upper visual field.
%   49 is the monstertestB
%   50 is the category
%   51 is the categoryC3 part 1.  <dres> should be [].
%   52 is the categoryC3 part 2.  <dres> should be [].
%   53 is the categoryC4 part 1.  <dres> should be [].
%   54 is the categoryC4 part 2.  <dres> should be [].
%   55 is the monstertestB but only the first 20 stimuli.
%   56 is the categoryC5 part 1.  <dres> should be [].
%   57 is the categoryC5 part 2.  <dres> should be [].
%   58 is the categoryC5 part 3.  <dres> should be [].
%   59 is the categoryC6 part 1.  <dres> should be [].
%   60 is the categoryC6 part 2.  <dres> should be [].
%   61 is the categoryC6 part 3.  <dres> should be [].
%   62 is the categoryC7 part 1.  <dres> should be [].
%   63 is the categoryC7 part 2.  <dres> should be [].
%   64 is the categoryC7 part 3.  <dres> should be [].
%   65 is the categoryC7 part 4.  <dres> should be [].
%   66 is the categoryC8.  <dres> should be [].
%   109 is the categoryC8 (trialtask version) part 1.  <dres> should be [].
%   110 is the categoryC8 (trialtask version) part 2.  <dres> should be [].
%   67 is the categoryC9 part 1a.
%   68 is the categoryC9 part 2a.
%   69 is the categoryC9 part 1b.
%   70 is the categoryC9 part 2b.
%   71 is the categoryC9 part 1c.
%   72 is the categoryC9 part 2c.
%   73 is retinotopy: b&w dartboard and color dartboard [CCW wedge]
%   74 is retinotopy: fast mashup and slow mashup [CCW wedge]
%   75 is retinotopy: wedges (CCW) and rings (expanding) [b&w dartboard]
%   76 is retinotopy: standard bar [R] and standard bar [U] [b&w dartboard]
%   77 is retinotopy: crazy bar [R] and crazy bar [U] [b&w dartboard]
%   78 is retinotopyB: wedges (CCW) and rings (expanding) [mashfast]
%   79 is retinotopyB: standard bar [R] and standard bar [U] [mashfast]
%   80 is reserved for something like 78 but with attention
%   81 is reserved for something like 79 but with attention
%   82,83,84,85,86,87 is reading (the six run types)
%   88 is reading ALT (one run type)
%   89 is retinotopyC: wedges CCW
%   90 is retinotopyC: wedges CW
%   91 is retinotopyC: rings expand
%   92 is retinotopyC: rings contract
%   93 is retinotopyC: multidirectional bars
%   94 is retinotopyC: wedges/rings mismash
%   95 is readingB
%   96 is readingB (short)
%   97 is readingB (short) T1
%   98 is readingB (short) T2
%   99 is readingB (short) T3
%   100 is readingBalt
%   101 is retinotopyCWORDS: multibars
%   102 is retinotopyCWORDS: wedge/ring mismash
%   103 is retinotopyCWORDS: multibars          (slow: 5-Hz refresh)
%   104 is retinotopyCWORDS: wedge/ring mismash (slow: 5-Hz refresh)
%   105 is retinotopyCWORDS: multibars          (slow: 5-Hz refresh + 5-Hz aperture refresh)
%   106 is retinotopyCWORDS: wedge/ring mismash (slow: 5-Hz refresh + 5-Hz aperture refresh)
%   107 is readingC
%   108 is readingD
%   109 is (see above)
%   110 is (see above)
%   111 is categoryC10
%   default: 1.
% <isseq> (optional) is whether to do the special sequential showing case.  should be either 0 which
%   means do nothing special, or a positive integer indicating which frame to use.  if a positive
%   integer, <fixationinfo> and <soafun> are ignored.  default: 0.
% <grayval> (optional) is the background color as uint8 1x1 or 1x3 (see ptviewmovie.m).  default: uint8(127).
% <iscolor> (optional) is whether to expect that the images are color.  default: 0.
% <numrep> (optional) is number of times to repeat the movie.  note that fixation stuff
%   is not repeated, but stochastically generated.  default: 1.
% <con> (optional) is the contrast in [0,100] to use (achieved via <moviemask> in ptviewmovie.m).
%   default: 100.  NOTE: this currently has a slow implementation (the initial setup time is long).
% <existingfile> (optional) is an old <outfile>.  if supplied, we pull the 'framedesign',
%   'classorder', 'fixationorder', 'trialoffsets', and 'digitrecord' variables from this old file 
%   instead of computing it fresh.  prior to May 13 2013, the trialtask and the digit-stream
%   were not preserved.  now, the 'trialoffsets' and 'digitrecord' means that they are preserved!
% <dres> (optional) is
%   [A B] where this is the desired resolution to imresize the images to (using bicubic interpolation).
%     if supplied, the imresize takes place immediately after loading the images in, and this imresized 
%     version is what is cached in the output of this function.
%  -C where C is the <scfactor> input in ptviewmovie.m
%   default is [] which means don't do anything special.  
% <triggerkey> (optional) is the input to ptviewmovie.m
% <framefiles> (optional) is the input to ptviewmovie.m
% <trialparams> (optional) is the {B E F G H} of the <trialtask> inputs
%   to ptviewmovie.m.  specify when <setnum> is 51 or 52 or 53 or 54 or 56,57,58, 59,60,61, 
%   62,63,64,65, 66,109,110  78,79
% <eyelinkfile> (optional) is the .edf file to save eyetracker data to.
%   default is [] which means to do not attempt to use the Eyelink.
% <maskimages> (optional) is a speed-up (no dependencies).  <maskimages> is applicable and can be reused only
%   if <setnum> stays within [73 74 75 76 77] or [78 79] or [89 90 91 92 93 94] or [101 102 103 104 105 106].
% <specialoverlay> (optional) is the input to ptviewmovie.m
% <stimulusdir> (optional) is the directory that contains the stimulus .mat files.
%   default to the parent directory of showmulticlass.m.
%
% show the stimulus and then save workspace (except the variable 'images') to <outfile>.

% history:
% 2015/01/25 - implement expt 111
% 2014/10/08 - implement the {G} case for <fixationinfo>
% 2014/07/15 - institute the input "stimulusdir" and release publicly.
% 2014/05/29 - use imresizememory to resample
% 2013/12/20 - add framecolor handling
% 2013/07/30 - add <specialoverlay>
% 2013/07/29 - new maskimages input and output (use it!); 
% 2013/05/17 - final tweaking of the eyelink eyetracking stuff!
% 2013/05/14 - now, trialoffsets and digitrecord is preserved with the existingfile mechanism.
%              so, in summary, existing 'trialoffsets' is shoved into 'trialtask' and
%              existing 'digitrecord' is shoved into 'fixationorder'.  this is potentially confusing!

%%%%%%%%%%%%% deal with <stimulusdir> up front

if ~exist('stimulusdir','var') || isempty(stimulusdir)
  stimulusdir = absolutepath(strrep(which('showmulticlass'),'showmulticlass.m',''));
end

%%%%%%%%%%%%% some constants

infofile = fullfile(stimulusdir,'multiclassinfo.mat');       % where the info.mat file is
infofile_monster = fullfile(stimulusdir,'multiclassinfo_monster.mat');
infofile_monstersub = fullfile(stimulusdir,'multiclassinfo_monstersub.mat');
infofile_monstertest = fullfile(stimulusdir,'multiclassinfo_monstertest.mat');
infofile_monstertestB = fullfile(stimulusdir,'multiclassinfo_monstertestB.mat');
infofile_monstertestBALT = fullfile(stimulusdir,'multiclassinfo_monstertestBALT.mat');
infofile_monsterB = fullfile(stimulusdir,'multiclassinfo_monsterB.mat');
infofile_category = fullfile(stimulusdir,'multiclassinfo_category.mat');
infofile_categoryC3 = fullfile(stimulusdir,'multiclassinfo_categoryC3.mat');
infofile_categoryC4 = fullfile(stimulusdir,'multiclassinfo_categoryC4.mat');
infofile_categoryC5 = fullfile(stimulusdir,'multiclassinfo_categoryC5.mat');
infofile_categoryC6 = fullfile(stimulusdir,'multiclassinfo_categoryC6.mat');
infofile_categoryC7 = fullfile(stimulusdir,'multiclassinfo_categoryC7.mat');
infofile_categoryC8 = fullfile(stimulusdir,'multiclassinfo_categoryC8.mat');
infofile_categoryC9 = fullfile(stimulusdir,'multiclassinfo_categoryC9.mat');
infofile_readingALT = fullfile(stimulusdir,'multiclassinfo_readingALT.mat');
infofile_readingB = fullfile(stimulusdir,'multiclassinfo_readingB.mat');
infofile_readingBalt = fullfile(stimulusdir,'multiclassinfo_readingBalt.mat');
infofile_readingBshort = fullfile(stimulusdir,'multiclassinfo_readingBshort.mat');
infofile_readingBshortT1 = fullfile(stimulusdir,'multiclassinfo_readingBshortT1.mat');
infofile_readingBshortT2 = fullfile(stimulusdir,'multiclassinfo_readingBshortT2.mat');
infofile_readingBshortT3 = fullfile(stimulusdir,'multiclassinfo_readingBshortT3.mat');
infofile_readingD = fullfile(stimulusdir,'multiclassinfo_readingD.mat');
infofile_subadd = fullfile(stimulusdir,'multiclassinfo_subadd.mat');
infofileB = fullfile(stimulusdir,'multiclassinfoB.mat');
infofile_wn = fullfile(stimulusdir,'multiclassinfo_wn.mat');
infofile_wnB = fullfile(stimulusdir,'multiclassinfo_wnB.mat');
infofile_wnB2 = fullfile(stimulusdir,'multiclassinfo_wnB2.mat');
infofile_wnB3 = fullfile(stimulusdir,'multiclassinfo_wnB3.mat');
infofile_wnB4 = fullfile(stimulusdir,'multiclassinfo_wnB4.mat');
infofile_wnB5 = fullfile(stimulusdir,'multiclassinfo_wnB5.mat');
infofile_wnB6 = fullfile(stimulusdir,'multiclassinfo_wnB6.mat');
infofile_wnB7 = fullfile(stimulusdir,'multiclassinfo_wnB7.mat');
infofile_wnB8 = fullfile(stimulusdir,'multiclassinfo_wnB8.mat');
infofile_wnD = fullfile(stimulusdir,'multiclassinfo_wnD.mat');
infofile_version2 = fullfile(stimulusdir,'multiclassinfo_version2.mat');
infofile_hrf = fullfile(stimulusdir,'multiclassinfo_hrf.mat');
switch setnum(1)
case {1 2 3 6 7 8}
  stimfile = fullfile(stimulusdir,'workspace.mat');  % where the 'images' variables can be obtained
case {82 83 84 85 86 87  88}
  stimfile = fullfile(stimulusdir,'workspace_reading.mat');
case {4 5}
  stimfile = fullfile(stimulusdir,'workspace_wn.mat');
case {50}
  stimfile = fullfile(stimulusdir,'workspace_category.mat');
case {51 52}
  stimfile = fullfile(stimulusdir,'workspace_categoryC3.mat');
case {53 54}
  stimfile = fullfile(stimulusdir,'workspace_categoryC4.mat');
case {56 57 58}
  stimfile = fullfile(stimulusdir,'workspace_categoryC5.mat');
case {62 63 64 65}
  stimfile = fullfile(stimulusdir,'workspace_categoryC7.mat');
case {66 109 110}
  stimfile = fullfile(stimulusdir,'workspace_categoryC8.mat');
case {111}
  stimfile = fullfile(stimulusdir,'workspace_categoryC10.mat');
case {95 96 97 98 99 100}
  stimfile = fullfile(stimulusdir,'workspace_readingB.mat');
case {107}
  stimfile = fullfile(stimulusdir,'workspace_readingC.mat');
case {108}
  stimfile = fullfile(stimulusdir,'workspace_readingD.mat');
case {67 68 69 70 71 72}
  stimfile = fullfile(stimulusdir,'workspace_categoryC9.mat');
case {59 60 61}
  stimfile = fullfile(stimulusdir,'workspace_categoryC6.mat');
case {9}
  stimfile = fullfile(stimulusdir,'workspace_hrf.mat');
case {10 11}
  stimfile = fullfile(stimulusdir,'workspace_wnB.mat');
case {12 13}
  stimfile = fullfile(stimulusdir,'workspace_wnC.mat');
case {14}
  stimfile = fullfile(stimulusdir,'workspace_wnD.mat');
case {15}
  stimfile = fullfile(stimulusdir,'workspace_wnD2.mat');
case {16 17}
  stimfile = fullfile(stimulusdir,'workspace_wnE.mat');
case {18 19}
  stimfile = fullfile(stimulusdir,'workspace_wnC2.mat');
case {20 21}
  stimfile = fullfile(stimulusdir,'workspace_wnC3.mat');
case {22 23 23.5 24 25 26}
  stimfile = fullfile(stimulusdir,'workspace_wnC5.mat');
case {27 28}
  stimfile = fullfile(stimulusdir,'workspace_wnC6.mat');
case {29}
  stimfile = fullfile(stimulusdir,'workspace_wnC7.mat');
case {30 31}
  stimfile = fullfile(stimulusdir,'workspace_wnC8.mat');
case {32 33 35 36 37}
  stimfile = fullfile(stimulusdir,'workspace_wnC9.mat');
case {34}
  stimfile = fullfile(stimulusdir,'workspace_wnC35combined.mat');
case {38 39 40 41 48}
  stimfile = fullfile(stimulusdir,'workspace_monster.mat');
case {42 43 44}
  stimfile = fullfile(stimulusdir,'workspace_monsterB.mat');
case {45}
  stimfile = fullfile(stimulusdir,'workspace_subadd.mat');
case {46 47}
  stimfile = fullfile(stimulusdir,'workspace_monstertest.mat');
case {49 55}
  stimfile = fullfile(stimulusdir,'workspace_monstertestB.mat');
case {73 74 75 76 77}
  stimfile = fullfile(stimulusdir,'workspace_retinotopy.mat');
case {78 79}
  stimfile = fullfile(stimulusdir,'workspace_retinotopyB.mat');
case {89 90 91 92 93 94}
  stimfile = fullfile(stimulusdir,'workspace_retinotopyCaltsmash.mat');
case {101 102 103 104 105 106}
  stimfile = fullfile(stimulusdir,'workspace_retinotopyCaltsmashWORDS.mat');
end
stimfileextra = fullfile(stimulusdir,'workspace_convert10.mat');

%%%%%%%%%%%%% input

if ~exist('setnum','var') || isempty(setnum)
  setnum = 1;
end
if ~exist('isseq','var') || isempty(isseq)
  isseq = 0;
end
if ~exist('grayval','var') || isempty(grayval)
  grayval = uint8(127);
end
if ~exist('iscolor','var') || isempty(iscolor)
  iscolor = 0;
end
if ~exist('numrep','var') || isempty(numrep)
  numrep = 1;
end
if ~exist('con','var') || isempty(con)
  con = 100;
end
if ~exist('existingfile','var') || isempty(existingfile)
  existingfile = [];
end
if ~exist('dres','var') || isempty(dres)
  dres = [];
end
if ~exist('triggerkey','var') || isempty(triggerkey)
  triggerkey = [];
end
if ~exist('framefiles','var') || isempty(framefiles)
  framefiles = [];
end
if ~exist('trialparams','var') || isempty(trialparams)
  trialparams = [];
end
if ~exist('eyelinkfile','var') || isempty(eyelinkfile)
  eyelinkfile = [];
end
if ~exist('maskimages','var') || isempty(maskimages)
  maskimages = [];
end
if ~exist('specialoverlay','var') || isempty(specialoverlay)
  specialoverlay = [];
end
if ~isempty(existingfile)
  efile = load(existingfile,'framedesign','classorder','fixationorder','trialoffsets','digitrecord');
end

%%%%%%%%%%%%% some experiments need some pre-setup

switch setnum(1)
case {109 110 111}
  if isempty(existingfile)
    [mastercuestim,digitnamerecord,digitcolorrecord,gentrialpattern,designmatrix] = setupmulticlassfun(setnum(1));
  else
    load(existingfile,'mastercuestim','digitnamerecord','digitcolorrecord','gentrialpattern','designmatrix');
  end
end

%%%%%%%%%%%%% load in the stimuli

if ~exist('images','var') || isempty(images)

  % load images
  load(stimfile,'images','maskimages');
  if ~exist('maskimages','var')
    maskimages = {};
  end
  if setnum(1)==6  % in this case, we have to use the newer versions of certain classes
    images0 = loadmulti(stimfileextra,'images');
    wh = cellfun(@(x) ~isempty(x),images0);
    images(wh) = images0(wh);
    clear images0;
  end
  
  % resize if desired
  if ~isempty(dres) && length(dres)==2
    tic;
    fprintf('resampling the stimuli; this may take a while');
    for p=1:length(images)
      statusdots(p,length(images));
      if iscolor
        imtemp = imresizememory(images{p},dres,4);
        images{p} = [];
        images{p} = imresizememory(imtemp);
      else
        imtemp = imresizememory(images{p},dres,3);
        images{p} = [];
        images{p} = imresizememory(imtemp);
      end
    end
    if iscell(maskimages)
      for p=1:length(maskimages)
        imtemp = imresizememory(maskimages{p},dres,3);
        maskimages{p} = [];
        maskimages{p} = imresizememory(imtemp);
      end
    else
      imtemp = imresizememory(maskimages,dres,3);
      maskimages = [];
      maskimages = imresizememory(imtemp);
    end
    fprintf('done!\n');
    toc
  end

end
numinclass = cellfun(@(x) size(x,choose(iscolor,4,3)),images);  % a vector with number of images in each class

%%%%%%%%%%%%% perform run-specific randomizations (NOTE THAT THERE ARE HARD-CODED CONSTANTS IN HERE)

% load in some aux info
switch setnum(1)
case {6}
  load(infofile_version2,'lettersix','numbersix','polygonsix');
case {1 2 3 4 5 7 8}
  load(infofile,'lettersix','numbersix','polygonsix');
end

% figure out frame assignment for each class
if ~isempty(existingfile)
  framedesign = efile.framedesign;
else
  switch setnum(1)
  case {1 2 3 7 8}  % note that this section used to go up to 69.  we extended it to 73 to take care of setnum being 7 or 8.
    if isseq
      framedesign = {};
      for p=1:73
        framedesign{p} = isseq;
      end
    else
      framedesign = {};
      framedesign{1} = reshape(permutedim(1:30),2,[]);
      framedesign{2} = [permutedim(1:15); permutedim(1:15)];
      framedesign{3} = framedesign{2};
      framedesign{4} = framedesign{2};
      framedesign{5} = framedesign{2};
      framedesign{6} = [permutedim(1:15); permutedim(1:15)];
      framedesign{7} = reshape(permutedim(1:30),2,[]);
      framedesign{8} = reshape(permutedim(1:30),2,[]);
      framedesign{9} = framedesign{7};
      framedesign{10} = framedesign{8};
      framedesign{11} = framedesign{7};
      framedesign{12} = framedesign{8};
      framedesign{13} = [permutedim(1:15); permutedim(1:15)];
      framedesign{14} = [permutedim(1:15); permutedim(1:15)];
      framedesign{15} = [permutedim(1:15); permutedim(1:15)];
      framedesign{16} = [permutedim(1:15); permutedim(1:15)];
      framedesign{17} = [permutedim(1:15); permutedim(1:15)];
      framedesign{18} = repeatuntil(@() reshape(permutedim(lettersix),2,[]),@(x) all(flatten(diff(x,1,2))~=0));
      framedesign{19} = repeatuntil(@() reshape(permutedim(numbersix),2,[]),@(x) all(flatten(diff(x,1,2))~=0));
      framedesign{20} = repeatuntil(@() reshape(permutedim(polygonsix),2,[]),@(x) all(flatten(diff(x,1,2))~=0));
      framedesign{21} = reshape(permutedim(1:30),2,[]);
      framedesign{22} = reshape(permutedim(1:30),2,[]);
      framedesign{23} = reshape(permutedim(1:30),2,[]);
      framedesign{24} = reshape(permutedim(1:30),2,[]);
      framedesign{25} = reshape(permutedim(1:30),2,[]);
      framedesign{26} = reshape(permutedim(1:30),2,[]);
      framedesign{27} = reshape(permutedim(1:30),2,[]);
      framedesign{28} = framedesign{27};
      framedesign{29} = framedesign{27};
      framedesign{30} = framedesign{27};
      framedesign{31} = framedesign{27};
      for p=32:73
        framedesign{p} = [permutedim(1:15); permutedim(1:15)];
      end
    end
  case {6}
    if isseq
      framedesign = {};
      for p=1:75
        framedesign{p} = isseq;
      end
    else
      framedesign = {};
      framedesign{1} = reshape(permutedim(1:20),2,[]);
      framedesign{2} = [permutedim(1:10); permutedim(1:10)];
      framedesign{3} = framedesign{2};
      framedesign{4} = framedesign{2};
      framedesign{5} = framedesign{2};
      framedesign{6} = [permutedim(1:10); permutedim(1:10)];
      framedesign{7} = reshape(permutedim(1:20),2,[]);
      framedesign{8} = reshape(permutedim(1:20),2,[]);
      framedesign{9} = framedesign{7};
      framedesign{10} = framedesign{8};
      framedesign{11} = framedesign{7};
      framedesign{12} = framedesign{8};
      framedesign{13} = [permutedim(1:10); permutedim(1:10)];
      framedesign{14} = [permutedim(1:10); permutedim(1:10)];
      framedesign{15} = [permutedim(1:10); permutedim(1:10)];
      framedesign{16} = [permutedim(1:10); permutedim(1:10)];
      framedesign{17} = [permutedim(1:10); permutedim(1:10)];
      framedesign{18} = repeatuntil(@() reshape(permutedim(lettersix),2,[]),@(x) all(flatten(diff(x,1,2))~=0));
      framedesign{19} = repeatuntil(@() reshape(permutedim(numbersix),2,[]),@(x) all(flatten(diff(x,1,2))~=0));
      framedesign{20} = repeatuntil(@() reshape(permutedim(polygonsix),2,[]),@(x) all(flatten(diff(x,1,2))~=0));
      framedesign{21} = reshape(permutedim(1:20),2,[]);
      framedesign{22} = reshape(permutedim(1:20),2,[]);
      framedesign{23} = reshape(permutedim(1:20),2,[]);
      framedesign{24} = reshape(permutedim(1:20),2,[]);
      framedesign{25} = reshape(permutedim(1:20),2,[]);
      framedesign{26} = reshape(permutedim(1:20),2,[]);
      framedesign{27} = reshape(permutedim(1:20),2,[]);
      framedesign{28} = framedesign{27};
      framedesign{29} = framedesign{27};
      framedesign{30} = framedesign{27};
      framedesign{31} = framedesign{27};
      for p=32:73
        framedesign{p} = [permutedim(1:15); permutedim(1:15)];
      end
      framedesign{74} = [permutedim(1:10); permutedim(1:10)];
      framedesign{75} = framedesign{20};
    end
  case {4 5}
    if isseq
      framedesign = {};
      for p=1:69
        framedesign{p} = isseq;
      end
    else
      framedesign = {};
      for p=1:69
        framedesign{p} = [permutedim(1:10); permutedim(1:10)];
      end
    end
  case {50}
    if isseq
      framedesign = {};
      for p=1:72+14
        framedesign{p} = isseq;
      end
    else

      framedesign = {};
      for p=1:72+14
        framedesign{p} = subscript(permutedim(1:20),1:7);
      end

%       framedesign = {};
%       for p=1:79
%         temp = [];
%         for q=1:9
%           temp = [temp (q-1)*5 + ceil(rand*5)];
%         end
%         framedesign{p} = permutedim(temp);
%       end

%       framedesign = {};
%       for p=1:79
%         framedesign{p} = picksubset(1:50,8,sum(100*clock));
%       end

    end
  case {10 11}
    if isseq
      framedesign = {};
      for p=1:69
        framedesign{p} = isseq;
      end
    else
      framedesign = {};
      for p=1:69
        framedesign{p} = [permutedim(1:40)];
      end
    end
  case {12 13}
    if isseq
      framedesign = {};
      for p=1:69
        framedesign{p} = isseq;
      end
    else
      framedesign = {};
      for p=1:69
        framedesign{p} = [permutedim(1:30)];
      end
    end
  case {30 31 32 33}
    if isseq
      framedesign = {};
      for p=1:69
        framedesign{p} = isseq;
      end
    else
      framedesign = {};
      for p=1:69
        framedesign{p} = [permutedim(1:9)];
      end
    end
  case {37}
    if isseq
      framedesign = {};
      for p=[16]
        framedesign{p} = isseq;
      end
    else
      framedesign = {};
      framedesign{16} = [];
      
      % should be made into function
      ord = []; prev = NaN;
      for ff=1:10000
        while 1
          tryit = ceil(rand*9);
          if ~isequal(prev,tryit)
            break;
          end
        end
        ord(ff) = tryit;
        prev = tryit;
      end
      % ord is 1 x 10000 of non-repeated frame indices (1 through 9)
      
      cnt = 0;
      for qq=1:1000
        framedesign{16} = [framedesign{16}; ord(cnt+(1:3))];
        cnt = cnt + 3;
      end
      % framedesign has 1000 different 3-frame chunks

    end
  case {35 36}
    if isseq
      framedesign = {};
      for p=1:69
        framedesign{p} = isseq;
      end
    else
      framedesign = {};
      for p=1:69
        while 1
          temp = permutedim([1:9 1:9]);
          temp = temp(1:12);
          if ~any(diff(temp)==0)
            break;
          end
        end
        framedesign{p} = temp;
      end
    end
  case {18 19}
    if isseq
      framedesign = {};
      for p=1:70
        framedesign{p} = isseq;
      end
    else
      framedesign = {};
      for p=1:70
        framedesign{p} = [permutedim(1:30)];
      end
    end
  case {20 21 22 23 23.5 26 34}
    if isseq
      framedesign = {};
      for p=1:72
        framedesign{p} = isseq;  %%[permutedim(1:30)]
      end
    else
      framedesign = {};
      for p=1:72
        framedesign{p} = [permutedim(1:30)];
      end
    end
  case {24 25}
    if isseq
      framedesign = {};
      for p=1:38
        framedesign{p} = isseq;
      end
    else
      framedesign = {};
      for p=1:38
        framedesign{p} = [permutedim(1:30)];
      end
    end
  case {27 28}
    if isseq
      framedesign = {};
      for p=1:70
        framedesign{p} = isseq;
      end
    else
      framedesign = {};
      for p=1:70
        framedesign{p} = flatten(permutedim(reshape(1:88,[11 8]),2));
      end
    end
  case {82 83 84 85 86 87}
    if isseq
      framedesign = {};
      for p=1:45
        framedesign{p} = isseq;
      end
    else

      % CAREFUL, THIS IS QUITE TRICKY!
      % the assignment of frames to the six run types is deterministic.
      setrandstate(0);
      
      framedesign = {};
      for p=1:45
        ordframes = randperm(12);
        framedesign{p} = repmat(vflatten(ordframes(((setnum-81)-1)*2 + (1:2))),[1 3]);  % 3 for the flashing
      end

      % Important: ensure that randomness is obtained again.
      setrandstate;

    end
  case {88}
    if isseq
      framedesign = {};
      for p=1:45
        framedesign{p} = isseq;
      end
    else
      
      % define
      targetprop = 0.5;
      
      % calc
      framedesign = {};
      framecolordesign = {};
      for p=1:45
      
        while 1

          while 1
            if ismember(p,[19 34:45])
              allframes = 1:12;  %repmat([1 7],[1 50]);
            else
              allframes = 1:12;
            end
            temp0 = subscript(permutedim(allframes),1:5);
            if ~any(diff(temp0)==0)  % this is moot now, oh well
              break;
            end
          end
        
          % mangle on some trials
          if rand < targetprop
%             repn = 1+ceil(4*rand);  % the repeat frame number is in [2,5]
%             temp0(repn) = temp0(repn-1);
%             if sum(diff(temp0)==0)==1
%               break;
%             end
            temp1 = permutedim([1 1 1 1 0.5]);  % one of the frames is 50% alpha
            break;
          else
            temp1 = [1 1 1 1 1];
            break;
          end

        end
        
        % record
%%        temp0 = upsamplematrix(temp0,[1 4],2,0,'nearest');
        framedesign{p} = temp0;
        framecolordesign{p} = temp1;

      end

    end
  case {29}
    framespecial = [1:10 11 11 11];
    if isseq
      framedesign = {};
      for p=1:13
        framedesign{p} = isseq;
      end
    else
      framedesign = {};
      for p=1:10
        framedesign{p} = upsamplematrix(permutedim(1:30),[1 2],[],[],'nearest');
      end
      xxx = [2 5 20];  % frames per second
      for p=1:length(xxx)
        framedesign{10+p} = resampleup(permutedim(picksubset(1:60,3*xxx(p))),[1 20/xxx(p)]);
      end
    end
  case {16 17}
    if isseq
      framedesign = {};
      for p=1:70
        framedesign{p} = isseq;
      end
    else
      framedesign = {};
      for p=1:70
        framedesign{p} = [permutedim(1:15)];
      end
    end
  case {14 15}
    if isseq
      framedesign = {};
      for p=1:24
        framedesign{p} = isseq;
      end
    else
      framedesign = {};
      temp = upsamplematrix(permutedim(1:8),[1 4],2,0,'nearest');
      for p=1:24
        framedesign{p} = temp;
      end
    end
  case {9}
    if isseq
      framedesign = {};
      for p=1:1
        framedesign{p} = isseq;
      end
    else
      framedesign = {};
      for p=1:1
        framedesign{p} = reshape(permutedim(1:16*9),16,[]);
      end
    end
  case {38 39 40 41}
    if isseq
      framedesign = {};
      for p=1:156
        framedesign{p} = isseq;
      end
    else
      framedesign = {};
      for p=1:156
        framedesign{p} = permutedim(1:9);
      end
      framedesign{105} = repeatuntil(@() subscript(permutedim(repmat(1:7,[1 2])),1:9),@(x) all(diff(x,1,2)~=0));
    end
  case {51 52}
    if isseq
      framedesign = {};
      for p=1:104
        framedesign{p} = isseq;
      end
    else
      framedesign = {};
      for p=1:104
        framedesign{p} = permutedim(1:7);
      end
    end
  case {53 54}
    if isseq
      framedesign = {};
      for p=1:98
        framedesign{p} = isseq;
      end
    else
      framedesign = {};
      for p=1:98
        framedesign{p} = permutedim(1:7);
      end
    end
  case {56 57 58}
    if isseq
      framedesign = {};
      for p=1:49*3
        framedesign{p} = isseq;
      end
    else
      framedesign = {};
      for p=1:49*3
        framedesign{p} = subscript(permutedim(1:20),1:7);
      end
    end
  case {62 63 64 65}
    if isseq
      framedesign = {};
      for p=1:49*4
        framedesign{p} = isseq;
      end
    else
      framedesign = {};
      for p=1:49*4
        framedesign{p} = subscript(permutedim(1:20),1:7);
      end
    end
  case {66 109 110}
    if isseq
      framedesign = {};
      for p=1:25
        framedesign{p} = isseq;
      end
    else
    
      switch setnum(1)
      case 66
        numpresrun = 2;
      case {109 110}
        numpresrun = 3;
      end
      
      % define
      numviewpoints = 7;
      numids = 95;
      targetprop = 0.5;      

      % calc
      framedesign = {};
      for p=1:25
        framedesign{p} = [];

        % there are several presentations in each run
        for zz=1:numpresrun
        
          % generate a sequence of viewpoint numbers.  viewpoint always changes.
          while 1
            vpnums = ceil(numviewpoints*rand(1,7));
            if ~any(diff(vpnums)==0)
              break;
            end
          end
        
          % generate a sequence of identity numbers.  no repeats at all.
          idnums = subscript(permutedim(1:numids),1:7);

          % decide if this is a target trial.  if so, repeat an identity
          if rand < targetprop
            repn = 1+ceil(6*rand);  % the repeat frame number is in [2,7]
            idnums(repn) = idnums(repn-1);
          end
          
          % record
          framedesign{p}(zz,:) = upsamplematrix((idnums-1)*numviewpoints + vpnums,[1 2],[],[],'nearest');
          
          % special case (REPEAT PHYSICALLY IDENTICAL)
          % thus, it is here that physicality of face sequences is enforced (all three tasks see the same faces).
          if ismember(setnum(1),[109 110])
            framedesign{p} = repmat(framedesign{p}(zz,:),[numpresrun 1]);
            break;
          end

        end
      end
    end
  case {111}
    if isseq
      framedesign = {};
      for p=1:15
        framedesign{p} = isseq;
      end
    else
    
      % define
      numpresrun = 3;
      numviewpoints = 7;
      numids = 95;
      targetprop1 = 0.5;      % phscr insert?
      targetprop2 = 0.5;      % one-back identity repeat?
      offsetph = numviewpoints*numids;  % offset due to the phscr images
      maxuniqueface = 4;      % number of maximum faces presented on a trial
      numfaceframes = 16;     % out of 20 frames, how long is a given face on for?

      % calc
      framedesign = {};
      for p=1:15
        framedesign{p} = [];

        % there are several presentations in each run
        for zz=1:numpresrun
        
          % should we insert phscr?
          if rand < targetprop1
            phspot = randintrange(2,maxuniqueface);
            numfacegen = maxuniqueface-1;
          else
            phspot = [];
            numfacegen = maxuniqueface;
          end
        
          % generate a sequence of viewpoint numbers.  viewpoint always changes.
          while 1
            vpnums = ceil(numviewpoints*rand(1,numfacegen));
            if ~any(diff(vpnums)==0)
              break;
            end
          end
        
          % generate a sequence of identity numbers.  no repeats at all.
          idnums = subscript(permutedim(1:numids),1:numfacegen);

          % decide if this is a target trial.  if so, repeat an identity
          if rand < targetprop2
            repn = randintrange(2,numfacegen);  % the repeat frame number (relative to the face slots)
            idnums(repn) = idnums(repn-1);
          end
          
          % record
          temp = insertelt(offsetph + ((idnums-1)*numviewpoints + vpnums),phspot,randintrange(1,offsetph));
          framedesign{p}(zz,:) = upsamplematrix(temp,[1 numfaceframes],[],[],'nearest');
          
          % special case (REPEAT PHYSICALLY IDENTICAL)
          % thus, it is here that physicality of face sequences is enforced (all three tasks see the same faces).
          if ismember(setnum(1),[111])
            framedesign{p} = repmat(framedesign{p}(zz,:),[numpresrun 1]);
            break;
          end

        end
      end
    end
  case {95}
    if isseq
      framedesign = {};
      for p=1:32
        framedesign{p} = isseq;
      end
    else
      framedesign = {};
      for p=1:32
        temp = reshape(permutedim(1:10),2,[]);  % 10 frames random order
        ix1 = randintrange(1,2);
        ix2 = randintrange(2,5);
        temp(ix1,ix2) = temp(ix1,ix2-1);  % one of the presentations has a repeat
        framedesign{p} = upsamplematrix(temp,[1 2],[],[],'nearest');
      end
    end
  case {100 107}
    if isseq
      framedesign = {};
      for p=1:32
        framedesign{p} = isseq;
      end
    else
      framedesign = {};
      for p=1:32
        temp = reshape(subscript(permutedim(1:10),{1:8}),2,[]);  % 8 frames (but whittle to 7 below)
        ix1 = randintrange(1,2);
        ix2 = randintrange(2,4);
        temp(ix1,ix2) = temp(ix1,ix2-1);  % one of the presentations has a repeat
        framedesign{p} = upsamplematrix(temp,[1 4],[],[],'nearest');
      end
    end
  case {108}
    if isseq
      framedesign = {};
      for p=1:24
        framedesign{p} = isseq;
      end
    else
      framedesign = {};
      for p=1:24
        temp = reshape(1:12,[3 4]);  % skeleton index
        ixout = randintrange(1,3);   % one of the three won't have a repeat
        ixin = setdiff(1:3,ixout);   % the other two will
        for bbb=1:length(ixin)
          ix2 = randintrange(2,4);
          temp(ixin(bbb),ix2) = temp(ixin(bbb),ix2-1);  % insert the repeat
        end
        assert(length(union(temp(:),[]))==10);  % exactly 10 distinct frames
        temp = reshape(calcposition(permutedim(flatten(union(temp(:),[]))),flatten(temp)),[3 4]);  % magic
        assert(isequal(1:10,flatten(union(temp(:),[]))));  % check that we are using only 10 frames
        framedesign{p} = upsamplematrix(temp,[1 4],[],[],'nearest');
      end
    end
  case {96 97 98 99}
    if isseq
      framedesign = {};
      for p=1:32
        framedesign{p} = isseq;
      end
    else
      switch setnum
      case 96
        nftochoose = 5;
        nupsam = 2;
      case 97
        nftochoose = 9;
        nupsam = 1;
      case 98
        nftochoose = 4;
        nupsam = 4;
      case 99
        nftochoose = 4;
        nupsam = 1;
      end
      framedesign = {};
      for p=1:32
        temp = subscript(permutedim(1:10),{1:nftochoose});  % just choose N distinct frames
        if rand < .5
          ix = randintrange(2,nftochoose);
          temp(ix) = temp(ix-1);  % insert repeat
        end
        framedesign{p} = upsamplematrix(temp,[1 nupsam],[],[],'nearest');
      end
    end
  case {67 68 69 70 71 72}
    if isseq
      framedesign = {};
      for p=1:81
        framedesign{p} = isseq;
      end
    else
      % CAREFUL, THIS IS QUITE TRICKY!
      setrandstate(0);
      totalset = subscript(permutedim(1:92),{1:60});  % choose a random 60 (but deterministic)
      totalset = [24 92 31 8 42 21 91 25 55 85 15 50 51 27 30 39 26 2 88 87 29 68 53 49 72 69 67 22 75 57 44 16 19 36 48 10 78 33 7 35 4 61 46 83 38 70 54 77 74 60 28 3 11 80 64 66 40 43 52 79];  % just hard code it!
      framedesign = {};
      for p=1:81
          % so, each stimulus has a random order of the 60, and this is
          % not dependent on the setnum
        totalset = permutedim(totalset);
          % this relies on the fact that successive pairs of runs are non-overlapping in what stimuli are selected:
        off0 = floor((setnum-67)/2);  % 0 0 1 1 2 2
        framedesign{p} = subscript(totalset,off0*20+(1:20));
      end
      setrandstate;  % Important: ensure that randomness is obtained again.
    end
  case {59 60 61}
    if isseq
      framedesign = {};
      for p=1:165
        framedesign{p} = isseq;
      end
    else
      framedesign = {};
      for p=1:165
        framedesign{p} = subscript(permutedim(1:20),1:7);
      end
    end
  case {48}
    if isseq
      framedesign = {};
      for p=1:156  % but only 9 shown
        framedesign{p} = isseq;
      end
    else
      framedesign = {};
      for p=1:156  % but only 9 shown
        framedesign{p} = [permutedim(1:9); permutedim(1:9)];
      end
    end
  case {46 47}
    if isseq
      framedesign = {};
      for p=1:64
        framedesign{p} = isseq;
      end
    else
      framedesign = {};
      for p=1:5
        framedesign{p} = permutedim(1:9);
      end
      for p=6:64
        framedesign{p} = ones(1,9);
      end
    end
  case {49}
    if isseq
      framedesign = {};
      for p=1:35
        framedesign{p} = isseq;
      end
    else
      framedesign = {};
      for p=1:35
        framedesign{p} = ones(1,9);
      end
    end
  case {55}
    if isseq
      framedesign = {};
      for p=1:20
        framedesign{p} = isseq;
      end
    else
      framedesign = {};
      for p=1:20
        framedesign{p} = ones(1,9);
      end
    end
  case {42 43}
    if isseq
      framedesign = {};
      for p=1:50
        framedesign{p} = isseq;
      end
    else
      framedesign = {};
      for p=1:50
        framedesign{p} = permutedim(1:9);
      end
    end
  case {45}
    if isseq
      framedesign = {};
      for p=1:30
        framedesign{p} = isseq;
      end
    else
      framedesign = {};
      for p=1:18
        framedesign{p} = permutedim(1:30);
      end
      for p=19:30
        framedesign{p} = 1:30;
      end
    end
  case {44}
    if isseq
      framedesign = {};
      for p=1:50
        framedesign{p} = isseq;
      end
    else
      % RUN ONCE:
      %stimrec = permutedim(repmat((1:9)',[1 50]),1,[],1);
      stimrec = [4 7 5 2 5 8 4 3 5 6 8 1 1 5 1 7 7 5 5 1 1 9 8 8 9 1 2 5 5 1 5 2 4 1 8 4 7 2 9 3 6 8 8 2 4 7 8 2 9 7;6 2 7 9 3 2 1 1 7 4 5 2 9 1 8 4 3 8 9 6 5 2 7 1 7 3 4 8 8 2 3 5 8 2 6 7 3 7 7 7 7 1 9 4 2 3 1 8 1 5;5 3 2 1 9 9 6 6 4 1 1 7 7 8 4 1 8 4 7 3 3 3 9 5 1 7 6 9 6 6 9 3 9 8 3 8 8 9 1 1 8 9 6 6 7 5 6 7 7 1;3 6 6 3 1 1 2 9 2 3 7 9 2 9 3 9 9 1 2 9 4 1 2 9 8 6 3 4 4 7 2 9 7 7 4 6 9 4 4 4 1 4 3 8 9 8 5 3 6 9;8 8 4 4 8 6 8 5 6 8 6 3 8 4 5 6 6 6 6 5 7 4 3 7 6 9 1 7 1 3 1 7 5 9 2 5 2 8 5 8 9 5 1 1 3 9 9 1 8 3;2 1 8 8 6 5 7 7 1 7 2 6 6 7 2 5 4 7 4 4 6 7 6 6 5 4 5 1 3 8 4 6 3 3 9 1 4 1 6 9 3 7 4 5 1 2 7 6 4 4;7 5 3 7 2 3 5 2 3 5 4 4 5 6 9 2 2 3 3 7 2 6 1 4 2 8 8 3 7 9 6 1 6 5 7 3 5 3 3 6 2 2 2 9 5 6 2 4 2 6;9 4 1 6 7 7 9 4 8 2 9 8 4 2 7 3 5 9 8 2 8 8 5 2 3 2 7 6 9 5 8 4 1 4 5 2 6 5 8 5 5 6 5 7 8 4 3 9 3 2;1 9 9 5 4 4 3 8 9 9 3 5 3 3 6 8 1 2 1 8 9 5 4 3 4 5 9 2 2 4 7 8 2 6 1 9 1 6 2 2 4 3 7 3 6 1 4 5 5 8];
      framedesign = {};
      for p=1:50
        framedesign{p} = stimrec(mod2(setnum(2),9),p)*ones(1,9);
      end
    end
  case {73 74 75 76 77  78 79  89 90 91 92 93 94  101 102 103 104 105 106}
    % N/A
  end
end

%%%%%%%%%%%%% more preparations

% load in some aux info
if isseq
  switch setnum(1)
  case 1
    trialpattern = eye(31);
    onpattern = [1];
  case {2 3 7 8}
    trialpattern = eye(21);
    onpattern = [1];
  case {4 5 10 11 12 13 16 17 18 19 30 31 32 33 35 36}
    trialpattern = eye(35);
    onpattern = [1];
  case {82 83 84 85 86 87  88}
    trialpattern = eye(45);
    onpattern = [1];
  case {37}
    trialpattern = 1;
    onpattern = [1];
  case {20 21 22 23 34}
    trialpattern = eye(36);
    onpattern = [1];  %%ones(1,30)
  case {23.5}
    trialpattern = eye(13);
    onpattern = [1];  %%ones(1,30)
  case {27 28}
    trialpattern = eye(35);
    onpattern = [1];
  case {29}
    trialpattern = eye(13);
    onpattern = [1];
  case {50}
    trialpattern = eye(72+14);
    onpattern = [1];
  case {26}
    trialpattern = eye(3);
    onpattern = [1];
  case {24 25}
    trialpattern = eye(19);
    onpattern = [1];
  case {14 15}
    trialpattern = eye(24);
    onpattern = [1];
  case 6
    trialpattern = eye(33);
    onpattern = [1];
  case 9
    trialpattern = eye(1);
    onpattern = [1];
  case {38 39 40 41}
    trialpattern = eye(39);
    onpattern = [1];
  case {51 52}
    trialpattern = eye(52+10);
    onpattern = [1];
  case {53 54}
    trialpattern = eye(49+10);
    onpattern = [1];
  case {56 57 58}
    trialpattern = eye(49+10);
    onpattern = [1];
  case {62 63 64 65}
    trialpattern = eye(49+10);
    onpattern = [1];
  case {66 109 110}
    trialpattern = eye(25);
    onpattern = [1];
  case {111}
    trialpattern = eye(15);
    onpattern = [1];
  case {95 96 97 98 99 100 107}
    trialpattern = eye(32);
    onpattern = [1];
  case {108}
    trialpattern = eye(24);
    onpattern = [1];
  case {67 68 69 70 71 72}
    trialpattern = eye(41+5);
    onpattern = [1];
  case {59 60 61}
    trialpattern = eye(55+11);
    onpattern = [1];
  case {48}
    trialpattern = eye(9);
    onpattern = [1];
  case {46 47}
    trialpattern = eye(32);
    onpattern = [1];
  case {49}
    trialpattern = eye(35);
    onpattern = [1];
  case {55}
    trialpattern = eye(20);
    onpattern = [1];
  case {42 43 44}
    trialpattern = eye(40);
    onpattern = [1];
  case {45}
    trialpattern = eye(30);
    onpattern = [1];
  case {73 74 75 76 77  78 79  89 90 91 92 93 94  101 102 103 104 105 106}
    % N/A
  end
else
  switch setnum(1)
  case 1
    load(infofile,'trialpattern','onpattern');
  case {38 39 40 41}
    load(infofile_monster,'trialpattern','onpattern');
  case {51 52}
    load(infofile_categoryC3,'trialpattern','onpattern');
  case {53 54}
    load(infofile_categoryC4,'trialpattern','onpattern');
  case {56 57 58}
    load(infofile_categoryC5,'trialpattern','onpattern');
  case {62 63 64 65}
    load(infofile_categoryC7,'trialpattern','onpattern');
  case {66}
    load(infofile_categoryC8,'trialpattern','onpattern');
    onpattern = upsamplematrix(onpattern,[1 2],[],[],'nearest');
  case {109 110}
    trialpattern = gentrialpattern;
    onpattern = [zeros(1,4*2) ones(1,7*2) zeros(1,2)];
  case {111}
    trialpattern = gentrialpattern;
    onpattern = [zeros(1,2*20) repmat([ones(1,16) zeros(1,4)],[1 4])];
  case {95}
    load(infofile_readingB,'trialpattern','onpattern');
  case {96}
    load(infofile_readingBshort,'trialpattern','onpattern');
  case {97}
    load(infofile_readingBshortT1,'trialpattern','onpattern');
  case {98}
    load(infofile_readingBshortT2,'trialpattern','onpattern');
  case {99}
    load(infofile_readingBshortT3,'trialpattern','onpattern');
  case {100 107}
    load(infofile_readingBalt,'trialpattern','onpattern');
  case {108}
    load(infofile_readingD,'trialpattern','onpattern');
  case {67 68 69 70 71 72}
    load(infofile_categoryC9,'trialpattern','onpattern');
  case {88}
    load(infofile_readingALT,'trialpattern','onpattern');
  case {59 60 61}
    load(infofile_categoryC6,'trialpattern','onpattern');
  case {48}
    load(infofile_monstersub,'trialpattern','onpattern');
  case {46 47}
    load(infofile_monstertest,'trialpattern','onpattern');
  case {49}
    load(infofile_monstertestB,'trialpattern','onpattern');
  case {55}
    load(infofile_monstertestBALT,'trialpattern','onpattern');
  case {50}
    load(infofile_category,'trialpattern','onpattern');
  case {42 43 44}
    load(infofile_monsterB,'trialpattern','onpattern');
  case {45}
    load(infofile_subadd,'trialpattern','onpattern');
  case {2 3 7 8}
    load(infofileB,'trialpattern','onpattern');
  case {4 5}
    load(infofile_wn,'trialpattern','onpattern');
  case {10 11}
    load(infofile_wnB,'trialpattern','onpattern');
  case {12 13 18 19}
    load(infofile_wnB2,'trialpattern','onpattern');
  case {30 31 32 33}
    load(infofile_wnB8,'trialpattern','onpattern');
  case {37}

    onpattern = [1 1 1];
    
    seq = gen_mseq(struct('base',7,'power',2,'shift',0));
    seq = seq(:,1)';
    
    % each trial is one second
    trialpattern = [];
    trialpattern = [trialpattern zeros(1,18)];
    for pp=1:length(seq)
      switch seq(pp)
      case 0
        trialpattern = [trialpattern zeros(1,8)];
      case 1
        trialpattern = [trialpattern 1+j 1 1 zeros(1,6)];
      case 2
        trialpattern = [trialpattern 1+2*j 1 1 0 0 0 1 1 1 zeros(1,6)];
      case 3
        trialpattern = [trialpattern 1+3*j 1 1 1 zeros(1,6)];
      case 4
        trialpattern = [trialpattern 1+4*j 1 1 1 0 0 1 1 1 1 zeros(1,6)];
      case 5
        trialpattern = [trialpattern 1+5*j 1 1 1 1 zeros(1,6)];
      case 6
        trialpattern = [trialpattern 1+6*j 1 1 1 1 0 1 1 1 1 1 zeros(1,6)];
      end
    end
    trialpattern = [trialpattern zeros(1,18)];
    trialpattern = trialpattern.';
    eventpattern = imag(trialpattern);
    trialpattern = real(trialpattern);

  case {82 83 84 85 86 87}
  
    % onpattern is easy: flashing 200ms ON, 200ms OFF
    onpattern = [1 0 1 0 1];

    % trialpattern is tricky.  first, we initialize the size.
    trialpattern = zeros(272,45+6);  % 45 actual stimulus conditions; 6 catch trials in total
    
    % now, we figure out the event ordering and the event durations.
    allevents = repmat([1:45 repmat(-1,[1 3])],[1 2]);  % -1 indicates the catch trials
    alldurations = repmat([2 3],[1 45+3]);  % how long each trial is in seconds
    while 1
      allevents = permutedim(allevents);
      alldurations = permutedim(alldurations);
      if any(diff(allevents)==0)  % if any repeat already, this is bad
      else
        wherecatch = find(allevents==-1);
        if wherecatch(1)==1  % if first trial is catch, this is bad
        else
          bad = 0;
          timebetweencatch = zeros(1,7);  % include beginning and end of the trials as boundaries
          for wii=1:7
            if wii==1
              temp = sum(alldurations(1:wherecatch(wii)-1));
            elseif wii==7
              temp = sum(alldurations(wherecatch(wii-1):end));
            else
              temp = sum(alldurations(wherecatch(wii-1):wherecatch(wii)-1));
            end
            bad = bad | (temp < 10 | temp > 70);  % time between catch trials must be in [10,70] s
            timebetweencatch(wii) = temp;
          end
          if bad
          else
            break;
          end
        end
      end
    end
    
    % cool. now fill in trialpattern.
    cnt = 17;   % TR counter (after the initial 16-s rest period)
    ecnt = 1;   % event counter (wrt allevents)
    ccnt = 1;   % catch trial counter (up to 6 catch trials)
    while 1
      if ecnt > length(allevents)
        break;
      end
      if allevents(ecnt)==-1  % if catch trial
        trialpattern(cnt,45+ccnt) = 1;
        ccnt = ccnt + 1;
      else
        trialpattern(cnt,allevents(ecnt)) = 1;
      end
      cnt = cnt + alldurations(ecnt);
      ecnt = ecnt + 1;
    end
    
  case {35 36}
    load(infofile_wnB8,'trialpattern','onpattern');
    onpattern = zeros(1,24);
    onpattern(1:12) = 1;
  case {20 21 22 23 34}
    load(infofile_wnB4,'trialpattern','onpattern');
  case {23.5}
    trialpattern = eye(13);  % no null trials, who cares
    onpattern = [ones(1,30) zeros(1,50)];
  case {27 28}
    load(infofile_wnB6,'trialpattern','onpattern');
  case {29}
    load(infofile_wnB7,'trialpattern','onpattern');
  case {26}
    trialpattern = zeros(25,3);  % 4 s trials (100 s total; 75 volumes, 3 vol per 4 s)
    trialpattern(5,1) = 1;
    trialpattern(5+7,2) = 1;
    trialpattern(5+7+7,3) = 1;
    onpattern = zeros(1,40);
    onpattern(1:30) = 1;
  case {24 25}
    load(infofile_wnB5,'trialpattern','onpattern');
  case {16 17}
    load(infofile_wnB3,'trialpattern','onpattern');
  case {14 15}
    load(infofile_wnD,'trialpattern','onpattern');
  case 6
    load(infofile_version2,'trialpattern','onpattern');
  case 9
    load(infofile_hrf,'trialpattern','onpattern');
  case {73 74 75 76 77  78 79  89 90 91 92 93 94  101 102 103 104 105 106}
    % N/A
  end
end

% skip trials?
if exist('trialpattern','var') && ~isempty(skiptrials)
  trialpattern = trialpattern(skiptrials+1:end,:);
end

% decide assignment of classes to the events in the experimental design
if ~isempty(existingfile)
  classorder = efile.classorder;
else
  switch setnum(1)
  case 1
    classorder = [1 2 3 4 5 6 7 9 11 8 10 12 13:31];  % e.g., event 1 will be classorder(1)
  case {82 83 84 85 86 87}
    classorder = [1:45];   % NOTE: this does not mention the catch trials. we will handle that manually.
  case {88}
    classorder = [1:45 repmat(NaN,[1 5])];
  case {42}
    classorder = [1:40 repmat(NaN,[1 9])];
  case {43 44}
    classorder = [1:30 41:50 repmat(NaN,[1 9])];
  case {45}
    classorder = [1:30];
  case {38 39 40 41}
% DO THIS ONCE:
%     temp = permutedim(1:156);
%     temp1 = temp(1:39); mat2str(temp1)
%     temp2 = temp(39+(1:39)); mat2str(temp2)
%     temp3 = temp(2*39+(1:39)); mat2str(temp3)
%     temp4 = temp(3*39+(1:39)); mat2str(temp4)
    switch setnum(1)
    case 38
      classorder = [122 96 114 57 47 17 10 110 126 120 95 148 44 6 128 28 105 32 31 13 145 27 39 53 156 3 113 131 75 36 115 139 123 119 146 23 117 89 61];
    case 39
      classorder = [54 50 78 121 93 82 38 2 143 134 118 55 112 25 46 111 136 109 64 48 9 37 62 11 127 15 142 85 151 56 152 74 42 133 81 58 63 80 7];
    case 40
      classorder = [153 155 141 34 132 8 99 52 41 88 144 33 116 104 147 98 71 79 68 103 138 51 125 150 21 92 26 140 94 45 129 77 20 102 124 69 60 101 18];
    case 41
      classorder = [84 49 86 35 67 65 66 24 22 43 30 1 108 5 137 59 16 97 154 130 100 87 29 90 70 106 72 107 76 73 19 91 40 14 4 149 12 83 135];
    end
  case {50}
    classorder = [1:72 repmat(NaN,[1 14])];
  case {51 52}
% % DO THIS ONCE:
%     temp = permutedim(1:52);
%     temp1 = [temp(1:26) temp(1:26)+52]; mat2str(temp1)
%     temp2 = [temp(27:52) temp(27:52)+52]; mat2str(temp2)
    switch setnum(1)
    case 51
      classorder = [37 23 35 40 13 21 17 31 28 20 18 26 1 9 3 25 10 50 7 38 39 45 36 16 5 42 89 75 87 92 65 73 69 83 80 72 70 78 53 61 55 77 62 102 59 90 91 97 88 68 57 94 repmat(NaN,[1 10])];
    case 52
      classorder = [6 12 29 19 4 34 22 48 32 2 43 51 44 30 27 52 41 33 47 49 46 15 24 14 11 8 58 64 81 71 56 86 74 100 84 54 95 103 96 82 79 104 93 85 99 101 98 67 76 66 63 60 repmat(NaN,[1 10])];
    end
  case {53 54}
% % DO THIS ONCE:
%     tempF = permutedim(1:49);
%     tempH = permutedim(1:49);
%     temp1 = [tempF(1:25) tempH(1:24)+49]; mat2str(temp1)
%     temp2 = [tempF(26:end) tempH(25:end)+49]; mat2str(temp2)
    switch setnum(1)
    case 53
      classorder = [6 26 43 18 8 38 3 42 25 24 35 36 34 13 1 45 28 29 12 10 23 11 14 47 5 82 79 97 81 80 64 51 94 69 89 88 54 93 56 59 83 60 62 61 84 86 65 75 70 repmat(NaN,[1 10])];
    case 54
      classorder = [22 16 31 17 19 15 37 27 2 49 21 9 46 39 30 40 32 48 33 44 7 4 20 41 98 91 50 78 57 63 85 72 95 55 90 68 74 53 87 73 77 92 71 58 66 67 96 76 52 repmat(NaN,[1 10])];
    end
  case {56 57 58}
% % DO THIS ONCE:  [idea: in run1 show a random third of each of the sizes, etc.]
%     tempF = permutedim(1:49);
%     tempH = permutedim(1:49);
%     tempI = permutedim(1:49);
%     temp1 = [tempF(1:17)         tempH(1:16)+49         tempI(1:16)+49+49];         mat2str(temp1)
%     temp2 = [tempF(17+(1:16))    tempH(16+(1:17))+49    tempI(16+(1:16))+49+49];    mat2str(temp2)
%     temp3 = [tempF(17+16+(1:16)) tempH(16+17+(1:16))+49 tempI(16+16+(1:17))+49+49]; mat2str(temp3)
    switch setnum(1)
    case 56
      classorder = [17 3 31 19 49 20 16 4 36 28 2 15 39 26 12 41 37 53 90 63 78 69 59 66 94 65 50 98 76 73 51 70 89 116 124 130 143 131 132 129 103 118 102 126 110 147 117 100 121 repmat(NaN,[1 10])];
    case 57
      classorder = [44 22 46 40 27 33 48 25 32 11 45 7 10 5 38 23 64 95 54 93 58 72 67 91 97 92 75 62 60 68 80 52 82 114 104 120 127 146 105 112 142 99 137 128 136 115 141 144 108 repmat(NaN,[1 10])];
    case 58
      classorder = [29 35 42 8 30 18 43 9 14 1 47 13 6 21 34 24 96 87 57 86 79 55 84 56 71 88 81 77 74 85 61 83 101 145 107 119 140 109 122 133 138 135 125 123 134 111 106 113 139 repmat(NaN,[1 10])];
    end
  case {62 63 64 65}
% % DO THIS ONCE:  [idea: in run1 show a random fourth of each of the types, etc.]
%     tempF = permutedim(1:49);
%     tempH = permutedim(1:49);
%     tempI = permutedim(1:49);
%     tempJ = permutedim(1:49);
%     temp1 = [tempF(1:12)         tempH(1:12)+49         tempI(1:12)+2*49          tempJ(1:13)+3*49]; mat2str(temp1)
%     temp2 = [tempF(12+(1:12))    tempH(12+(1:12))+49    tempI(12+(1:13))+2*49     tempJ(13+(1:12))+3*49];    mat2str(temp2)
%     temp3 = [tempF(12+12+(1:12)) tempH(12+12+(1:13))+49 tempI(12+13+(1:12))+2*49  tempJ(13+12+(1:12))+3*49]; mat2str(temp3)
%     temp4 = [tempF(12+12+12+(1:13)) tempH(12+12+13+(1:12))+49 tempI(12+13+12+(1:12))+2*49  tempJ(13+12+12+(1:12))+3*49]; mat2str(temp4)
    switch setnum(1)
    case 62
      classorder = [16 45 5 35 8 44 23 14 41 32 49 6 62 81 90 97 70 88 50 55 93 53 94 85 144 121 112 118 141 113 139 145 129 137 117 100 193 173 162 166 181 152 195 151 169 180 154 174 192 repmat(NaN,[1 10])];
    case 63
      classorder = [36 15 19 29 20 33 25 27 46 10 21 38 98 75 74 80 72 61 92 60 52 71 79 76 123 127 132 133 114 131 115 107 120 103 142 135 105 177 150 157 186 160 182 190 187 189 172 158 149 repmat(NaN,[1 10])];
    case 64
      classorder = [28 18 30 13 47 43 3 22 11 34 40 12 63 86 82 95 67 69 54 51 57 91 58 59 56 109 126 108 122 125 102 146 106 101 130 138 110 163 165 188 178 191 164 184 170 196 185 171 156 repmat(NaN,[1 10])];
    case 65
      classorder = [37 42 9 4 1 2 26 7 31 17 48 39 24 89 78 68 87 73 66 84 83 77 65 64 96 124 134 140 116 147 104 128 136 111 99 143 119 153 179 159 155 183 176 175 148 194 168 167 161 repmat(NaN,[1 10])];
    end
  case {66}
    classorder = [1:25];
  case {109 110}
% % DO THIS ONCE:  [idea: in run1 show a random half of the 25 (13), in run2 show the other half (12)]
%     tempF = permutedim(1:25);
%     temp1 = [tempF(1:13)]; mat2str(sort(temp1))
%     temp2 = [tempF(13+(1:12))]; mat2str(sort(temp2))
    switch setnum(1)
    case 109
      classorder = [1 2 6 7 11 12 14 15 19 20 21 22 25];  % note that randomization is implemented via setupmulticlassfun
    case 110
      classorder = [3 4 5 8 9 10 13 16 17 18 23 24];
    end
  case {111}
    classorder = [1:15];
  case {95 96 97 98 99 100 107}
    classorder = [1:32];
  case {108}
    classorder = [1:24];
  case {59 60 61}
% % DO THIS ONCE:  [idea: just split randomly into thirds.]
%     tempF = permutedim(1:165);
%     mat2str(tempF(1:55))
%     mat2str(tempF(55+(1:55)))
%     mat2str(tempF(2*55+(1:55)))
    switch setnum(1)
    case 59
      classorder = [163 21 100 159 52 82 63 121 111 54 146 19 62 39 44 96 61 22 18 124 36 48 38 140 51 23 149 101 117 14 130 92 156 91 25 153 162 109 161 94 155 8 3 131 129 34 85 160 4 20 64 87 49 113 125 repmat(NaN,[1 11])];
    case 60
      classorder = [127 58 103 133 115 128 108 46 33 83 72 154 43 104 56 16 40 135 97 45 13 66 84 106 65 26 75 93 144 119 37 148 137 9 11 107 145 98 6 132 95 139 110 141 150 31 114 74 12 164 120 151 77 81 10 repmat(NaN,[1 11])];
    case 61
      classorder = [30 69 5 17 71 99 29 73 55 157 118 32 152 90 142 78 126 136 41 80 53 60 28 24 158 7 147 27 116 122 42 112 1 76 67 70 59 2 35 123 86 134 50 143 165 88 57 138 89 79 15 68 102 105 47 repmat(NaN,[1 11])];
    end
  case {67 68 69 70 71 72}
% % DO THIS ONCE:  [idea: just split randomly into halves.]
%     tempF = permutedim(1:81);
%     mat2str(tempF(1:41))
%     mat2str(tempF(42:end))
    switch setnum(1)
    case {67 69 71}
      classorder = [18 66 58 21 9 70 81 67 75 53 1 48 28 30 36 79 72 44 38 80 7 11 17 73 24 68 59 71 32 57 46 42 10 16 19 29 63 49 37 77 76 repmat(NaN,[1 5])];
    case {68 70 72}
      classorder = [31 26 41 8 78 35 74 33 39 3 45 50 25 64 6 54 40 47 15 23 12 5 56 20 14 4 65 69 60 61 2 62 52 55 13 43 51 34 27 22 repmat(NaN,[1 6])];
    end
  case {48}
    classorder = [[43 59 47] 145:147 148:150];  % bottom top full (Z)   bottom top full (V)   bottom top full (H)
  case {46 47}
% DO THIS ONCE:
%     temp = permutedim(1:64);
%     temp1 = temp(1:32); mat2str(temp1)
%     temp2 = temp(32+(1:32)); mat2str(temp2)
    switch setnum(1)
    case 46
      classorder = [1 54 52 21 41 40 43 60 15 34 47 25 27 59 13 4 50 3 23 19 24 37 33 36 62 53 2 39 63 64 10 58];
    case 47
      classorder = [8 31 6 5 48 28 26 57 17 61 12 14 11 29 56 20 42 16 18 22 55 9 51 38 45 44 30 7 32 46 35 49];
    end
  case {49}
    classorder = [1:35];
  case {55}
    classorder = [1:20];
  case 2
    classorder = [32:50 21 18];  % small scale checkerboard and letters
  case 3
    classorder = [51:69 24 1];  % natural and white noise
  case {4 10 12 16}
    classorder = [1:31 63:66];
  case {5 11 13   }
    classorder = [31+(1:31) 67:69 NaN];
  case {30 32 35}
    classorder = [1:31 63:2:69];
  case {37}
    classorder = [16];
  case {31 33 36}
    classorder = [31+(1:31) 64:2:69 NaN];
  case 18
    classorder = [1:31 63 65 67 69];
  case 19
    classorder = [31+(1:31) 64 66 68 70];
  case {20 22 34}
    classorder = [1:19 39:2:72];
  case 23.5
    classorder = [38+8+13+(1:13)];
  case {26}
    classorder = [10 5 15];
  case {27}
    classorder = [1:35];
  case {28}
    classorder = [35+(1:35)];
  case {29}
    classorder = [1:13];
  case {21 23}
    classorder = [19+(1:19) 40:2:72];
  case 24
    classorder = [1:19];
  case 25
    classorder = [19+(1:19)];
  case 17
    classorder = [31+(1:31) 67:70];
  case {14 15}
    classorder = [1:24];
  case 6
    classorder = [1 2 3 4 5 6 7 9 11 8 10 12 74 13:20 75 21:31];
  case 7
    classorder = [32:50 70:71];
  case 8
    classorder = [51:69 72:73];
  case 9
    classorder = [1];
  case {73 74 75 76 77  78 79  89 90 91 92 93 94  101 102 103 104 105 106}
    % N/A but do this just so the below line won't fail
    classorder = [];
  end
  if ~isseq && ~ismember(setnum(1),[26 109 110 111])  % 109-111 is special. we pre-specify, so no randomization etc.
    classorder = permutedim(classorder);
    
    % make sure beginning and end are stimulus trials and make sure no two consecutive blank trials
    if ismember(setnum(1),[50 51 52 53 54 56 57 58 59 60 61 62 63 64 65 67 68 69 70 71 72 88])
      while 1
        wherenan = find(isnan(classorder));
        if any(ismember([1 length(classorder)],wherenan)) || any(diff(wherenan)==1)
          classorder = permutedim(classorder);
        else
          break;
        end
      end
    end
  end
end

% some abbrevations for the retinotopy cases
switch setnum(1)
case {73 74 75 76 77}
  firstslot = 16*15 + (1:4*32*15);
  secondslot = (16+4*32+16)*15 + (1:4*32*15);
  standardwedge = repmat(upsamplematrix(1:16,30,2,0,'nearest'),[1 4]);
  standardgap = repmat([upsamplematrix(1:14,30,2,0,'nearest') zeros(1,4*15)],[1 4]);
  flickernogap = repmat([1 2],[1 4*32*15/2]);
  flickergap = repmat([repmat([1 2],[1 28*15/2]) zeros(1,4*15)],[1 4]);
  totalframesstandard = (16+4*32+16+4*32+16)*15;
case {78 79}
  firstslot = 16*15 + (1:4*32*15);
  secondslot = (16+4*32+16)*15 + (1:4*32*15);
  standardwedge = repmat(upsamplematrix(1:32,15,2,0,'nearest'),[1 4]);
  standardgap = repmat([upsamplematrix(1:28,15,2,0,'nearest') zeros(1,4*15)],[1 4]);
  mashnogap = randintrange(1,100,[1 4*32*15],1);
  mashgap = repmat([randintrange(1,100,[1 28*15],1) zeros(1,4*15)],[1 4]);   
  totalframesstandard = (16+4*32+16+4*32+16)*15;
case {89 90 91 92 93 94  101 102 103 104 105 106}

  % deal with dfactor0
  if ismember(setnum(1),[103 104 105 106])
    dfactor0 = 3;
  else
    dfactor0 = 1;
  end

  totalframesstandard = 300*15;  % 300 s exactly
  cycleslots = 22*15 + (1:8*32*15);  % after 22-s rest, 8 cycles of 32 s
  mashcycles = randintrange(1,100,[1 8*32*15],1);  % long string of 8*32 stimuli
  mashgapcycles = [];  % interrupted 28/4 stimuli
  for nn=1:8
    mashgapcycles = [mashgapcycles randintrange(1,100,[1 28*15],1) zeros(1,4*15)];
  end
  mashgapcyclesSP = mashgapcycles;  % special for contracting-ring case
  mashgapcyclesSP(1:32*15:end) = 0;
  mashgaponecyclefun = @() upsamplematrix(randintrange(1,100,[1 28*15/dfactor0],1),dfactor0,2,[],'nearest');  % just do stimuli for one 28-s thing
  standardcycles = repmat(1:32*15,[1 8]);        % repeat a regular 32-s stim 8 times
  reversecycles = repmat([1 32*15:-1:2],[1 8]);  % repeat a reversed 32-s stim 8 times
  standardgapcycles = repmat([1:28*15 zeros(1,4*15)],[1 8]);       % regular order for 28/4 interruptions
  reversegapcycles = repmat([0 28*15:-1:2 zeros(1,4*15)],[1 8]);   % reversed order for contracting-ring-specific!
  standardgaponecycle = 1:28*15;
  reversegaponecycle = [1 28*15:-1:2];  % note the first 1 is really a blank anyway.
  
  % special handling of making the apertures slow too
  if ismember(setnum(1),[105 106])
    standardgaponecycle = upsamplematrix(1:dfactor0:28*15,dfactor0,2,[],'nearest');
    temp = fliplr(upsamplematrix(1:dfactor0:28*15,dfactor0,2,[],'nearest'));
    reversegaponecycle = [ones(1,dfactor0) temp(1:end-dfactor0)];  % note the first 1 is really a blank anyway.
  end

end

% init of framecolor (might be overridden later)
framecolor = [];

% figure out frameorder (handle the special retinotopy cases first)
switch setnum(1)

case 73

  % init
  frameorder = zeros(2,totalframesstandard);

  % handle the first stim (b&w dartboard)
  stimoffset = 0; maskoffset = 0;
  frameorder(1,firstslot) = stimoffset + flickernogap;
  frameorder(2,firstslot) = maskoffset + standardwedge;

  % handle the second stim (color dartboard)
  stimoffset = 2; maskoffset = 0;
  frameorder(1,secondslot) = stimoffset + flatten([randintrange(1,100,[1 length(secondslot)/2]); randintrange(101,200,[1 length(secondslot)/2])]);
  frameorder(2,secondslot) = maskoffset + standardwedge;

case 74

  % init
  frameorder = zeros(2,totalframesstandard);

  % handle the first stim (fast mashup)
  stimoffset = 2+200; maskoffset = 0;
  frameorder(1,firstslot) = stimoffset + randintrange(1,100,[1 length(firstslot)],1);
  frameorder(2,firstslot) = maskoffset + standardwedge;

  % handle the second stim (slow mashup)
  stimoffset = 2+200; maskoffset = 0;
  frameorder(1,secondslot) = stimoffset + upsamplematrix(randintrange(1,100,[1 length(secondslot)/10],1),10,2,0,'nearest');  % ten times slower
  frameorder(2,secondslot) = maskoffset + standardwedge;

case 75

  % init
  frameorder = zeros(2,totalframesstandard);
  
  % handle the first stim (CCW wedge)
  stimoffset = 0; maskoffset = 0;
  frameorder(1,firstslot) = stimoffset + flickernogap;
  frameorder(2,firstslot) = maskoffset + standardwedge;
  
  % handle the second stim (expanding ring)
  stimoffset = 0; maskoffset = 16;
  frameorder(1,secondslot) = stimoffset + flickergap;
  frameorder(2,secondslot) = maskoffset + standardgap;

case 76

  % init
  frameorder = zeros(2,totalframesstandard);
  
  % handle the first stim (standard bar right)
  stimoffset = 0; maskoffset = 16+14;
  frameorder(1,firstslot) = stimoffset + flickergap;
  frameorder(2,firstslot) = maskoffset + standardgap;
  
  % handle the second stim (standard bar up)
  stimoffset = 0; maskoffset = 16+14+14;
  frameorder(1,secondslot) = stimoffset + flickergap;
  frameorder(2,secondslot) = maskoffset + standardgap;

case 77

  % init
  frameorder = zeros(2,totalframesstandard);
  
  % handle the first stim (crazy bar right)
  stimoffset = 0; maskoffset = 16+14+14+14;
  frameorder(1,firstslot) = stimoffset + flickergap;
  frameorder(2,firstslot) = maskoffset + standardgap;
  
  % handle the second stim (crazy bar up)
  stimoffset = 0; maskoffset = 16+14+14+14+14;
  frameorder(1,secondslot) = stimoffset + flickergap;
  frameorder(2,secondslot) = maskoffset + standardgap;

case 78

  % init
  frameorder = zeros(2,totalframesstandard);
  
  % handle the first stim (CCW wedge)
  stimoffset = 0; maskoffset = 0;
  frameorder(1,firstslot) = copymatrix(stimoffset + mashnogap,mashnogap==0,0);  %%%%% HMM, THIS SHOULD BE RETROACTIVELY APPLIED TO 73-77?!
  frameorder(2,firstslot) = copymatrix(maskoffset + standardwedge,standardwedge==0,0);
  
  % handle the second stim (expanding ring)
  stimoffset = 0; maskoffset = 32;
  frameorder(1,secondslot) = copymatrix(stimoffset + mashgap,mashgap==0,0);
  frameorder(2,secondslot) = copymatrix(maskoffset + standardgap,standardgap==0,0);

case 79

  % init
  frameorder = zeros(2,totalframesstandard);
  
  % handle the first stim (standard bar right)
  stimoffset = 0; maskoffset = 32+28;
  frameorder(1,firstslot) = copymatrix(stimoffset + mashgap,mashgap==0,0);
  frameorder(2,firstslot) = copymatrix(maskoffset + standardgap,standardgap==0,0);
  
  % handle the second stim (standard bar up)
  stimoffset = 0; maskoffset = 32+28+28;
  frameorder(1,secondslot) = copymatrix(stimoffset + mashgap,mashgap==0,0);
  frameorder(2,secondslot) = copymatrix(maskoffset + standardgap,standardgap==0,0);

case 89

  % init
  frameorder = zeros(2,totalframesstandard);
  
  % handle all cycles of stimulus
  maskoffset = 0;
  frameorder(1,cycleslots) = mashcycles;
  frameorder(2,cycleslots) = maskoffset + standardcycles;

case 90

  % init
  frameorder = zeros(2,totalframesstandard);
  
  % handle all cycles of stimulus
  maskoffset = 0;
  frameorder(1,cycleslots) = mashcycles;
  frameorder(2,cycleslots) = maskoffset + reversecycles;

case 91

  % init
  frameorder = zeros(2,totalframesstandard);
  
  % handle all cycles of stimulus
  maskoffset = 32*15;
  frameorder(1,cycleslots) = mashgapcycles;
  frameorder(2,cycleslots) = copymatrix(maskoffset + standardgapcycles,standardgapcycles==0,0);

case 92

  % init
  frameorder = zeros(2,totalframesstandard);
  
  % handle all cycles of stimulus
  maskoffset = 32*15;
  frameorder(1,cycleslots) = mashgapcyclesSP;
  frameorder(2,cycleslots) = copymatrix(maskoffset + reversegapcycles,reversegapcycles==0,0);

case {93 101 103 105}

  % init
  frameorder = zeros(2,totalframesstandard);
  
  % L to R
  maskoffset = 32*15+28*15+0*(28*15);
  slot0 = 16*15 + 0*(32*15) + (1:28*15);
  frameorder(1,slot0) = mashgaponecyclefun();
  frameorder(2,slot0) = maskoffset + standardgaponecycle;

  % D to U
  maskoffset = 32*15+28*15+2*(28*15);
  slot0 = 16*15 + 1*(32*15) + (1:28*15);
  frameorder(1,slot0) = mashgaponecyclefun();
  frameorder(2,slot0) = maskoffset + standardgaponecycle;

  % R to L
  maskoffset = 32*15+28*15+0*(28*15);
  slot0 = 16*15 + 2*(32*15) + (1:28*15);
  frameorder(1,slot0) = mashgaponecyclefun();
  frameorder(2,slot0) = maskoffset + reversegaponecycle;

  % U to D
  maskoffset = 32*15+28*15+2*(28*15);
  slot0 = 16*15 + 3*(32*15) + (1:28*15);
  frameorder(1,slot0) = mashgaponecyclefun();
  frameorder(2,slot0) = maskoffset + reversegaponecycle;

  % LL to UR
  maskoffset = 32*15+28*15+1*(28*15);
  slot0 = 16*15 + 4*(32*15) + 12*15 + 0*(32*15) + (1:28*15);
  frameorder(1,slot0) = mashgaponecyclefun();
  frameorder(2,slot0) = maskoffset + standardgaponecycle;

  % LR to UL
  maskoffset = 32*15+28*15+3*(28*15);
  slot0 = 16*15 + 4*(32*15) + 12*15 + 1*(32*15) + (1:28*15);
  frameorder(1,slot0) = mashgaponecyclefun();
  frameorder(2,slot0) = maskoffset + standardgaponecycle;

  % UR to LL
  maskoffset = 32*15+28*15+1*(28*15);
  slot0 = 16*15 + 4*(32*15) + 12*15 + 2*(32*15) + (1:28*15);
  frameorder(1,slot0) = mashgaponecyclefun();
  frameorder(2,slot0) = maskoffset + reversegaponecycle;

  % UL to LR
  maskoffset = 32*15+28*15+3*(28*15);
  slot0 = 16*15 + 4*(32*15) + 12*15 + 3*(32*15) + (1:28*15);
  frameorder(1,slot0) = mashgaponecyclefun();
  frameorder(2,slot0) = maskoffset + reversegaponecycle;

case {94 102 104 106}

  % init
  frameorder = zeros(2,totalframesstandard);
  
  % handle wedgeCCW
  frameorder(1,22*15 + (1:2*32*15)) = upsamplematrix(randintrange(1,100,[1 2*32*15/dfactor0],1),dfactor0,2,[],'nearest');
  frameorder(2,22*15 + (1:2*32*15)) = 0 + repmat(upsamplematrix(1:dfactor0:32*15,dfactor0,2,[],'nearest'),[1 2]);
  
  % handle ringexpand
  mashgapcycles = [];  % interrupted 28/4 stimuli
  for nn=1:2
    mashgapcycles = [mashgapcycles upsamplematrix(randintrange(1,100,[1 28*15/dfactor0],1),dfactor0,2,[],'nearest') zeros(1,4*15)];
  end
  standardgapcycles = repmat([upsamplematrix(1:dfactor0:28*15,dfactor0,2,[],'nearest') zeros(1,4*15)],[1 2]);       % regular order for 28/4 interruptions
  frameorder(1,22*15 + 2*32*15 + (1:2*32*15)) = mashgapcycles;
  frameorder(2,22*15 + 2*32*15 + (1:2*32*15)) = copymatrix(32*15 + standardgapcycles,standardgapcycles==0,0);
  
  % handle wedgeCW
  frameorder(1,22*15 + 4*32*15 + (1:2*32*15)) = upsamplematrix(randintrange(1,100,[1 2*32*15/dfactor0],1),dfactor0,2,[],'nearest');
        %%OLD:   frameorder(2,22*15 + 4*32*15 + (1:2*32*15)) = 0 + repmat([1 32*15:-1:2],[1 2]);
  temp = fliplr(upsamplematrix(1:dfactor0:32*15,dfactor0,2,[],'nearest'));
  frameorder(2,22*15 + 4*32*15 + (1:2*32*15)) = 0 + repmat([ones(1,dfactor0) temp(1:end-dfactor0)],[1 2]);
  
  % handle ringcontract
  mashgapcycles = [];  % interrupted 28/4 stimuli
  for nn=1:2
    mashgapcycles = [mashgapcycles upsamplematrix(randintrange(1,100,[1 28*15/dfactor0],1),dfactor0,2,[],'nearest') zeros(1,4*15)];
  end
  for zzz=1:dfactor0
    mashgapcycles(zzz:32*15:end) = 0;
  end
  temp = fliplr(upsamplematrix(1:dfactor0:28*15,dfactor0,2,[],'nearest'));
  reversegapcycles = repmat([zeros(1,dfactor0) temp(1:end-dfactor0) zeros(1,4*15)],[1 2]);   % reversed order for contracting-ring-specific!
  frameorder(1,22*15 + 6*32*15 + (1:2*32*15)) = mashgapcycles;
  frameorder(2,22*15 + 6*32*15 + (1:2*32*15)) = copymatrix(32*15 + reversegapcycles,reversegapcycles==0,0);

otherwise  % this is the normal case
  framedesign0 = framedesign;  % just a temporary copy for convenience
  frameorder = zeros(3,0);
  if exist('framecolordesign','var')
    framecolordesign0 = framecolordesign;
    framecolor = zeros(0,1);  % we always prep for the alpha case!
  else
    framecolor = [];
  end
  stimclassrec = [];
  for p=1:size(trialpattern,1)

    % if a null trial
    if all(trialpattern(p,:)==0)
      frameorder = [frameorder zeros(3,length(onpattern))];
      if exist('framecolordesign','var')
        framecolor = [framecolor; zeros(length(onpattern),1)];
      end

    % if a stimulus trial
    else

      % 1-index of the event
      event = find(trialpattern(p,:)); assert(~isempty(event));
      
      % handle special catch trials
      if ismember(setnum,[82 83 84 85 86 87])
        if event > 45

          % this is a cool move that repeats the last physical stimulus trial frames.
          % note that it does assume that catch trials are never the first trial.
          % also, note that stimclassrec is not updated to reflect catch trials.
          frameorder = [frameorder laststimaddition];
          %%% NOTE: framecolor is not touched here because of the specialness of the setnum
          continue;

        end
      end
      
      % the stimulus class index we are doing now
      stimclass = classorder(event);
    
      % figure out frameorder
      if isnan(stimclass)
        frameorder = [frameorder zeros(3,length(onpattern))];
        if exist('framecolordesign','var')
          framecolor = [framecolor; zeros(length(onpattern),1)];
        end
      else
        stimclassrec = [stimclassrec stimclass];
        if exist('framespecial','var')  % in this special case, we have to look up which entry in images it actually is
          stimactual = framespecial(stimclass);
        else
          stimactual = stimclass;
        end
        temp = onpattern;
      
        % this is a special case to modulate position on the fly  [%% framecolor not handled because specialness of setnum]
        if ismember(setnum(1),[66 109 110 67 68 69 70 71 72 111])
        
          % they either (1) all come from the first and only class or (2) we handled the offset manually,
          % so there is no need for offset here
          temp(onpattern==1) = framedesign0{stimclass}(1,:);
        
          % taken from prepareimages_categoryC.m:
          switch setnum
          case {66 109 110}
            csfirst = [-189 -189 -189 -189 -189;-94 -94 -94 -94 -94;0 0 0 0 0;95 95 95 95 95;189 189 189 189 189];
            cssecond = [-189 -94 0 95 189;-189 -94 0 95 189;-189 -94 0 95 189;-189 -94 0 95 189;-189 -94 0 95 189];
            gridnn = 5;
            rowii = ceil(stimclass/gridnn);
            colii = mod2(stimclass,gridnn);
          case {67 68 69 70 71 72}
            csfirst = [-172 -172 -172 -172 -172 -172 -172 -172 -172;-129 -129 -129 -129 -129 -129 -129 -129 -129;-86 -86 -86 -86 -86 -86 -86 -86 -86;-43 -43 -43 -43 -43 -43 -43 -43 -43;0 0 0 0 0 0 0 0 0;43 43 43 43 43 43 43 43 43;86 86 86 86 86 86 86 86 86;129 129 129 129 129 129 129 129 129;172 172 172 172 172 172 172 172 172];
            cssecond = [-156 -117 -78 -39 0 39 78 117 156;-156 -117 -78 -39 0 39 78 117 156;-156 -117 -78 -39 0 39 78 117 156;-156 -117 -78 -39 0 39 78 117 156;-156 -117 -78 -39 0 39 78 117 156;-156 -117 -78 -39 0 39 78 117 156;-156 -117 -78 -39 0 39 78 117 156;-156 -117 -78 -39 0 39 78 117 156;-156 -117 -78 -39 0 39 78 117 156];
            gridnn = 9;
            rowii = ceil(stimclass/gridnn);
            colii = mod2(stimclass,gridnn);
          case {111}
            % assume that dres is specified as [A B], where A==B and this is what
            % is necessary to make it such that what used to be 26.5/66*378*2
            % pixels would then correspond to 4 deg.
            degtopx = (dres(1)/378 * (26.5/66*378*2)) / 4;  % convert deg to px
            xlocs0 = linspace(-18,18,15);
            ylocs0 = 0;
            cssecond = [];
            csfirst = [];
            for rowii=1:length(ylocs0)
              for colii=1:length(xlocs0)
                cssecond(rowii,colii) = round(xlocs0(colii) * degtopx);    % circshift second arg
                csfirst(rowii,colii) = round(-ylocs0(rowii) * degtopx);    % circshift first arg
              end
            end
            rowii = 1;
            colii = stimclass;
          end

          % construct frameorder
          temp2 = repmat([csfirst(rowii,colii); cssecond(rowii,colii)],[1 length(temp)]);
          laststimaddition = [temp; temp2];
          frameorder = [frameorder laststimaddition];

        % this is the normal case
        else
          temp(onpattern==1) = sum(numinclass(1:stimactual-1)) + framedesign0{stimclass}(1,:);
          laststimaddition = [temp; zeros(2,length(temp))];
          frameorder = [frameorder laststimaddition];
          if exist('framecolordesign','var')
            temp(onpattern==1) = framecolordesign0{stimclass}(1,:);
            framecolor = [framecolor; temp(:)];
          end
        end

        framedesign0{stimclass}(1,:) = [];  % we're done with this row
        if exist('framecolordesign','var')
          framecolordesign0{stimclass}(1,:) = [];
        end
      end

    end

  end
  clear framedesign0 framecolordesign0;
end
  % ah, repeat it
frameorder = repmat(frameorder,[1 numrep]);
if exist('framecolordesign','var')
  framecolor = repmat(framecolor,[1 numrep]);
end
if exist('stimclassrec','var')
  stimclassrec = repmat(stimclassrec,[1 numrep]);
end

% figure out fixationorder and fixationcolor
if isseq
  fixationorder = [];
  fixationcolor = [];
% HACK IN:
%   fixationorder = [-ones(1,1+length(frameorder)+1) 1];
%   fixationcolor = uint8([255 0 0]);
else
  if iscell(fixationinfo{1})
  
    % this is focase3 (the original way)
    if length(fixationinfo{1}{2})==2
  
      fixationorder = fixationinfo{1};
      fixationcolor = [];
    
      % if we have an existing file, hack it in
      if ~isempty(existingfile)
        fixationorder{7} = efile.digitrecord;
      end
    
    % this is focase4 (the new way)
    else
    
      % if we have an existing file, use it
      if ~isempty(existingfile)
        fixationorder = efile.fixationorder;  % seems easiest way. just shove it in.
      else
        fixationorder = fixationinfo{1};  % {A C X}
        fixationorder{4} = digitnamerecord;
        fixationorder{5} = digitcolorrecord;
      end
      fixationcolor = [];

    end

  else
    fixationorder = zeros(1,1+size(frameorder,2)+1);
    lastfix = 0;  % act as if a fixation flip is shown just before we start the movie, which we interpret as starting at 1
    while 1
      lastfix = lastfix + feval(soafun);  % figure out next fixation flip
      if lastfix <= size(frameorder,2)  % if we're still within the duration of the movie
        fixationorder(1+lastfix) = 1;  % be careful; the 1+ is necessary because there is an initial entry indicating what happens before the movie even starts
      else
        break;
      end
    end
    % deal with case of alpha flipping
    if length(fixationinfo)==3
      isreg = fixationorder==0;
      isflip = fixationorder==1;
      fixationorder(isreg) = fixationinfo{2};
      fixationorder(isflip) = fixationinfo{3};
      fixationcolor = fixationinfo{1};
    % deal with case of color changes
    else
      cur = 1;  % start with the first fixation dot color
      for q=1:length(fixationorder)
        if fixationorder(q)==1
          cur = firstel(permutedim(setdiff(1:size(fixationinfo{1},1),cur)));  % change to a new color
        end
        fixationorder(q) = -cur;
      end
      fixationorder = [fixationorder fixationinfo{2}];  % tack on alpha value
      fixationcolor = fixationinfo{1};
    end

    % if we have an existing file, hack it in [IS THIS CORRECTLY PLACED HERE??]
    if ~isempty(existingfile)
      fixationorder = efile.fixationorder;
    end

  end
end

% figure out specialcon
switch setnum(1)
case {42 43 44}
  load(stimfile,'imagecontrasts');
  specialcon = {ptonparams{3} imagecontrasts double(fixationcolor) round(2*60/frameduration)};  % gamma change about 2 s before
  nnn = size(fixationcolor,1);
  fixationcolor = uint8(repmat((255-nnn+1:255)',[1 3]));  % hack it so that we use the end of the CLUT
otherwise
  specialcon = [];
end

% figure out trialtask [i.e. for the red dot task]
switch setnum(1)
case {51 52 53 54 56 57 58 59 60 61 62 63 64 65 66 109 110}

  % figure out A [THIS IS VOODOO, WATCH OUT]
  temp = strsplit(char(frameorder(1,:)),char(0));
  A = zeros(size(frameorder,2),size(frameorder,2));  % liberal on rows, restrict later
  cnt = 0;
  trialcnt = 1;
  for pp=1:length(temp)
    if isempty(temp{pp})
      cnt = cnt + 1;
    else
      A(trialcnt,cnt+(1:length(temp{pp}))) = 1;
      trialcnt = trialcnt + 1;
      cnt = cnt + length(temp{pp}) + 1;
    end
  end
  A = A(1:trialcnt-1,:);
  
  % load necessary stuff
  load(stimfile,'validlocations');

  % do a check on number of stimulus trials
  assert(length(stimclassrec)==size(A,1));
  
  % deal with movieflip now (not in ptviewmovie.m)
  for p=1:length(validlocations)
    if ~isempty(movieflip) && movieflip(1)==1
      validlocations{p}(2,:) = -validlocations{p}(2,:);
    end
    if ~isempty(movieflip) && movieflip(2)==1
      validlocations{p}(1,:) = -validlocations{p}(1,:);
    end
  end
  
  % construct trialtask
  trialtask = {A trialparams{1} validlocations stimclassrec trialparams{2} trialparams{3} trialparams{4} trialparams{5}};
  
  % add on if we have an existingfile
  if ~isempty(existingfile)
    trialtask{9} = efile.trialoffsets;
  end
  
  % now, we handle the special B case (note: this is a little dumb if existingfile is specified, since the existingfile should override)
  if ismember(setnum(1),[109 110])
    switch setnum(1)
    case 109
      nin = 13;
    case 110
      nin = 12;
    end
    mastercuestim0 = mastercuestim(floor(mastercuestim/10) <= nin);  % extract only stimulus trials
    assert(size(A,1)==length(mastercuestim0));  % sanity check
    trialgrouping = {};
    for p=1:nin
      trialgrouping{p} = find(floor(mastercuestim0/10)==p);  % 1 x 4 vector of indices relative to the valid stimulus trials
    end
    trialtask{2} = {trialtask{2} trialgrouping};
  end

case {78 79}

  % 78 and 79 will not have trialparams
  if isempty(trialparams)
    trialtask = [];

  % 80 and 81 will have trialparams, so we need to compute trialtask
  else

    % figure out A
    switch setnum(1)
    case 78
      A = zeros(32*4+28*4,15*304);
      for vv=1:32*4
        A(vv,15*16+(vv-1)*15+(1:15)) = 1;
      end
      for vv=1:4
        for uu=1:28
          A(32*4+(vv-1)*28+uu,15*(16+4*32+16)+(vv-1)*32*15+(uu-1)*15+(1:15)) = 1;
        end
      end
      trialmapping = [repmat(1:32,[1 4]) repmat(32+(1:28),[1 4])];
    case 79
      A = zeros(28*4+28*4,15*304);
      for vv=1:4
        for uu=1:28
          A((vv-1)*28+uu,15*16+(vv-1)*32*15+(uu-1)*15+(1:15)) = 1;
        end
      end
      for vv=1:4
        for uu=1:28
          A(28*4+(vv-1)*28+uu,15*(16+4*32+16)+(vv-1)*32*15+(uu-1)*15+(1:15)) = 1;
        end
      end
      trialmapping = [repmat(32+28+(1:28),[1 4]) repmat(32+28+28+(1:28),[1 4])];
    end

    % load necessary stuff
    load(stimfile,'validlocations');

    % deal with movieflip now (not in ptviewmovie.m)
    for p=1:length(validlocations)
      if ~isempty(movieflip) && movieflip(1)==1
        validlocations{p}(2,:) = -validlocations{p}(2,:);
      end
      if ~isempty(movieflip) && movieflip(2)==1
        validlocations{p}(1,:) = -validlocations{p}(1,:);
      end
    end
  
    % construct trialtask
    trialtask = {A trialparams{1} validlocations trialmapping trialparams{2} trialparams{3} trialparams{4} trialparams{5}};
  
    % add on if we have an existingfile
    if ~isempty(existingfile)
      trialtask{9} = efile.trialoffsets;
    end

  end

otherwise
  trialtask = [];
end

%%%%%%%%%%%%% figure out some last minute things

if ~isempty(dres) && length(dres)==1
  scfactor = -dres;
else
  scfactor = [];
end

%%%%%%%%%%%%% show the movie

% setup PT
oldclut = pton(ptonparams{:});

% initialize, setup, calibrate, and start eyelink
if ~isempty(eyelinkfile)

  assert(EyelinkInit()==1);
  win = firstel(Screen('Windows'));
  el = EyelinkInitDefaults(win);
  [wwidth,wheight] = Screen('WindowSize',win);  % returns in pixels
  fprintf('Pixel size of window is width: %d, height: %d.\n',wwidth,wheight);
  Eyelink('command','screen_pixel_coords = %ld %ld %ld %ld',0,0,wwidth-1,wheight-1);
  Eyelink('message','DISPLAY_COORDS %ld %ld %ld %ld',0,0,wwidth-1,wheight-1);
  Eyelink('command','calibration_type = HV5');
  Eyelink('command','active_eye = LEFT');
  Eyelink('command','automatic_calibration_pacing=1500');
    % what events (columns) are recorded in EDF:
  Eyelink('command','file_event_filter = LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON');
    % what samples (columns) are recorded in EDF:
  Eyelink('command','file_sample_data = LEFT,RIGHT,GAZE,HREF,AREA,GAZERES,STATUS');
    % events available for real time:
  Eyelink('command','link_event_filter = LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON');
    % samples available for real time:
  Eyelink('command','link_sample_data = LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS');
  Eyelink('Openfile',eyelinkfile);
  fprintf('Please perform calibration. When done, the subject should press a button in order to proceed.\n');
  EyelinkDoTrackerSetup(el);
%  EyelinkDoDriftCorrection(el);
  fprintf('Button detected from subject. Starting recording of eyetracking data. Proceeding to stimulus setup.\n');
  Eyelink('StartRecording');
  % note that we expect that something should probably issue the command:
  %   Eyelink('Message','SYNCTIME');
  % before we close out the eyelink.

end

% call ptviewmovie
timeofptviewmoviecall = datestr(now);
if iscolor
    % OLD AND WASTEFUL: cat(4,images{:})
  [timeframes,timekeys,digitrecord,trialoffsets] = ptviewmovie(images, ...
    frameorder,framecolor,frameduration,fixationorder,fixationcolor,fixationsize,grayval,[],[], ...
      offset,choose(con==100,[],1-con/100),movieflip,scfactor,[],triggerfun,framefiles,[], ...
      triggerkey,specialcon,trialtask,maskimages,specialoverlay);
else
    % OLD AND WASTEFUL: reshape(cat(3,images{:}),size(images{1},1),size(images{1},2),1,[])
  [timeframes,timekeys,digitrecord,trialoffsets] = ptviewmovie(images, ...
    frameorder,framecolor,frameduration,fixationorder,fixationcolor,fixationsize,grayval,[],[], ...
      offset,choose(con==100,[],1-con/100),movieflip,scfactor,[],triggerfun,framefiles,[], ...
      triggerkey,specialcon,trialtask,maskimages,specialoverlay);
end

% close out eyelink
if ~isempty(eyelinkfile)
  Eyelink('StopRecording');
  Eyelink('CloseFile');
  Eyelink('ReceiveFile');
  Eyelink('ShutDown');
end

% unsetup PT
ptoff(oldclut);

%%%%%%%%%%%%% clean up and save

% figure out names of all variables except 'images' and 'maskimages' and others   [MAKE THIS INTO A FUNCTION?]
vars = whos;
vars = {vars.name};
vars = vars(cellfun(@(x) ~isequal(x,'images') & ~isequal(x,'maskimages') & ~isequal(x,'validlocations') & ~isequal(x,'A') & ~isequal(x,'trialtask') & ~isequal(x,'mashgaponecyclefun'),vars));
  % avoid mashgaponecyclefun because it somehow saves workspace stuff

% save
save(outfile,vars{:});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [mastercuestim,digitnamerecord,digitcolorrecord,gentrialpattern,designmatrix] = setupmulticlassfun(setnum)

% this function generates a fresh random specification for the 109-110 and 111 experiments.

% notes (109-110):
% - 18-s rest + 51 trials x 6-s trials + 18-s rest = 342 seconds (171 TRs)
% - 18-s rest + 48 trials x 6-s trials + 18-s rest = 324 seconds (162 TRs)
% - (13 stimuli + 3 blank trials) x 3 tasks + 3 blank-blank trials = 51 trials
% - (12 stimuli + 3 blank trials) x 3 tasks + 3 blank-blank trials = 48 trials
% - blank trials still involve cue and digits; blank-blank trials just involve digits
% - GDF: Digit, Dot, Face
% - enforce physical identicality across tasks (including the blank trials)
%   - this includes digit sequence and face frames (if applicable) and dots

% notes (111):
% - 18-s rest + 57 trials x 6-s trials + 18-s rest = 378 seconds (189 TRs)
% - (15 stimuli + 3 blank trials) x 3 tasks + 3 blank-blank trials = 57 trials
% - blank trials still involve cue and digits; blank-blank trials just involve digits
% - GOF: Digit, Oddball, Face
% - enforce physical identicality across tasks (including the blank trials)
%   - this includes digit sequence and face frames (if applicable)

% define
switch setnum
case 109
  taskletters = 'GDF';
  nin = 13;
  tottrials = 51;
  classorder = [1 2 6 7 11 12 14 15 19 20 21 22 25];  % ugly, steal this from above
  fps = 4;
  digitups = 1;
  numdigitstr = 2+7;
  totstimunique = 25;
case 110
  taskletters = 'GDF';
  nin = 12;
  tottrials = 48;
  classorder = [3 4 5 8 9 10 13 16 17 18 23 24];  % ugly, steal this from above
  fps = 4;
  digitups = 1;
  numdigitstr = 2+7;
  totstimunique = 25;
case 111
  taskletters = 'GOF';
  nin = 15;
  tottrials = 57;
  classorder = 1:15;
  fps = 20;
  digitups = 5;          % number of frames for a single digit
  numdigitstr = 2+8;     % number of digits occurring in a trial
  totstimunique = 15;    % total number of unique stimuli (not tasks) in the experiment
end

% this tells us the master sequence
mastercuestim = flatten(bsxfun(@plus,[1 2 3]',10*(1:(nin+3))));  % 1 x trials
  % mod(a,10) is 1-3 indicating which task (the cue to use)
  % floor(a/10) is 1-(nin+3) indicating which stimulus (the true stimulus class is controlled by classorder) (the extra 3 are the blank cases)
  % however, NaNs can exist and indicate blank-blank trials

% add the blank-blanks
mastercuestim = [mastercuestim repmat(NaN,[1 3])];
assert(length(mastercuestim)==tottrials);

% randomize, make sure beginning and end are stimulus trials and make sure no two consecutive blank-blank trials
while 1
  mastercuestim = permutedim(mastercuestim);
  wherenan = find(isnan(mastercuestim));
  if ~(any(ismember([1 length(mastercuestim)],wherenan)) || any(diff(wherenan)==1))
    break;
  end
end

% initialize full record of digits
digitnamerecord = NaN*zeros(1,fps*(18 + tottrials*6 + 18));
digitcolorrecord = zeros(size(digitnamerecord));  % 0 means black, 1 means white

% handle initial blank period
  %OLD:ix = linspacefixeddiff(1,2,2*18);  % 2 digits per second
offset = 0;
ix = [];
for p=1:18
  ix = [ix (p-1)*fps+[1:digitups digitups*2+(1:digitups)]];  % 2 digits per second
end
  % fully random [1-10]
digitnamerecord(:,offset + ix) = upsamplematrix(randintrange(0,9,[1 length(ix)/digitups])+1,digitups,2,[],'nearest');
digitcolorrecord(:,offset + ix) = repmat([1*ones(1,digitups) 0*ones(1,digitups)],[1 18]);  % white/black alternation

% record
clock0 = sum(100*clock);

% handle each trial
for p=1:length(mastercuestim)

  % calc
  offset = fps*18 + (p-1)*(fps*6);

  % handle blank-blank trials up front
  if isnan(mastercuestim(p))
    ix = [];
    for q=1:6
      ix = [ix (q-1)*fps+[1:digitups digitups*2+(1:digitups)]];  % 2 digits per second
    end
    digitnamerecord(:, offset + ix) = upsamplematrix(randintrange(0,9,[1 length(ix)/digitups])+1,digitups,2,[],'nearest');  % fully random [1-10]
    digitcolorrecord(:,offset + ix) = repmat([1*ones(1,digitups) 0*ones(1,digitups)],[1 6]);  % white/black alternation
    continue;
  end

  % make deterministic depending on 1-(nin+3) (thus, different tasks should have identical digits).
  % thus, it is here that physicality of digit sequences is enforced (all three tasks see the same digits).
  setrandstate({clock0+999*floor(mastercuestim(p)/10)});
  
  % cue (red)
  digitnamerecord(:, offset + (1:fps/2)) = 10+(double(taskletters(mod(mastercuestim(p),10)))-64);
  digitcolorrecord(:,offset + (1:fps/2)) = 2;
  
  % generate digits and then repeat the digit maybe
  temp = randintrange(0,9,[1 numdigitstr],1)+1;
  if rand < .5
    fr = randintrange(2,numdigitstr);
    temp(fr) = temp(fr-1);
  end

  % stream of digits
     % OLD: ix = linspacefixeddiff(1,2,numdigitstr);
  ix = [];
  for q=1:numdigitstr
    ix = [ix (q-1)*(fps/2)+[1:digitups]];
  end
  
  % record
  digitnamerecord(:, offset + fps + ix) = upsamplematrix(temp,digitups,2,[],'nearest');
  digitcolorrecord(:,offset + fps + ix) = subscript(repmat([1*ones(1,digitups) 0*ones(1,digitups)],[1 ceil(numdigitstr/2)]),{':' 1:digitups*numdigitstr});

end

% reset rand seed
setrandstate;

% handle ending blank period
offset = fps*18 + fps*(tottrials*6);
ix = [];
for p=1:18
  ix = [ix (p-1)*fps+[1:digitups digitups*2+(1:digitups)]];  % 2 digits per second
end
  % fully random [1-10]
digitnamerecord(:,offset + ix) = upsamplematrix(randintrange(0,9,[1 length(ix)/digitups])+1,digitups,2,[],'nearest');
digitcolorrecord(:,offset + ix) = repmat([1*ones(1,digitups) 0*ones(1,digitups)],[1 18]);  % white/black alternation

% now deal with gentrialpattern [note that we intend to circumvent classorder (i.e. we don't randomize it)].
% this may have a limited number of columns (e.g. 13 or 12), and the columns have random onsets.
gentrialpattern = zeros(3+tottrials+3,nin);
for p=1:length(mastercuestim)
  stimnum = floor(mastercuestim(p)/10);
  if ~isnan(stimnum) && stimnum <= nin
    gentrialpattern(3+p,stimnum) = 1;
  end
end

% now deal with designmatrix
designmatrix = zeros(3+tottrials+3,totstimunique*3+3);  % stim-evoked Task1, Task2, Task3; cue-related Task1, Task2, Task3
for p=1:length(mastercuestim)
  if isnan(mastercuestim(p))
    continue;
  end
  stimnum = floor(mastercuestim(p)/10);
  tasknum = mod(mastercuestim(p),10);
  if stimnum <= nin
    designmatrix(3+p,(tasknum-1)*totstimunique+classorder(stimnum)) = 1;
  end
  designmatrix(3+p,totstimunique*3+tasknum) = 1;
end
