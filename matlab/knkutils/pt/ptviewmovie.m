function [timeframes,timekeys,digitrecord,trialoffsets] = ...
  ptviewmovie(images,frameorder,framecolor,frameduration,fixationorder,fixationcolor,fixationsize, ...
              grayval,detectinput,wantcheck,offset,moviemask,movieflip,scfactor,allowforceglitch, ...
              triggerfun,framefiles,frameskip,triggerkey,specialcon,trialtask,maskimages,specialoverlay)

% function [timeframes,timekeys,digitrecord,trialoffsets] = ...
%   ptviewmovie(images,frameorder,framecolor,frameduration,fixationorder,fixationcolor,fixationsize, ...
%               grayval,detectinput,wantcheck,offset,moviemask,movieflip,scfactor,allowforceglitch, ...
%               triggerfun,framefiles,frameskip,triggerkey,specialcon,trialtask,maskimages,specialoverlay)
% 
% <images> is a .mat file with 'images' as a uint8
%   A x B x 1/3 x N matrix with different images along the fourth dimension.  
%   the third dimension can be of size 1 or 3.  can be a cell vector, in 
%   which case we just concatenate all images together (the A and B 
%   dimensions should be consistent across cases).  images are referred 
%   to by their index (1-indexed).  if combining grayscale and color
%   images, we just repmat the grayscale images along the third dimension.
%   <images> can also be the uint8 matrix directly.  <images> can
%   also be a cell vector of uint8 elements, each of which is A x B x N 
%   or A x B x 3 x N (we detect this case if the size along the fourth
%   dimension is more than 1).  in these cases, the A and B must be 
%   consistent across the elements and <moviemask> must be [].
% <frameorder> (optional) is a vector of non-negative integers.  positive
%   integers refer to specific images in <images>.  zero means to show
%   no image (just the background and fixation).  this vector determines 
%   the order of presentation of images.  default is 1:N where N is the 
%   total number of images.  a special case is a matrix of dimensions
%   3 x N.  in this case, the first row is the usual format and the second
%   and third rows provide arguments to circshift (thus, they should be
%   integer values).  this input format allows circular shifting
%   of the images just prior to presentation.  another special case is a matrix
%   of dimensions 2 x N.  in this case, the first row is the usual format and
%   the second row is a vector of positive integers referring to specific
%   masks in <maskimages>.  (for entries where the first row is 0, just pass in
%   0 in the second row.)  this input format allows for a mask to be applied
%   to images in <images>.
% <framecolor> (optional) is size(<frameorder>,2) x 3 with values in [0,255].
%   each row indicates a multiplication on color channels to apply for a 
%   given frame.  default is 255*ones(size(<frameorder>,2),3) which means 
%   to multiply each channel by 1 (i.e. do nothing special).  note that
%   when an entry in <frameorder> is 0, <framecolor> has no effect for that entry.
%   <framecolor> can also be size(<frameorder>,2) x 1 with values in [0,1]
%   indicating an alpha change.
% <frameduration> (optional) is how many monitor refreshes you want a
%   single movie frame to last.  default: 15.
% <fixationorder> (optional) is:
%   (1) a vector of length 1+size(<frameorder>,2)+1 with elements that are non-
%       negative numbers in [0,1].  elements indicate alpha values.  the first 
%       element indicates what to use before the movie starts; the last element 
%       indicates what to use after the movie ends.  this is the regular
%       case, in which there is a single fixation dot color whose alpha value gets
%       modulated on different frames.
%   (2) elements can be negative integers plus an additional element that is an 
%       alpha value (thus, the length is 1+size(<frameorder>,2)+1+1).  the 
%       negative integers indicate row indices into <fixationcolor>, and the 
%       alpha value is used for all of the fixation colors.  in this alternative 
%       case, there are multiple possible fixation dot colors, all
%       of which get blended with a fixed alpha value.
%   (3) {A B C D E F G H} where A is the font size in (0,1), which is relative to
%       the size indicated by the first element of <fixationsize>; B is [ON OFF] 
%       where ON is a non-negative integer indicating the number of
%       frames for a digit to last and OFF is a non-negative integer
%       indicating the number of frames in between successive digits; C is whether
%       to omit the gray disc background; D is 0 (meaning just randomly pick), 1
%       (meaning randomly pick but ensure that successive digits are unique), or
%       a fraction between 0 and 1 indicating what proportion (on average) of cases
%       should have repeated digits (if you supply a negative fraction, we will
%       ensure that at most 2 successive digits will be the same); E 
%       is a positive integer indicating the number of CLUT 
%       entries to allocate at the end of the gamma table for pure
%       white and/or black (you need only one when F is 0, but you need two when
%       F is 1, and you can allocate more if you want to take up
%       entries); F (optional) is 0 means white digits and 1 means alternate
%       white and black digits (default: 0); G (optional) is a computed
%       digitrecord from a previous run (if this is supplied, there is no
%       stochasticity, as we just use what happened before); H (optional) is
%       [H V] indicating horizontal and vertical offsets.  we show a stream of 
%       random digits.  if C is 0, then the digits are shown on a gray disc that is
%       smoothly alpha-blended into the rest of the stimulus, with the
%       characteristics of this disc being determined by <fixationsize>. if C is
%       1, then the digits are directly superimposed on the rest of the stimulus.
%       before and after the movie, we show the digit '0'. note that
%       <fixationcolor> is ignored when <fixationorder> is of the {A B C D E F G H}
%       case.  also, note that if C is 1, then only the first element of
%       <fixationsize> is used (to determine the transparent box within
%       which the digits are presented).
%   (4) {A C X Y Z} where A and C are just like from case (3) and where X
%       is a cell vector of [H V] indicating horizontal and vertical offsets for each
%       digit, Y is numdigits x frames where each row specifies the digit to
%       show on each frame (1-10 mean '0':'9', 11-36 mean 'A':'Z', NaN means
%       none), and Z is numdigits x frames where each row specifies the color 
%       of the digit on each frame (0: black, 1: white, 2: red, 3: green,
%       4: blue, 5: yellow, 6: magenta, 7: cyan).  this case is essentially a 
%       modified version of case (3).  if this case is used, <specialcon> must be [].
%   default: ones(1,1+size(<frameorder>,2)+1).
% <fixationcolor> (optional) is a uint8 vector of dimensions 1 x 3, 
%   indicating the color to use for the fixation dot.  when <fixationorder>
%   is the special negative-integers case, <fixationcolor> should be a uint8
%   matrix of dimensions max(-fixationorder) x 3, indicating the possible colors
%   to use for the fixation dot.
%   default: uint8([255 255 255]).
% <fixationsize> (optional) is
%   A where A is the size in pixels for the fixation dot.  default: 5.  
%     special case is [A B] where A is the size for the fixation dot
%     and B is the number of pixels for the width of the border of the dot 
%     (the border is black).  for example, [6 1] means to use a dot that has
%     radius 3 and that has a border from 2 pixels to 3 pixels away from the center.
%     note that the default of 5 is equivalent to [5 0].
%   B where B is an N x N image of values in [0,1].  This image defines the
%     "fixation dot".  A value of 1 means to be fully opaque, and a value of 0
%     means to be fully transparent.  We detect this case by checking if
%     the size of the first dimension is greater than 1.  Note that any changes
%     to alpha (e.g. through <fixationorder>) are a modulation of the base 
%     alpha specified by <fixationsize>.
% <grayval> (optional) is the background color as uint8 1x1 or 1x3.
%   default: uint8(127).
% <detectinput> (optional) is whether to attempt to detect input during the 
%   showing of the movie.  if set to 1, you risk inaccuracies in 
%   the recorded times (for sure) and reduction (maybe) of your 
%   ability to achieve the desired framerate.
%   default: 1.
% <wantcheck> (optional) is whether to show some posthoc diagnostic figures
%   via ptviewmoviecheck.m.  default: 1.
% <offset> (optional) is [X Y] where X and Y are the
%   horizontal and vertical offsets to apply.  for example,
%   [5 -10] means shift 5 pixels to right, shift 10 pixels up.
%   default: [0 0].
% <moviemask> (optional) is an A x B matrix with values in [0,1].
%   0 means to pass through; 1 means to block.  we 
%   apply this mask to the images specified by <images>.  we blend 
%   with <grayval>.  default is [] which means do not apply a mask.
%   special case is when A or B (or both) are equal to 1; in this case,
%   we automatically expand (via bsxfun.m) to match the size of the images.
%   be careful: applying the mask is potentially a slow operation!
% <movieflip> (optional) is [J K] where J is whether to flip first
%   dimension and K is whether to flip second dimension.  this flipping
%   gets applied also to the fixation-related items.
%   default to [0 0].
% <scfactor> (optional) is a positive number with the scaling to apply
%   to the images in <images>.  if supplied, we multiply the number
%   of pixels in each dimension by <scfactor> and then round.  we use
%   bilinear filtering when rendering the images.  default is 1, and in
%   this case, we use nearest neighbor filtering (which is presumably
%   faster than bilinear filtering).
% <allowforceglitch> (optional) is
%   0 means do nothing special
%   [1 D] means allow keyboard input 'p' to force a glitch of duration D secs.
%     note that this has an effect only if <detectinput> is on.
%     forcing glitches is useful for testing purposes.
%   default: 0.
% <triggerfun> (optional) is the function to call right before we start the movie.
%   default is [] which means do not call any function.
%   if supplied, we create a 'trigger' event in <timekeys>, recording
%   the time of completion.
% <framefiles> (optional) is an sprintf string like '~/frame%05d.png'.  if supplied,
%   we write images containing the actual final frames shown on the display to 
%   the filenames specified by <framefiles>.  the files are 1-indexed, from 1 
%   through size(<frameorder>,2).  since writing to disk takes time, you may need 
%   to artificially increase <frameduration> to avoid glitches.  special case is
%   {A B} where A is like '~/frame%05d.png' and B is [R C] with the image dimensions
%   to crop to (relative to the center).  we make the parent directory for you
%   automatically (if necessary).  default: [].
% <frameskip> (optional) is a positive integer indicating how many frames to skip
%   when showing the movie.  for example, <frameskip>==2 means to show the 1st, 3rd,
%   5th, ... frames.  default: 1.  can also be 1/N for some positive integer N.
% <triggerkey> (optional) is
%   [] means any key can start the movie
%   X means if the first character of KbName(keyCode) where keyCode is obtained from
%     KbWait is X, then start the movie.  for example, you could pass in X as '5'.
%   default: [].
% <specialcon> (optional) is {A B C D}
%   where A is a Psychtoolbox calibration string, e.g. 'cni_lcd'
%         B is a vector of contrast values in (0,100], where the entries of this
%           vector matches the entries in <images>.  we determine the unique entries 
%           in this vector and do some precomputations based on that.
%         C is a N x 3 set of CLUT entries to use at the end of the gamma table,
%           e.g. for the fixation dot.  these are the linear values (before gamma correction).
%           we allocate 256-N entries to deal with the normal stimulus display.
%           thus, values in <images> should range from 0 through 255-N.  note that
%           when <fixationorder> is the {A B C D E F G H} case, we ignore the C input 
%           and always use E CLUT entries at the end of the gamma table, and when
%           <fixationorder> is the {A C X Y Z} case, we use 2 CLUT entries at the end.
%         D is how many movie frames before a gamma change to attempt to do the 
%           gamma change.  the reason for this is that the gamma changes seem to take
%           a relatively long time and trying to do it at the last minute produces
%           weird glitching behavior.  note that because we have to do gamma changes
%           ahead of time, no gamma changes will be performed for the first D movie frames,
%           so don't expect any!
%   if supplied, we do special handling of the gamma table.  for example, 50 contrast
%   value means to restrict the range of the values in the gamma table to 0.25 through 
%   0.75.  we use a contrast value of 100 before and after the movie.  note that we load
%   the gamma table whenever a contrast change is needed.  we do not touch the gamma table 
%   when blank frames are shown.  it is unknown what value to use for D in order to avoid 
%   glitching behavior, so please test your movie.  (of course, the value you use for D 
%   should be compatible with your stimulus paradigm!)  if [] or not supplied, do nothing special.
% <trialtask> (optional) is {A B C D E F G H I} where
%     A indicates the trial design, a matrix of dimensions T x F.  T corresponds to different trials.
%       F corresponds to the total number of frames in the movie.  each row should be 0 except 
%       for a consecutive string of 1s indicating the duration of the trial.  
%       trials should be mutually exclusive with respect to frames.
%     B is the fraction of trials (on average) that should present a dot.  we randomly flip a coin
%       for each trial to decide whether a dot is presented on that trial.  if negative, then
%       we enforce the fact that a dot cannot be presented on two successive trials.  can also
%       be of the form {B C} where B is as usual (but cannot be negative) and C is a cell vector
%       of trial numbers that go together.  for example, if C is {[1 3 5] [2 4]}, then a coin
%       is flipped for the first group, and if we decide to show a dot, the dot is shown
%       identically for the 1st, 3rd, and 5th trials, and similarly for the second group
%       (consisting of the 2nd and 4th trials).
%     C is a cell vector of elements. each element should be a 2 x V matrix, each column 
%       indicating a valid location for the dot.  the units should be signed x- and y-coordinates 
%       in pixel units and is to be interpreted relative to the fixation location.  note that
%       these locations (and the <trialtask> stuff) are not affected by <movieflip>.
%     D is a vector 1 x T with the mapping from trials to the elements of C.
%     E is a uint8 vector of dimensions 1 x 3, indicating the color to use for the dot.
%     F is the size in pixels for the dot.  if negative, then when using <maskimages>, the
%       dot is restricted to be within the mask.
%     G is a non-negative number in [0,1] indicating the alpha value to use for the dot.
%     H is the positive number of frames to show the dot for.  should be less than or equal to
%       the shortest trial in A.  can also be a negative number, in which case we only start
%       the dot at integral multiples of that number (e.g. -2 means show the dot starting at
%       the first frame, the third frame, the fifth frame, and so on).
%     I (optional) is a previously obtained trialoffsets.  If supplied, we re-use the 
%       computed behavior as stored in trialoffsets.
%   if supplied, we randomly choose trials to present a dot and during those trials we 
%   present the dot at a random location and at a random point in time.
%   if [] or not supplied, do nothing special.
% <maskimages> (optional) is a .mat file with 'maskimages' as a double A x B x M matrix
%   with different masks along the third dimension.  can be a cell vector, in which case
%   we just concatenate all masks together (the A and B dimensions should be consistent
%   across cases).  masks are referred to by their index (1-indexed).  <maskimages> can
%   also be the double matrix directly.  <maskimages> can also be a cell vector of
%   double elements, each of which is A x B x M.  masks should have values in [0,1] where
%   1 means pass, 0 means show background, and fractional values allow blending.
%   <maskimages> can also use uint8 format (to save memory); in this case, values should
%   be between 0 and 255 which we automatically map to 0 and 1.
% <specialoverlay> (optional) is a uint8 image matrix with four channels along the third
%   dimension (the last gives the alpha channel).  if supplied, this is an image that gets
%   drawn on top of the stimulus but below the fixation.
%
% return <timeframes> as a 1 x size(<frameorder>,2) vector with the time of each frame showing.
%   (time is relative to the time of the first frame.)
% return <timekeys> as a record of the input detected.  the format is {(time button;)*}.
%   where button is a string (single button depressed) or a cell vector of strings (multiple
%   buttons depressed).  for regular button presses, the recorded time is what KbCheck returns.
%   the very first entry in <timekeys> is special ('absolutetimefor0') and indicates the absolute
%   time corresponding to the time of the first frame; all other time entries are relative to the
%   time of the first frame.
% return <digitrecord> as {A B C} where A is a 1 x size(<frameorder>,2) vector with the digit (0-9)
%   shown on each frame (only the onsets of digits are recorded; the rest of the entries are NaN),
%   and B and C are auxiliary items that are useful for replicating the exact behavior (through
%   appropriate call to <fixationorder>).  this input will be returned as [] if <fixationorder> 
%   is not the {A B C D E F G H} case.
% return <trialoffsets> as 2 x size(<frameorder>,2) with NaNs in all columns except those columns
%   corresponding to the presentation of the dot; for these columns, the first and second rows
%   have the x- and y-offsets of the dot in pixels, respectively.  note that in this case,
%   positive means to the right and to the top.
%
% Here are some additional notes:
% - What happens in the presentation of the movie:
%     First, we fill the background with gray and draw the fixation.
%     Then, we wait for a key from any keyboard (see <triggerkey>).
%       (You can toggle a safe mode by pressing '='.  In the safe mode,
%       nothing will happen until '=' is pressed again.)
%     Next, we wait until next vertical retrace and then issue 
%       <triggerfun> and proceed to show the movie.
%     In the movie, each frame either results in filling with gray (i.e.
%       when <frameorder> is 0) or results in showing an image,
%       and then the fixation is drawn.
%     Finally, we fill the background with gray and draw the fixation.
% - Before starting the movie presentation, we call Priority(MaxPriority) and hide the cursor.
%   At the end of the presentation, we restore the original priority and show the cursor.
% - We attempt to achieve frame-accurate showing of the movie.  If we glitch (i.e. we
%   show a frame too late), we attempt to catch up using a simple strategy --- try to show
%   the next frame at the ideal/perfect time.  However, there is no guarantee that
%   catching up will actually succeed (e.g. if the frame rate is really high).
%   Testing your movie is key.
% - In processing the images for the movie, we perform the following in order (if applicable):
%   Mask the image, flip the image, scale the dimensions of the image, and offset the image.
%   Masking involves converting the image to double, performing the mask, and converting
%   back to uint8, so beware of numerical precision issues.
% - We do not attempt to pre-call any functions.  You should make sure to perform a dry run of
%   your movie to make sure all mex files, functions, etc. are cached (for best performance).
% - During the movie presentation, we KbCheck all devices.  Note that KbCheck reports 
%   buttons for as long as they are held down.  The ESCAPE key forces an immediate exit 
%   from the movie.  
% - Even if <detectinput>, it is possible that there is no time to actually read input.
%   So it is important to test your particular setup!
%
% history:
% 2015/01/25 - fill gray is now whole screen; new implementation of the circshifting;
%              movierect defined right before use
% 2014/10/21 - implement colors for the {A C X Y Z} case of <fixationorder>
% 2014/10/15 - implement the H argument for <fixationorder>
% 2014/10/08 - implement the {A C X Y Z} case for <fixationorder>.  implement special B case for <trialtask>.
%              institute capital letters in addition to the digits for the fixation digits stuff.
% 2014/02/17 - allow <maskimages> to be uint8
% 2013/12/20 - allow framecolor to be alpha values
% 2013/08/20 - implement negative case for B of <trialtask>,
%              implement negative case for F of <trialtask>.
% 2013/08/18 - add input <specialoverlay>.  add another special case 
%              for <frameorder>.  add <maskimages>.  tweak the call to sound.m.
% 2013/05/18 - allow trialtask{8} to be a negative number
% 2013/05/14 - digitrecord is now returned as a cell vector; fixationorder can allow an existing digitrecord
% 2013/05/14 - allow <frameorder> to be a matrix; trialtask can allow an existing trialoffsets
% 2012/11/04 - add <fixationsize> image case.
% 2012/09/11 - add <trialtask>
% 2011/11/02 - add F to <fixationorder>; allow fourth argument of <fixationorder> to be negative
% 2011/11/02 - fix bug (would have crashed)
% 2011/10/26 - make <fixationorder> more flexible; make sure flipping happens for fixation stuff
% 2011/10/23 - tweak the <fixationorder> case to become {A B C D E}
% 2011/10/22 - add <fixationorder> {A B C D} case; add output <digitrecord>
% 2011/10/13 - add <specialcon>
% 2011/09/16 - add <triggerkey> and a = safe mode
% 2011/07/30 - add special entry for 'absolutetimefor0'
% 2011/04/22 - make parent directory of <framefiles> automatically.
% 2011/04/22 - fix <framefiles> bug by making sure the images are written out AFTER the flip.
%              (it previously was not writing out the last frame.)
% 
% example:
% pton;
% [timeframes,timekeys,digitrecord,trialoffsets] = ptviewmovie(uint8(255*rand(100,100,3,100)),[],[],2);
% ptoff;

% to do:
% - Rush? [probably not]
% - test: VBLSyncTest
% - useful for reference: [times,badout,misses]=CheckFrameTiming(100,1,10,5,0,0,0);
% - speedup: [resident [texidresident]] = Screen('PreloadTextures', windowPtr [, texids]);
% - async flips was causing problems
% - using "don't clear" in flip was causing problems.

% input
if ~exist('frameorder','var') || isempty(frameorder)
  frameorder = [];  % deal with later
end
if ~exist('framecolor','var') || isempty(framecolor)
  framecolor = [];  % deal with later
end
if ~exist('frameduration','var') || isempty(frameduration)
  frameduration = 15;
end
if ~exist('fixationorder','var') || isempty(fixationorder)
  fixationorder = [];  % deal with later
end
if ~exist('fixationcolor','var') || isempty(fixationcolor)
  fixationcolor = uint8([255 255 255]);
end
if ~exist('fixationsize','var') || isempty(fixationsize)
  fixationsize = 5;
end
if ~exist('grayval','var') || isempty(grayval)
  grayval = uint8(127);
end
if ~exist('detectinput','var') || isempty(detectinput)
  detectinput = 1;
end
if ~exist('wantcheck','var') || isempty(wantcheck)
  wantcheck = 1;
end
if ~exist('offset','var') || isempty(offset)
  offset = [0 0];
end
if ~exist('moviemask','var') || isempty(moviemask)
  moviemask = [];
end
if ~exist('movieflip','var') || isempty(movieflip)
  movieflip = [0 0];
end
if ~exist('scfactor','var') || isempty(scfactor)
  scfactor = 1;
end
if ~exist('allowforceglitch','var') || isempty(allowforceglitch)
  allowforceglitch = 0;
end
if ~exist('triggerfun','var') || isempty(triggerfun)
  triggerfun = [];
end
if ~exist('framefiles','var') || isempty(framefiles)
  framefiles = [];
end
if ~exist('frameskip','var') || isempty(frameskip)
  frameskip = 1;
end
if ~exist('triggerkey','var') || isempty(triggerkey)
  triggerkey = [];
end
if ~exist('specialcon','var') || isempty(specialcon)
  specialcon = [];
end
if ~exist('trialtask','var') || isempty(trialtask)
  trialtask = [];
end
if ~exist('maskimages','var') || isempty(maskimages)
  maskimages = [];
end
if ~exist('specialoverlay','var') || isempty(specialoverlay)
  specialoverlay = [];
end
if ischar(images)
  images = {images};
end
if ischar(maskimages)
  maskimages = {maskimages};
end
if size(fixationsize,1) == 1 && length(fixationsize) == 1
  fixationsize = [fixationsize 0];
end
wantframefiles = ~isempty(framefiles);
if ~isempty(framefiles)
  if ischar(framefiles)
    framefiles = {framefiles []};
  end
end
focase3 = iscell(fixationorder) && length(fixationorder{2})==2;
focase4 = iscell(fixationorder) && length(fixationorder{2})==1;
if focase3 && (length(fixationorder) == 5)
  fixationorder{6} = [];
end
if focase3 && isempty(fixationorder{6})
  fixationorder{6} = 0;
end
if (focase3 && length(fixationorder) < 8) || (focase3 && isempty(fixationorder{8}))
  fixationorder{8} = [0 0];
end
if focase3 && ~isempty(specialcon)
  if fixationorder{6}==0
    specialcon{3} = repmat([255 255 255],[fixationorder{5} 1]);
  else
    specialcon{3} = repmat([255 255 255],[fixationorder{5} 1]);
    specialcon{3}(end-1,:) = [0 0 0];
  end
end

%%%%%%%%%%%%%%%%% DEAL WITH THE IMAGES

% load in the images
fprintf('loading images: starting...\n');
if iscell(images) && ischar(images{1})
  moviefile = images;
  images = [];
  for p=1:length(moviefile)
    temp = load(moviefile{p},'images');
    if ~isempty(images) & size(images,3)==1 & size(temp.images,3)==3  % do we need to make images into color?
      images = repmat(images,[1 1 3]);
    end
    if size(images,3)==3 & size(temp.images,3)==1  % do we need to make the temp.images into color?
      temp.images = repmat(temp.images,[1 1 3]);
    end
    images = cat(4,images,temp.images);
  end
  clear temp;
end
assert(isa(images,'uint8') || all(cellfun(@(x) isa(x,'uint8'),images)));  % check sanity
fprintf('loading images: done\n');

% load in the maskimages
if iscell(maskimages) && ischar(maskimages{1})
  fprintf('loading maskimages: starting...\n');
  moviefile = maskimages;
  maskimages = [];
  for p=1:length(moviefile)
    temp = load(moviefile{p},'maskimages');
    maskimages = cat(3,cast(maskimages,class(temp.maskimages)),temp.maskimages);
  end
  clear temp;
  fprintf('loading maskimages: done\n');
end

% concatenate all maskimages together
if iscell(maskimages)
  maskimages = cat(3,maskimages{:});
end

% convert maskimages to uint8 alpha images if necessary
if ~isa(maskimages,'uint8')
  maskimages = uint8(255*maskimages);
end

% deal with mask
if ~isempty(moviemask)
  fprintf('applying mask: starting...\n');
% OLD WAY:
%   moviemask = repmat(moviemask,[1 1 size(images,3) size(images,4)]);
%   images = uint8(moviemask * double(grayval) + (1 - moviemask) .* double(images));
%   clear moviemask;
  if iscell(images)
    for p=1:length(images)
      images{p} = uint8(bsxfun(@plus,moviemask * double(grayval),bsxfun(@times,1 - moviemask,double(images{p}))));
    end
  else
    chunks = chunking(1:size(images,4),10);
    for p=1:length(chunks)  % to save on memory
      images(:,:,:,chunks{p}) = uint8(bsxfun(@plus,moviemask * double(grayval),bsxfun(@times,1 - moviemask,double(images(:,:,:,chunks{p})))));
    end
  end
  fprintf('applying mask: done\n');
end

% deal with movieflip
if movieflip(1) && movieflip(2)
  flipfun = @(x) flipdim(flipdim(x,1),2);
elseif movieflip(1)
  flipfun = @(x) flipdim(x,1);
elseif movieflip(2)
  flipfun = @(x) flipdim(x,2);
else
  flipfun = @(x) x;
end

%%%%%%%%%%%%%%%%% PREP

% get information about the PT setup
win = firstel(Screen('Windows'));
rect = Screen('Rect',win);

% calc
if iscell(images)
  d1images = size(images{1},1);
  d2images = size(images{1},2);
  if size(images{1},4) > 1
    dimwithim = 4;
  else
    dimwithim = 3;
  end
  csimages = cumsum(cellfun(@(x) size(x,dimwithim),images));
  nimages = csimages(end);
else
  d1images = size(images,1);
  d2images = size(images,2);
  nimages = size(images,4);
end

% deal with input (finally)
if isempty(frameorder)
  frameorder = 1:nimages;
end
if isempty(framecolor)
  framecolor = 255*ones(size(frameorder,2),3);
end
if isempty(fixationorder)
  fixationorder = ones(1,1+size(frameorder,2)+1);
end

% prepare fixationrect (movierect is now computed right before it is needed)
if size(fixationsize,1) == 1  % dot case
  % easy case
  if ~focase4
    if focase3
      fixationoff0 = repmat(fixationorder{8},[1 2]);  % in this case, we can have offsets specified
    else
      fixationoff0 = 0;
    end
    fixationrect = CenterRect([0 0 2*fixationsize(1) 2*fixationsize(1)],rect) + [offset(1) offset(2) offset(1) offset(2)] + fixationoff0;  % allow doubling of fixationsize for room for anti-aliasing
  % special case of multiple digits
  else
    fixationrect = {};
    for p=1:length(fixationorder{3})
      fixationrect{p} = CenterRect([0 0 2*fixationsize(1) 2*fixationsize(1)],rect) + [offset(1) offset(2) offset(1) offset(2)] + repmat(fixationorder{3}{p},[1 2]);
    end
  end
else  % image case
  fixationrect = CenterRect([0 0 size(fixationsize,2) size(fixationsize,1)],rect) + [offset(1) offset(2) offset(1) offset(2)];
end
if ~iscell(fixationrect)
  fixationrect = {fixationrect};
end
if ~isempty(specialoverlay)
  overlayrect = CenterRect([0 0 size(specialoverlay,2) size(specialoverlay,1)],rect) + [offset(1) offset(2) offset(1) offset(2)];
end

% prepare fixation image
if ~iscell(fixationorder)

  fixationcase = any(fixationorder < 0);  % 0 means regular case, 1 means negative-integer case

  % dot case
  if size(fixationsize,1) == 1

      % 2*fixationsize x 2*fixationsize x 3 x N; several different uint8 solid colors
    fixationimage = zeros([2*fixationsize(1) 2*fixationsize(1) 3 size(fixationcolor,1)]);
    temp = find(makecircleimage(2*fixationsize(1),fixationsize(1)/2-fixationsize(2)));  % this tells us where to insert color
    for p=1:size(fixationcolor,1)
      temp0 = zeros([2*fixationsize(1)*2*fixationsize(1) 3]);  % everything is initially black
      temp0(temp,:) = repmat(fixationcolor(p,:),[length(temp) 1]);  % insert color in the innermost circle
      fixationimage(:,:,:,p) = reshape(temp0,[2*fixationsize(1) 2*fixationsize(1) 3]);
    %OLD:    fixationimage(:,:,:,p) = repmat(reshape(fixationcolor(p,:),[1 1 3]),[2*fixationsize 2*fixationsize]);
    end
    fixationalpha = 255*makecircleimage(2*fixationsize(1),fixationsize(1)/2);  % 2*fixationsize x 2*fixationsize; double [0,255] alpha values (255 in circle, 0 outside)
  
  % image case
  else
  
    fixationimage = zeros([size(fixationsize,1) size(fixationsize,2) 3 size(fixationcolor,1)]);
    for p=1:size(fixationcolor,1)
      temp0 = repmat(fixationcolor(p,:),[size(fixationsize,1)*size(fixationsize,2) 1]);  % insert color everywhere
      fixationimage(:,:,:,p) = reshape(temp0,[size(fixationsize,1) size(fixationsize,2) 3]);
    end
    fixationalpha = 255*fixationsize;
    
  end

else

  % prepare digits as 2*fixationsize x 2*fixationsize x 3 x N; uint8 format
  digits = drawtexts(2*fixationsize(1),0,0,'Helvetica',fixationorder{1}, ...
                     [1 1 1],[0 0 0],mat2cell(['0':'9' 'A':'Z'],1,ones(1,10+26)));
  digits = round(normalizerange(digits,0,1));  % binarize so that values are either 0 or 1
  digsize = sizefull(digits,3);
  digits = repmat(vflatten(digits),[1 3]);  % T x 3
  if length(grayval)==1
    grayval0 = repmat(grayval,[1 3]);
  else
    grayval0 = grayval;
  end
  whzero = digits(:,1)==0;
  digits(whzero,:) = repmat(grayval0,[sum(whzero) 1]);  % 0 maps to the grayval
  digits(~whzero,:) = 255;  % 1 maps to the white value (remember to reserve this in the <specialcon> case)
  fixationimage = uint8(permute(reshape(digits,digsize(1),digsize(2),digsize(3),3),[1 2 4 3]));
    % make copy with black
  fixationimageB = fixationimage;
  fixationimageB(fixationimageB==255) = choose(isempty(specialcon),0,254);  % 1 maps to the black value (remember to reserve this in the <specialcon> case)
  fixationimage = cat(4,fixationimage,fixationimageB);
    % finally, add pure gray frame
  fixationimage(:,:,:,end+1) = repmat(reshape(grayval0,[1 1 3]),[size(fixationimage,1) size(fixationimage,2)]);
  
  % prepare alpha as 2*fixationsize x 2*fixationsize x N; uint8 [0,255] alpha values
  if (focase3 & fixationorder{3}) | (focase4 & fixationorder{2})
    % the digits themselves are the 255 alpha values
    fixationalpha = uint8(reshape(255*double(~whzero),digsize(1),digsize(2),digsize(3)));
    fixationalpha = cat(3,fixationalpha,fixationalpha);
    fixationalpha(:,:,end+1) = 0;  % the last frame is completely transparent
  else
    % (255 in circle, 0 outside). gradual ramp.
    fixationalpha = repmat(uint8(255*makecircleimage(2*fixationsize(1), ...
                    fixationsize(1)/2-fixationsize(2),[],[],fixationsize(1)/2)),[1 1 (10+26)*2+1]);
  end
  
end

% prepare trial image
if ~isempty(trialtask)
  trialimage = repmat(reshape(trialtask{5},1,1,[]),[2*abs(trialtask{6}) 2*abs(trialtask{6})]);
  trialalpha = uint8(trialtask{7} * (255*makecircleimage(2*abs(trialtask{6}),abs(trialtask{6})/2)));
end

% now deal with flipping of fixation stuff
fixationimage = flipdims(fixationimage,movieflip);
fixationalpha = flipdims(fixationalpha,movieflip);

% now deal with flipping of overlay stuff
if ~isempty(specialoverlay)
  specialoverlay = flipdims(specialoverlay,movieflip);
end

% prepare window for alpha blending
Screen('BlendFunction',win,GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);

% init variables, routines, constants
timeframes = repmat(NaN,[1 floor((size(frameorder,2)-1)/frameskip)+1]);
timekeys = {};
digitrecord = [];
trialoffsets = [];
digitframe = [];
digitpolarity = [];
when = 0;
oldPriority = Priority(MaxPriority(win));
HideCursor;
mfi = Screen('GetFlipInterval',win);  % re-use what was found upon initialization!
filtermode = choose(scfactor==1,0,1);
getoutearly = 0;
glitchcnt = 0;
sound(zeros(1,100),1000);

% precomputations for case when images is a cell of uint8
iscellimages = iscell(images);
if iscellimages
  whs = zeros(size(frameorder,2),1+size(frameorder,1));
  for frame=1:size(frameorder,2)
    if frameorder(1,frame) ~= 0
      whs(frame,1) = firstel(find(frameorder(1,frame) <= csimages));
      whs(frame,2) = size(images{whs(frame,1)},dimwithim) - (csimages(whs(frame,1))-frameorder(1,frame));
      if size(frameorder,1)==2
        whs(frame,3) = frameorder(2,frame);
      elseif size(frameorder,1)==3
        whs(frame,3:4) = frameorder(2:3,frame);
      end
    end
  end
end

% make directory
if wantframefiles
  mkdirquiet(stripfile(framefiles{1}));
end

% precompute cluts
if ~isempty(specialcon)

  % load and prep the cal
  [cal,cals] = LoadCalFile(specialcon{1});
  cal = SetGammaMethod(cal,0);

  % how many entries do we dedicate to the display of the images?
  nn = 256-size(specialcon{3},1);
  
  % what is a list of all the contrast levels that we will be using?
  allcons = union([100],specialcon{2});  % include 100 since we will use that before and after movie

  % do it
  specialcluts = [];
  for p=1:length(allcons)
    mn = 0.5 - 0.5*(allcons(p)/100);
    mx = 0.5 + 0.5*(allcons(p)/100);
    linearValues = [ones(3,1)*linspace(mn,mx,nn) specialcon{3}'];
    specialcluts(:,:,p) = PrimaryToSettings(cal,linearValues)';
  end

end

% deal with figuring out digit sequence for special fixation task
if focase3

  % if we have an existing record, use it
  if length(fixationorder) >= 7 && ~isempty(fixationorder{7})
  
    digitrecord = fixationorder{7}{1};
    digitframe = fixationorder{7}{2};
    digitpolarity = fixationorder{7}{3};

  % otherwise, we have to compute
  else

      % this will record onsets (for user consumption)
    digitrecord = NaN*zeros(1,ceil(size(frameorder,2)/sum(fixationorder{2})) * sum(fixationorder{2}));
      % this will tell us what to put on each frame [entries 1-10 map to '0':'9' white.
      %                                              entries 37-46 map to '0':'9' black.
      %                                              entry 73 maps to gray.]
    digitframe = digitrecord;
    digitpolarity = digitrecord;  % 0 means white.  1 means black.
    lastdigit = NaN;
    p = 1; cnt = 0; repeated = 0;
    while 1
      if p > size(frameorder,2)
        break;
      end
      if fixationorder{4}==1
        while 1
          digit = floor(rand*10);
          if ~isequal(digit,lastdigit)
            break;
          end
        end
      elseif fixationorder{4}==0
        digit = floor(rand*10);
      else
        case1 = fixationorder{4} < 0 && repeated;  % if we want to avoid triplets and we have just repeated...
        case2 = fixationorder{4} < 0 && ~repeated;  % if we want to avoid triplets and we haven't just repeated...
        case3 = fixationorder{4} > 0;  % if we just want to plow ahead
        if (case2 || case3) && (rand <= abs(fixationorder{4}))
          digit = lastdigit;
          repeated = 1;
          if isnan(digit)
            digit = 0;
            repeated = 0;
          end
        else
          repeated = 0;
          tttt = setdiff(0:9,lastdigit);
          ssss = randperm(length(tttt));
          digit = tttt(ssss(1));
        end
      end
      digitrecord(p) = digit;
      digitframe(p-1+(1:sum(fixationorder{2}))) = [repmat(digit+1,[1 fixationorder{2}(1)]) repmat((10+26)*2+1,[1 fixationorder{2}(2)])];
      digitpolarity(p-1+(1:sum(fixationorder{2}))) = mod(cnt,2);
      lastdigit = digit;
      p = p + sum(fixationorder{2});
      cnt = cnt + 1;
    end
  
    % now truncate
    digitrecord = digitrecord(1:size(frameorder,2));
    digitframe = digitframe(1:size(frameorder,2));
    digitpolarity = digitpolarity(1:size(frameorder,2));

  end
  
end

% figure out trialtask stuff
if ~isempty(trialtask)

  % if user supplied trialoffsets, use it
  if length(trialtask) >= 9
    trialoffsets = trialtask{9};
  
  % otherwise, compute it fresh
  else
    numtrials = size(trialtask{1},1);
    numframes = size(trialtask{1},2);
    
    % easy case (consecutive is okay)
    if iscell(trialtask{2}) || trialtask{2} > 0

      if iscell(trialtask{2})
        dogroups = find(rand(1,length(trialtask{2}{2})) <= trialtask{2}{1});  % indices of groups to show dot for
        dotrials = catcell(2,trialtask{2}{2}(dogroups));  % indices of trials to show dot on
          % figure out how these trials are grouped (we will use random-seed strategy)
        cnt = 1;
        dogroupings = [];
        for p=1:length(dogroups)
          dogroupings = [dogroupings cnt*ones(1,length(trialtask{2}{2}{dogroups(p)}))];
          cnt = cnt + 1;
        end
        assert(length(dotrials)==length(dogroupings));
      else
        dotrials = find(rand(1,numtrials) <= trialtask{2});  % indices of trials to show a dot on
        dogroupings = 1:length(dotrials);
      end

    % hard case (consecutive is not okay)
    else
      dotrials = zeros(1,numtrials);
      lastdo = 0;
      cnt = 1;
      while cnt <= numtrials
        if lastdo
          lastdo = 0;
          cnt = cnt + 1;
        else
          dotrials(cnt) = rand <= -trialtask{2};
          lastdo = dotrials(cnt);
          cnt = cnt + 1;
        end
      end
      dotrials = find(dotrials);
      dogroupings = 1:length(dotrials);
    end
    
    % have a random starting point at least
    clock0 = sum(100*clock);
    
    % compute
    trialoffsets = NaN*zeros(2,numframes);  % compute x- and y-offsets for each frame
    for pp=1:length(dotrials)
      
      % this ensures that trials that are grouped together will experience the same
      % dot parameters.  thus, it is here that physicality of dots is enforced 
      % (all four tasks see the same dots).
      setrandstate({clock0+999*dogroupings(pp)});
      
      dotframes = find(trialtask{1}(dotrials(pp),:));  % indices of frames to show the dot on
        % choose a random duration within the trial by ignoring a random number of frames at the beginning
      if trialtask{8} > 0
        offset0 = randintrange(0,length(dotframes)-trialtask{8});
        dotframes = dotframes(offset0 + (1:trialtask{8}));
      else
        offset0 = randintrange(0,floor((length(dotframes)-(-trialtask{8})) / (-trialtask{8}))) * (-trialtask{8});
        dotframes = dotframes(offset0 + (1:(-trialtask{8})));
      end
      locs = trialtask{3}{trialtask{4}(dotrials(pp))};  % 2 x L matrix of potential locations
      whichloc = ceil(rand*size(locs,2));  % pick one location. this is the index.
      trialoffsets(:,dotframes) = repmat(locs(:,whichloc),[1 length(dotframes)]);
    end
    
    % restore randomness
    setrandstate;
    
  end

end

%%%%%%%%%%%%%%%%% START THE EXPERIMENT

% draw the background, overlay, and fixation
Screen('FillRect',win,grayval,rect);
if ~isempty(specialoverlay)
  texture = Screen('MakeTexture',win,specialoverlay);
  Screen('DrawTexture',win,texture,[],overlayrect,[],0);
  Screen('Close',texture);
end
if focase3
  texture = Screen('MakeTexture',win,cat(3,fixationimage(:,:,:,1),fixationalpha(:,:,1)));
elseif focase4
  texture = {};
  for p=1:length(fixationrect)
    texture{p} = Screen('MakeTexture',win,cat(3,fixationimage(:,:,:,1),fixationalpha(:,:,1)));
  end
else
  if fixationcase==0
    texture = Screen('MakeTexture',win,cat(3,fixationimage,uint8(fixationorder(1)*fixationalpha)));
  else
    texture = Screen('MakeTexture',win,cat(3,fixationimage(:,:,:,-fixationorder(1)),uint8(fixationorder(end)*fixationalpha)));
  end
end
if ~iscell(texture)
  texture = {texture};
end
for p=1:length(texture)
  Screen('DrawTexture',win,texture{p},[],fixationrect{p},[],0);
  Screen('Close',texture{p});
end
if ~isempty(specialcon)
  Screen('LoadNormalizedGammaTable',win,specialcluts(:,:,allcons==100),1);  % use loadOnNextFlip!
  lastsc = 100;
end
Screen('Flip',win);

% wait for a key press to start
fprintf('press a key to begin the movie. (make sure to turn off network, energy saver, spotlight, software updates! mirror mode on!)\n');
safemode = 0;
while 1
  [secs,keyCode,deltaSecs] = KbWait(-3,2);
  temp = KbName(keyCode);
  if isequal(temp(1),'=')
    if safemode
      safemode = 0;
      fprintf('SAFE MODE OFF (the scan can start now).\n');
    else
      safemode = 1;
      fprintf('SAFE MODE ON (the scan will not start).\n');
    end
  else
    if safemode
    else
      if isempty(triggerkey) || isequal(temp(1),triggerkey)
        break;
      end
    end
  end
end

  % IS THE PREVIOUS LINE (RELATED TO KBWAIT) RELIABLE?  SHOULD WE USE SOMETHING DIFFERENT, LIKE:??
  % % just wait for any press
  %    % KbWait is unreliable probably would need a device input as well
  %    % but this is not an option (Psychtoolbox 1.0.5)
  %    % pause;
  %    iwait = 0;
  %    while ~iwait
  %        [~, ~, c] = KbCheck(-1);
  %        if find(c) == KbName('t')
  %            iwait = 1;
  %        end
  %    end

% wait until next vertical retrace (to reduce run-to-run variability)
Screen('Flip',win);

% issue the trigger and record it
if ~isempty(triggerfun)
  feval(triggerfun);
  timekeys = [timekeys; {GetSecs 'trigger'}];
end

% show the movie
framecnt = 0;
for frame=1:frameskip:size(frameorder,2)+1
  framecnt = framecnt + 1;
  frame0 = floor(frame);

  % we have to wait until the last frame is done.  so this is how we hack that in.
  if frame0==size(frameorder,2)+1
    while 1
      if GetSecs >= when
        getoutearly = 1;
        break;
      end
    end
  end

  % get out early?
  if getoutearly
    break;
  end

  % if special 0 case, just fill with gray
  MI = [];
  if frameorder(1,frame0) == 0
    Screen('FillRect',win,grayval);   % REMOVED! this means do whole screen.    % ,movierect);

  % otherwise, make a texture, draw it at a particular position
  else
    extracircshift = [0 0];
    if iscellimages
      if dimwithim==4   % THIS IS VERY VERY UGLY
        switch size(whs,2)
        case 2
          txttemp = feval(flipfun,images{whs(frame0,1)}(:,:,:,whs(frame0,2)));
        case 3
          MI = maskimages(:,:,whs(frame0,3));
          txttemp = feval(flipfun,cat(3,images{whs(frame0,1)}(:,:,:,whs(frame0,2)),MI));
        case 4
          txttemp = feval(flipfun,images{whs(frame0,1)}(:,:,:,whs(frame0,2)));
          extracircshift = whs(frame0,3:4);
        end
        texture = Screen('MakeTexture',win,txttemp);
      else
        switch size(whs,2)
        case 2
          txttemp = feval(flipfun,images{whs(frame0,1)}(:,:,whs(frame0,2)));
        case 3
          MI = maskimages(:,:,whs(frame0,3));
          txttemp = feval(flipfun,cat(3,images{whs(frame0,1)}(:,:,whs(frame0,2)),MI));
        case 4
          txttemp = feval(flipfun,images{whs(frame0,1)}(:,:,whs(frame0,2)));
          extracircshift = whs(frame0,3:4);
        end
        texture = Screen('MakeTexture',win,txttemp);
      end
    else
      switch size(frameorder,1)
      case 1
        txttemp = feval(flipfun,images(:,:,:,frameorder(1,frame0)));
      case 2
        MI = maskimages(:,:,frameorder(2,frame0));
        txttemp = feval(flipfun,cat(3,images(:,:,:,frameorder(1,frame0)),MI));
      case 3
        txttemp = feval(flipfun,images(:,:,:,frameorder(1,frame0)));
        extracircshift = frameorder(2:3,frame0)';
      end
      texture = Screen('MakeTexture',win,txttemp);
    end
    movierect = CenterRect([0 0 round(scfactor*d2images) round(scfactor*d1images)],rect) + ...
                repmat(extracircshift([2 1]),[1 2]) + ...
                [offset(1) offset(2) offset(1) offset(2)];
    if size(framecolor,2) == 3  % the usual case
      Screen('DrawTexture',win,texture,[],movierect,0,filtermode,1,framecolor(frame0,:));
    else
      Screen('DrawTexture',win,texture,[],movierect,0,filtermode,framecolor(frame0));
    end
    Screen('Close',texture);
  end
  
  % draw the overlay
  if ~isempty(specialoverlay)
    texture = Screen('MakeTexture',win,specialoverlay);
    Screen('DrawTexture',win,texture,[],overlayrect,0,0);
    Screen('Close',texture);
  end
  
  % draw the fixation
  if focase3
    if fixationorder{6}==1
      if digitframe(frame) == 73
        whtodo = 73;
      else
        whtodo = digitframe(frame) + (10+26)*digitpolarity(frame);
      end
    else
      whtodo = digitframe(frame);
    end
    texture = Screen('MakeTexture',win,cat(3,fixationimage(:,:,:,whtodo),fixationalpha(:,:,whtodo)));
  elseif focase4
    texture = {};
    for p=1:length(fixationrect)
      dg0 = fixationorder{4}(p,frame);
      cl0 = fixationorder{5}(p,frame);
      if isnan(dg0)
        whtodo = 73;
      else
        whtodo = dg0 + (10+26)*(cl0==0);
      end
      texture{p} = Screen('MakeTexture',win,cat(3,fixthecolor(fixationimage(:,:,:,whtodo),cl0),fixationalpha(:,:,whtodo)));
    end
  else
    if fixationcase==0
      texture = Screen('MakeTexture',win,cat(3,fixationimage,uint8(fixationorder(1+frame0)*fixationalpha)));
    else
      texture = Screen('MakeTexture',win,cat(3,fixationimage(:,:,:,-fixationorder(1+frame0)),uint8(fixationorder(end)*fixationalpha)));
    end
  end
  if ~iscell(texture)
    texture = {texture};
  end
  for p=1:length(texture)
    Screen('DrawTexture',win,texture{p},[],fixationrect{p},0,0);
    Screen('Close',texture{p});
  end
  
  % draw the trial task dot
  if ~isempty(trialtask)
    if ~all(isnan(trialoffsets(:,frame0)))
      trialrect = CenterRect([0 0 2*abs(trialtask{6}) 2*abs(trialtask{6})],rect) + ...
                  [offset(1) offset(2) offset(1) offset(2)] + ...
                  repmat(trialoffsets(:,frame0)' .* [1 -1],[1 2]);
                  % multiply y-coordinate by -1 because in PT, positive means down

      %%%% voodoo detour: restrict the trial task dot to the mask, if there is one

      % this is the easy case.
      % if there is no mask happening OR if the trialtask{6} is positive,
      % then do the regular thing.
      if isempty(MI) || trialtask{6} > 0
        texture = Screen('MakeTexture',win,cat(3,trialimage,trialalpha));
        Screen('DrawTexture',win,texture,[],trialrect,0,0);

      % this is the hard case.
      else
        movierect0 = ceil(movierect);  % this rect is the master mask (the stimulus)
        trialrect0 = ceil(trialrect);  % this rect is the trial task dot mask
        rect0 = ClipRect(movierect0,trialrect0);  % this is the intersection
          % extract from the master alpha
        extractMA =         MI(rect0(2)-movierect0(2) + (1:(rect0(4)-rect0(2))), ...
                               rect0(1)-movierect0(1) + (1:(rect0(3)-rect0(1))));
          % extract from the trial image
        extractTI = trialimage(rect0(2)-trialrect0(2) + (1:(rect0(4)-rect0(2))), ...
                               rect0(1)-trialrect0(1) + (1:(rect0(3)-rect0(1))),:);
          % extract from the trial alpha
        extractTA = trialalpha(rect0(2)-trialrect0(2) + (1:(rect0(4)-rect0(2))), ...
                               rect0(1)-trialrect0(1) + (1:(rect0(3)-rect0(1))));
          % construct the final alpha
        extractFA = uint8((double(extractMA)/255) .* double(extractTA));
          % proceed
        texture = Screen('MakeTexture',win,cat(3,extractTI,extractFA));
        Screen('DrawTexture',win,texture,[],rect0,0,0);
      end
      
      %%%% end voodoo
      
      Screen('Close',texture);
    end
  end

  % give hint to PT that we're done drawing
  Screen('DrawingFinished',win);
  
  % read input until we have to do the flip
  while 1
  
    % load the gamma table (for a future frame)
    if ~isempty(specialcon)
      frameL = frame0 + specialcon{4};
      if frameL <= size(frameorder,2)
        if frameorder(1,frameL)==0  % if blank frame, who cares, don't change
        else
          con = specialcon{2}(frameorder(1,frameL));
          if lastsc ~= con
%            sound(sin(1:100),1);
            Screen('LoadNormalizedGammaTable',win,specialcluts(:,:,allcons==con));  % don't use loadOnNextFlip!
            lastsc = con;
          end
        end
      end
    end

    % if we are in the initial case OR if we have hit the when time, then display the frame
    if when == 0 | GetSecs >= when
  
      % issue the flip command and record the empirical time
      [VBLTimestamp,StimulusOnsetTime,FlipTimestamp,Missed,Beampos] = Screen('Flip',win,when);
%      sound(sin(1:2000),100);
      timeframes(framecnt) = VBLTimestamp;

      % if we missed, report it
      if Missed > 0 & when ~= 0
        glitchcnt = glitchcnt + 1;
        didglitch = 1;
      else
        didglitch = 0;
      end
      
      % get out of this loop
      break;
    
    % otherwise, try to read input
    else
      if detectinput
        [keyIsDown,secs,keyCode,deltaSecs] = KbCheck(-3);  % all devices
        if keyIsDown

          % get the name of the key and record it
          kn = KbName(keyCode);
          timekeys = [timekeys; {secs kn}];

          % check if ESCAPE was pressed
          if isequal(kn,'ESCAPE')
            fprintf('Escape key detected.  Exiting prematurely.\n');
            getoutearly = 1;
            break;
          end

          % force a glitch?
          if allowforceglitch(1) && isequal(kn,'p')
            WaitSecs(allowforceglitch(2));
          end

        end
      end
    end

  end

  % write to file if desired
  if wantframefiles
    if isempty(framefiles{2})
      imwrite(Screen('GetImage',win),sprintf(framefiles{1},framecnt));
    else
      imwrite(uint8(placematrix(zeros([framefiles{2} 3]),Screen('GetImage',win))),sprintf(framefiles{1},framecnt));
    end
  end

  % update when
  if didglitch
    % if there were glitches, proceed from our earlier when time.
    % set the when time to half a frame before the desired frame.
    % notice that the accuracy of the mfi is strongly assumed here.
    when = (when + mfi / 2) + mfi * frameduration - mfi / 2;
  else
    % if there were no glitches, just proceed from the last recorded time
    % and set the when time to half a frame before the desired time.
    % notice that the accuracy of the mfi is only weakly assumed here,
    % since we keep resetting to the empirical VBLTimestamp.
    when = VBLTimestamp + mfi * frameduration - mfi / 2;  % should we be less aggressive??
  end
  
end

% draw the background, overlay, and fixation
Screen('FillRect',win,grayval,rect);
if ~isempty(specialoverlay)
  texture = Screen('MakeTexture',win,specialoverlay);
  Screen('DrawTexture',win,texture,[],overlayrect,[],0);
  Screen('Close',texture);
end
if focase3
  texture = Screen('MakeTexture',win,cat(3,fixationimage(:,:,:,1),fixationalpha(:,:,1)));
elseif focase4
  texture = {};
  for p=1:length(fixationrect)
    texture{p} = Screen('MakeTexture',win,cat(3,fixationimage(:,:,:,1),fixationalpha(:,:,1)));
  end
else
  if fixationcase==0
    texture = Screen('MakeTexture',win,cat(3,fixationimage,uint8(fixationorder(end)*fixationalpha)));
  else
    texture = Screen('MakeTexture',win,cat(3,fixationimage(:,:,:,-fixationorder(end-1)),uint8(fixationorder(end)*fixationalpha)));
  end
end
if ~iscell(texture)
  texture = {texture};
end
for p=1:length(texture)
  Screen('DrawTexture',win,texture{p},[],fixationrect{p},[],0);
  Screen('Close',texture{p});
end
if ~isempty(specialcon)
  Screen('LoadNormalizedGammaTable',win,specialcluts(:,:,allcons==100),1);  % use loadOnNextFlip!
end
Screen('Flip',win);

%%%%%%%%%%%%%%%%% CLEAN UP

% restore priority and cursor
Priority(oldPriority);
ShowCursor;

% adjust the times in timeframes and timekeys to be relative to the first time recorded.
% thus, time==0 corresponds to the showing of the first frame.
starttime = timeframes(1);
timeframes = timeframes - starttime;
if size(timekeys,1) > 0
  timekeys(:,1) = cellfun(@(x) x - starttime,timekeys(:,1),'UniformOutput',0);
end
timekeys = [{starttime 'absolutetimefor0'}; timekeys];

% report basic timing information to stdout
fprintf('we had %d glitches!\n',glitchcnt);
dur = (timeframes(end)-timeframes(1)) * (length(timeframes)/(length(timeframes)-1));
fprintf('projected total movie duration: %.10f\n',dur);
fprintf('frames per second: %.10f\n',length(timeframes)/dur);

% prepare output
digitrecord = {digitrecord digitframe digitpolarity};

% do some checks
if wantcheck
  ptviewmoviecheck(timeframes,timekeys,[],'t');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function im = fixthecolor(im,clr)

switch clr
case 0  % black
case 1  % white
case 2  % red
  im(:,:,2:3) = 0;
case 3  % green
  im(:,:,[1 3]) = 0;
case 4  % blue
  im(:,:,1:2) = 0;
case 5  % yellow
  im(:,:,3) = 0;
case 6  % magenta
  im(:,:,2) = 0;
case 7  % cyan
  im(:,:,1) = 0;
end



% JUNK
%   fixation to draw.  0 means do not draw anything.  1 means use the color
%   in the first row of <fixationcolor>, 2 means use the color in the
%   second row of <fixationcolor>, and so forth. 
%  Screen('FillOval',win,fixationcolor(fixationorder(1),:),fixationrect);
%    Screen('FillOval',win,fixationcolor(fixationorder(1+frame),:),fixationrect);
%  Screen('FillOval',win,fixationcolor(fixationorder(end),:),fixationrect);
