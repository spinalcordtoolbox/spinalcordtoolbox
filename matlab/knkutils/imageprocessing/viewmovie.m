function viewmovie(data,mode,rot,prct,autopermute,cmap,offset)

% function viewmovie(data,mode,rot,prct,autopermute,cmap,offset)
%
% <data> is A x B x C x T, where A x B indicates slices and T indicates different volumes.
%   T must be >= 1.
% <mode> (optional) is
%   0 means display stuff in a new figure window
%   A where A is a string with %d in it (e.g. 'blah/image%03d')
%     means write 1-indexed files like blah/image001.png, blah/image002.png, etc.
%     we automatically create the directory if necessary.
%   default: 0.
% <rot> (optional) is an integer indicating CCW in-slice rotation to apply. default: 0.
% <prct> (optional) is
%   A means the non-zero percentile (e.g. 1 for 1 percent) to normalize with.
%     we map the Ath and (100-A)th percentiles to black and white (using colormap gray).
%  -B means the non-zero percentile (e.g. -2 for 2 percent) to normalize with.
%     we calculate V, the (100-B)th percentile of the abs of the data, and then
%     map -V,0,V to blue,black,red (using colormap cmapsign).
%   [C D] means to normalize with this range.  we map the bottom and top of this range
%     to black and white (using colormap gray).
%   note that the percentile is determined from the first volume, and 
%   the same normalization is then applied to all volumes.
%   default: 1.
% <autopermute> (optional) is to automatically permute the smallest of A, B, C
%   into the third-dimension slot.  default: 0.
% <cmap> (optional) is the colormap to use.  if supplied, this overrides the default.
%   default is [] which means use the default colormap.
% <offset> (optional) is a non-negative integer to add to the 1-index for the filename
%   in the case that <mode> is A.  default: 0.
%
% when <mode> is 0, display <data> in a new figure window.  interact using the following
% (assuming Psychtoolbox is available):
%   space : play/pause
%   left arrow : go to previous volume (when paused)
%   right arrow : go to next volume (when paused)
%   , : rewind to the beginning
%   . : fastforward to the end
%   - : slower (slowest speed is 1 s for each frame)
%   = : faster (fastest speed is 10 ms for each frame)
%   ESCAPE : quit
% when <mode> is 1, write images according to the string specified by <mode>.
%
% example:
% viewmovie(randn(64,64,16,100));

% NOTE: WE COULD GO ALL THE WAY AND AUTOMATE THE CONVERSION OF IMAGE SEQUENCES INTO A QUICKTIME MOVIE USING QT_TOOLS OR SOMETHING...
%     however, there might still be that bug where the duration is always 1 s (or more?)

% constants
framedelay = 0.1;      % delay between successive frames (but this can be changed by the user)
framefactor = 1.5;     % framedelay increases/decreases by this factor
loopdelay = 0.01;      % delay in the input-checking loop
minframedelay = 0.01;  % minimum frame delay in seconds
maxframedelay = 1;     % maximum frame delay in seconds

% input
if ~exist('mode','var') || isempty(mode)
  mode = 0;
end
if ~exist('rot','var') || isempty(rot)
  rot = 0;
end
if ~exist('prct','var') || isempty(prct)
  prct = 1;
end
if ~exist('autopermute','var') || isempty(autopermute)
  autopermute = 0;
end
if ~exist('cmap','var') || isempty(cmap)
  cmap = [];
end
if ~exist('offset','var') || isempty(offset)
  offset = 0;
end

% deal with autopermute
if autopermute
  [d,ix] = min(fliplr(sizefull(data,3)));
  ix = 4-ix;
  data = permute(data,[setdiff(1:3,ix) ix 4]);
end

% rotate data
data = rotatematrix(data,1,2,rot);

% scale data
if length(prct)==1
  if prct > 0
    rng = prctile(flatten(data(:,:,:,1)),[prct 100-prct]);
  else
    rng = prctile(flatten(abs(data(:,:,:,1))),[100+prct]) * [-1 1];
  end
else
  rng = prct;
end
if rng(1)==rng(2), rng(2)= max(flatten(data(:,:,:,1))); end
data = uint8(normalizerange(data,0,255,rng(1),rng(2),1,0,1));

% do it
if isequal(mode,0)

  % make new figure window
  fig = figure;
  if isempty(cmap)
    if length(prct)==2 || prct > 0
      colormap(gray);
    else
      colormap(cmapsign);
    end
  else
    colormap(cmap);
  end

  % loop
  cnt = 1;            % the current volume to show
  ispause = 0;        % are we paused?
  forceredraw = 0;    % do we need to be forced to draw on this iteration?
  isnewkey = 1;       % has there been a gap? (i.e. no keys pressed)  if so, then process the next key press
  while 1
  
    % if we are being forced to redraw OR if we are not paused, show the image
    if forceredraw || ~ispause
      forceredraw = 0;  % reset explicitly
      im = makeimagestack(data(:,:,:,cnt),[],j);
      imagesc(im,[0 255]); axis equal tight;
      set(gca,'XTick',[],'YTick',[]);
      xlabel(sprintf('volume %d',cnt));
      title(sprintf('range is [%.1f %.1f]',rng));
      drawnow;
    end

    % wait for a key press
    polltime = GetSecs;  % the time that we started to poll for input
    while 1
      [keyIsDown,secs,keyCode,deltaSecs] = KbCheck;
      if ~keyIsDown
        isnewkey = 1;  % if there is no key pressed, then we are okay to process the next key press
      end
      if keyIsDown || secs > polltime+framedelay  % if we detect a key or time is up, get out
        break;
      end
      WaitSecs(loopdelay);  % some short delay to keep from looping too fast
    end
    
    % if we detect a key, handle it
    if keyIsDown
      
      % if the user has let go of keys since the last key press
      if isnewkey

        % update the flag
        isnewkey = 0;

        % deal with the key
        currentkey = KbName(keyCode);  % the current key pressed
        if ischar(currentkey)  % ignore the cell-vector case (i.e. multiple keys pressed)
          switch currentkey
          case 'space'
  
            % if we are done, start up again
            if cnt==size(data,4)
              cnt = 0;
              ispause = 0;
              framedelay = framedelayold;  % restore old speed

            % if we're in the middle of the movie and paused, set pause off
            elseif ispause
              ispause = 0;
              framedelay = framedelayold;  % restore old speed
            
            % if we're in the middle of the movie and not paused, set pause on
            else
              ispause = 1;
              framedelayold = framedelay;
              framedelay = minframedelay;  % temporarily go into fast mode
            end
          
          case 'LeftArrow'
            if ispause
              cnt = max(1,cnt - 1);
              forceredraw = 1;
            end
  
          case 'RightArrow'
            if ispause
              cnt = min(size(data,4),cnt + 1);
              forceredraw = 1;
            end
  
          case ',<'
            if ispause
              cnt = 1;
              forceredraw = 1;
            else
              cnt = 0;
            end
  
          case '.>'
            if ispause
              cnt = size(data,4);
              forceredraw = 1;
            else
              cnt = size(data,4)-1;
            end
  
          case '-_'
            framedelay = min(maxframedelay,framedelay * framefactor);
            WaitSecs('UntilTime',polltime+framedelay);  % make sure we wait at least this long

          case '=+'
            framedelay = max(minframedelay,framedelay / framefactor);
            WaitSecs('UntilTime',polltime+framedelay);  % make sure we wait at least this long
 
          case 'ESCAPE'
            return;
  
          end
        end

      % ok, the same key has been held down
      else

        WaitSecs('UntilTime',polltime+framedelay);  % make sure we wait at least this long

      end
      
    end

    % advance cnt if we're not paused    
    if ~ispause
      cnt = cnt + 1;
      if cnt > size(data,4)  % if we have completed a loop, we are done and so mark it as such
        cnt = size(data,4);
        ispause = 1;
        framedelayold = framedelay;
        framedelay = minframedelay;  % temporarily go into fast mode
      end
    end

  end

else

  % make the directory if necessary
  if ~isempty(stripfile(mode))
    mkdirquiet(stripfile(mode));
  end

  % write the images
  if isempty(cmap)
    cmap0 = choose(length(prct)==2 | prct(1) > 0,gray(256),cmapsign(256));
  else
    cmap0 = cmap;
  end
  for cnt=1:size(data,4)
    imwrite(makeimagestack(data(:,:,:,cnt),[],j),cmap0,sprintf([mode '.png'],offset+cnt));
  end

end
