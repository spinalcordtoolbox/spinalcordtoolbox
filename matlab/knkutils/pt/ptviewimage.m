function offset = ptviewimage(files,imageflip,mode)

% function offset = ptviewimage(files,imageflip,mode)
% 
% <files> is a pattern that matches one or more image files (see matchfiles.m).
%   alternatively, can be the images themselves as a matrix.  if this matrix 
%     has more than one element along the fourth dimension, we treat the matrix
%     as a series of color images.  if not, we treat the matrix as a series 
%     of grayscale images.  we convert values to uint8 automatically.
%   alternatively, can be a scalar (we convert to uint8 automatically), 
%     in which case we fill the whole window with this scalar.
% <imageflip> (optional) is [J K] where J is whether to flip first
%   dimension and K is whether to flip second dimension.
%   default: [0 0].
% <mode> (optional) is
%   0 means normal operation
%   1 means automatically cycle through images (1 per second).  in this mode,
%     the only possible key is escape, and you must hold it down until it
%     is detected.
%   default: 0.
% 
% show the images named by <files>.  use the arrow keys (increments of 1 pixel),
% ijkl (increments of 10 pixels), and wasd (increments of 30 pixels) to 
% position the images.  use spacebar or r to cycle
% through the different images.  use escape to exit.
%
% upon exiting, we return <offset> as [X Y] which are integers
% representing the final position of the images (0 means centered).
%
% example:
% pton;
% offset = ptviewimage('/research/figures/calibrationimages/*.png');
% ptoff;

% input
if ~exist('imageflip','var') || isempty(imageflip)
  imageflip = [0 0];
end
if ~exist('mode','var') || isempty(mode)
  mode = 0;
end

% get information about the PT setup
win = firstel(Screen('Windows'));
rect = Screen('Rect',win);

% prepare the images
if iscell(files) || ischar(files)

  % get the filenames ready
  files = matchfiles(files);
  assert(length(files) >= 1,'<files> did not match any files');

  % read images
  images = {};
  for p=1:length(files)
    images{p} = imread(files{p});
  end

else

  % scalar case
  if numel(files) == 1
    images = {repmat(uint8(files),[rect(4)-rect(2) rect(3)-rect(1)])};

  % images case
  else
    images = {};
    if size(files,4) > 1
      for p=1:size(files,4)
        images{p} = uint8(files(:,:,:,p));
      end
    else
      for p=1:size(files,3)
        images{p} = uint8(files(:,:,p));
      end
    end
  end
  
end

% figure out rects of the images
imagerects = {};
for p=1:length(images)
  imagerects{p} = CenterRect([0 0 size(images{p},2) size(images{p},1)],rect);
end

% flip
if imageflip(1)
  images = cellfun(@(x) flipdim(x,1),images,'UniformOutput',0);
end
if imageflip(2)
  images = cellfun(@(x) flipdim(x,2),images,'UniformOutput',0);
end

% initialize
imnum = 1;        % start with the first image
dx = 0; dy = 0;   % initial offsets are 0
reportimage = 1;  % should we report which image it is?

% do it
while 1
  
  % make a texture, draw it at a particular position, and issue the flip command
  texture = Screen('MakeTexture',win,images{imnum});
  Screen('DrawTexture',win,texture,[],imagerects{imnum} + [dx dy dx dy],[],0);
  Screen('Close',texture);
  [VBLTimestamp,StimulusOnsetTime,FlipTimestamp,Missed,Beampos] = Screen('Flip',win);
  
  % report to stdout
  if reportimage
    if iscell(files)
      fprintf('now showing image number %d (%s).\n',imnum,files{imnum});
    else
      fprintf('now showing image number %d.\n',imnum);
    end
    reportimage = 0;
  end
  
  % this is the normal mode
  if mode==0

    % wait for a key press from any and all input devices
    [secs,keyCode,deltaSecs] = KbWait(-3,2);
    
    % handle the key press
    switch KbName(keyCode)
    case 'UpArrow'
      dy = dy - 1;
    case 'DownArrow'
      dy = dy + 1;
    case 'LeftArrow'
      dx = dx - 1;
    case 'RightArrow'
      dx = dx + 1;
    case 'i'
      dy = dy - 10;
    case 'k'
      dy = dy + 10;
    case 'j'
      dx = dx - 10;
    case 'l'
      dx = dx + 10;
    case 'w'
      dy = dy - 30;
    case 's'
      dy = dy + 30;
    case 'a'
      dx = dx - 30;
    case 'd'
      dx = dx + 30;
    case {'space' 'r'}
      imnum = mod2(imnum+1,length(images));
      reportimage = 1;
    case 'ESCAPE'
      break;
    end
  
  % this is the automatic cycle mode
  else

    % check for a key press from any and all input devices
    [keyIsDown,secs,keyCode,deltaSecs] = KbCheck(-3);
    
    % handle the key press
    if keyIsDown
      switch KbName(keyCode)
      case 'ESCAPE'
        break;
      end
    end

    % cycle to next image and wait one second
    imnum = mod2(imnum+1,length(images));
    reportimage = 1;
    WaitSecs(3);
    
  end

end

% output
offset = [dx dy];





% THIS IS THAT XYZ IMPELEMENTATION THING.
%
% function [offset,xyz] = ptviewimage(files,imageflip,mode)
% 
% % function [offset,xyz] = ptviewimage(files,imageflip,mode)
% % 
% % <files> is a pattern that matches one or more image files (see matchfiles.m).
% %   alternatively, can be the images themselves as a matrix.  if this matrix 
% %     has more than one element along the fourth dimension, we treat the matrix
% %     as a series of color images.  if not, we treat the matrix as a series 
% %     of grayscale images.  we convert values to uint8 automatically.
% %   alternatively, can be a scalar (we convert to uint8 automatically), 
% %     in which case we fill the whole window with this scalar.
% % <imageflip> (optional) is [J K] where J is whether to flip first
% %   dimension and K is whether to flip second dimension.
% %   default: [0 0].
% % <mode> (optional) is
% %   0 means normal operation
% %   1 means automatically cycle through images (1 per second).  in this mode,
% %     the only possible key is escape, and you must hold it down until it
% %     is detected.
% %   2 means
% %   default: 0.
% % 
% % show the images named by <files>.  use the arrow keys (increments of 1 pixel),
% % ijkl (increments of 10 pixels), and wasd (increments of 30 pixels) to 
% % position the images.  use spacebar or r to cycle
% % through the different images.  use escape to exit.
% %
% % upon exiting, we return <offset> as [X Y] which are integers
% % representing the final position of the images (0 means centered).
% %
% % example:
% % pton;
% % offset = ptviewimage('/research/figures/calibrationimages/*.png');
% % ptoff;
% 
% % input
% if ~exist('imageflip','var') || isempty(imageflip)
%   imageflip = [0 0];
% end
% if ~exist('mode','var') || isempty(mode)
%   mode = 0;
% end
% 
% % init
% xyz = [];
% 
% % get information about the PT setup
% win = firstel(Screen('Windows'));
% rect = Screen('Rect',win);
% 
% % prepare the images
% if iscell(files) || ischar(files)
% 
%   % get the filenames ready
%   files = matchfiles(files);
%   assert(length(files) >= 1,'<files> did not match any files');
% 
%   % read images
%   images = {};
%   for p=1:length(files)
%     images{p} = imread(files{p});
%   end
% 
% else
% 
%   % scalar case
%   if numel(files) == 1
%     images = {repmat(uint8(files),[rect(4)-rect(2) rect(3)-rect(1)])};
% 
%   % images case
%   else
%     images = {};
%     if size(files,4) > 1
%       for p=1:size(files,4)
%         images{p} = uint8(files(:,:,:,p));
%       end
%     else
%       for p=1:size(files,3)
%         images{p} = uint8(files(:,:,p));
%       end
%     end
%   end
%   
% end
% 
% % figure out rects of the images
% imagerects = {};
% for p=1:length(images)
%   imagerects{p} = CenterRect([0 0 size(images{p},2) size(images{p},1)],rect);
% end
% 
% % flip
% if imageflip(1)
%   images = cellfun(@(x) flipdim(x,1),images,'UniformOutput',0);
% end
% if imageflip(2)
%   images = cellfun(@(x) flipdim(x,2),images,'UniformOutput',0);
% end
% 
% % initialize
% imnum = 1;        % start with the first image
% dx = 0; dy = 0;   % initial offsets are 0
% reportimage = 1;  % should we report which image it is?
% 
% % do it
% while 1
%   
%   % make a texture, draw it at a particular position, and issue the flip command
%   texture = Screen('MakeTexture',win,images{imnum});
%   Screen('DrawTexture',win,texture,[],imagerects{imnum} + [dx dy dx dy],[],0);
%   Screen('Close',texture);
%   [VBLTimestamp,StimulusOnsetTime,FlipTimestamp,Missed,Beampos] = Screen('Flip',win);
%   
%   % report to stdout
%   if reportimage
%     if iscell(files)
%       fprintf('now showing image number %d (%s).\n',imnum,files{imnum});
%     else
%       fprintf('now showing image number %d.\n',imnum);
%     end
%     reportimage = 0;
%   end
%   
%   % this is the normal mode
%   switch mode
%   case 0
% 
%     % wait for a key press from any and all input devices
%     [secs,keyCode,deltaSecs] = KbWait(-3,2);
%     
%     % handle the key press
%     switch KbName(keyCode)
%     case 'UpArrow'
%       dy = dy - 1;
%     case 'DownArrow'
%       dy = dy + 1;
%     case 'LeftArrow'
%       dx = dx - 1;
%     case 'RightArrow'
%       dx = dx + 1;
%     case 'i'
%       dy = dy - 10;
%     case 'k'
%       dy = dy + 10;
%     case 'j'
%       dx = dx - 10;
%     case 'l'
%       dx = dx + 10;
%     case 'w'
%       dy = dy - 30;
%     case 's'
%       dy = dy + 30;
%     case 'a'
%       dx = dx - 30;
%     case 'd'
%       dx = dx + 30;
%     case {'space' 'r'}
%       imnum = mod2(imnum+1,length(images));
%       reportimage = 1;
%     case 'ESCAPE'
%       break;
%     end
%   
%   % this is the automatic cycle mode
%   case {1 2}
% 
%     % check for a key press from any and all input devices
%     [keyIsDown,secs,keyCode,deltaSecs] = KbCheck(-3);
%     
%     % handle the key press
%     if keyIsDown
%       switch KbName(keyCode)
%       case 'ESCAPE'
%         break;
%       end
%     end
% 
%     % just wait? or should we take measurement?
%     if mode==1
%       WaitSecs(3);
%     else
%       xyz(:,end+1) = PR650measxyz;
%       if imnum==length(images)
%         break;
%       end
%     end
% 
%     % cycle to next image
%     imnum = mod2(imnum+1,length(images));
%     reportimage = 1;
%   
%   end
% 
% end
% 
% % output
% offset = [dx dy];
