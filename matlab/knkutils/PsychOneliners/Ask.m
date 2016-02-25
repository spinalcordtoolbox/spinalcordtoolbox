function reply=Ask(window,message,textColor,bgColor,replyFun,rectAlign1,rectAlign2,fontsize)
% reply = Ask(window,message,[textColor],[bgColor],[replyFun],[rectAlign1],[rectAlign2],[fontSize=30])
%
% Draw the message, using textColor, right-justified in the upper right of
% the window, call reply=eval(replyFun), then erase (by drawing text again
% using bgColor) and return. The default "replyFun" is 'GetClicks'. You may
% want to use 'GetChar' or 'GetString'.
% 
% "rectAlign1" and "rectAlign2", if present, are applied after the above
% alignment. The values should be selected from: RectLeft, RectRight,
% RectTop, RectBottom. Alternatively you can pass in one of the strings
% 'left','top','right','bottom','center'.
%
% If you want to align the text to a different rectangular box than the
% window, then pass the rectangle defining that box as argument
% 'rectAlign1', and the wanted alignment as argument 'rectAlign2', e.g.,
% if rectAlign1 = [400 300 600 400] and rectAlign2 = 'center', then the
% text 'message' would get centered in the box with left-top corner
% (400,300) and right-bottom corner (600,400).
%
% "fontSize" is the font size you want text typed in; will restore old
% fontsize before returning.
%
% Typical uses:
% reply=Ask(window,'Click when ready.'); % Wait for (multiple) mouse clicks, return number of clicks.
% reply=Ask(window,'What''s your name?',[],[],'GetString'); % Accept keyboard input, but don't show it.
% reply=Ask(window,'Who are you?',[],[],'GetChar',RectLeft,RectTop); % Accept keyboard input, echo it to screen.
%
% See also GetString.

% 3/9/97  dgp	Wrote it, based on dhb's WaitForClick.m
% 3/19/00  dgp	Suggest turning off font smoothing. Default colors.
% 8/14/04  dgp	As suggested by Paul Thiem, added an example (and better argument checking) 
%               to make it clear that replyFun must be supplied as a string and rectAlign1 as a value.
% 8/14/04  dgp	Call Screen 'WindowToFront'.
% 1/19/07  asg  Modified to work in OSX (for asg's purposes).
% 6/6/07   mk   remove Screen('WindowToFron') unsupported on PTB-3, other
%               small fixes...

dontClear = 1;

if nargin < 2
    error('Ask: Must provide at least the first two arguments.');
end

if ~Screen(window, 'WindowKind')
	error('Invalid window handle provided.')
end

if nargin > 7 && ~isempty(fontsize)
    oldFontSize=Screen('TextSize', window, fontsize);
else
    oldFontSize=Screen('TextSize', window, 30);
end;

if nargin>4
    if isempty(replyFun)
        replyFun='GetClicks';
    end

    if isa(replyFun,'double')
		error('Ask: replyFun must be [] or a string, e.g. ''GetClicks''.');
    end
else
    replyFun='GetClicks';
end

% Create the box to hold the text that will be drawn on the screen.
screenRect = Screen('Rect', window);
if ~isempty(message)
    tbx = Screen('TextBounds', window, message);
    width = tbx(3);
    height = tbx(4);
else
    width = 0;
    height = 0;
    message = '';
end

if nargin>5 && ~isempty(rectAlign1) && (length(rectAlign1) == 4) && isnumeric(rectAlign1)
    % rectAlign1 overrides reference box screenRect:
    screenRect = rectAlign1;
end

r=[0 0 width height + Screen('TextSize', window)];
if strcmp(replyFun,'GetChar')
    % In 'GetChar' mode we default to left-alignment of message:
    r=AlignRect(r,screenRect,RectLeft,RectTop); % asg changed to align on Left side of screen
else
    % For other replyFun's we default to good ol' right-alignment:
    r=AlignRect(r,screenRect,RectRight,RectTop);
end

if nargin>6 && ~isempty(rectAlign2)
	r=AlignRect(r,screenRect,rectAlign2);
end

if nargin>5  && ~isempty(rectAlign1) && ((length(rectAlign1) ~= 4) || ischar(rectAlign1))
	r=AlignRect(r,screenRect,rectAlign1);
end

if nargin<4 || isempty(bgColor)
	bgColor=WhiteIndex(window);
end

if nargin<3 || isempty(textColor)
	textColor=BlackIndex(window);
end

%Screen(window,'WindowToFront');       % asg commented out

[oldX, oldY]=Screen(window,'DrawText',message,r(RectLeft),r(RectBottom),textColor);
Screen('Flip', window, 0, dontClear);      % asg added

if strcmp(replyFun,'GetChar')
    FlushEvents('keyDown');
    i=1;
    reply(i)=GetChar(0,1);  % get the 1st typed character (with no timing info. and ascii codes only)
    [newX(i), newY(i)]=Screen(window,'DrawText',char(reply(i)),oldX,oldY,textColor); % put coverted ascii code letter on screen
    Screen('Flip', window, 0, dontClear);   % flip it to the screen
    i=2;
    reply(i)=GetChar(0,1);  % get the 2nd typed character (with no timing info. and ascii codes only)

    while reply(i)==8  % backspace/delete was typed
        i=1;
        Screen('FillRect', window, bgColor);
        [oldX, oldY]=Screen(window,'DrawText',message,r(RectLeft),r(RectBottom),textColor); % redraw text with no response letters
        Screen('Flip', window, 0, dontClear);   % flip it to the screen
        reply(i)=GetChar(0,1);  % get the next typed character (with no timing info. and ascii codes only)
        if reply(i)~=8
            [newX(i), newY(i)]=Screen(window,'DrawText',char(reply(i)),oldX,oldY,textColor);
            Screen('Flip', window, 0, dontClear);
            i=2;
            reply(i)=GetChar(0,1);
        end;
    end;

    while ~eq(reply(i),10)  && ~eq(reply(i),13) % until they hit RETURN
        [newX(i), newY(i)]=Screen(window,'DrawText',char(reply(i)),newX(i-1),newY(i-1),textColor); % put coverted ascii code letter on screen
        Screen('Flip', window, 0, dontClear);   % flip it to the screen
        i=i+1;
        reply(i)=GetChar(0,1);  % get the next character (with no timing info. and ascii codes only)
        while reply(i)==8  % backspace/delete was typed
            i=i-1;
            if i<2  % can't backspace too far!
                i=1;
                Screen('FillRect', window, bgColor);
                [oldX, oldY]=Screen(window,'DrawText',message,r(RectLeft),r(RectBottom),textColor);% redraw text with no response letters
                Screen('Flip', window, 0, dontClear);   % flip it to the screen
                reply(i)=GetChar(0,1);
                if reply(i)~=8
                    [newX(i), newY(i)]=Screen(window,'DrawText',char(reply(i)),oldX,oldY,textColor);
                    Screen('Flip', window, 0, dontClear);
                    i=2;
                    reply(i)=GetChar(0,1);
                end;
            elseif i>1
                Screen('FillRect', window, bgColor);
                [oldX, oldY]=Screen(window,'DrawText',message,r(RectLeft),r(RectBottom),textColor);
                [newX(i-1), newY(i-1)]=Screen(window,'DrawText',char(reply(1:i-1)), oldX, oldY, textColor); % put old letters on screen
                Screen('Flip', window, 0, dontClear);   % flip it to the screen
                reply(i)=GetChar(0,1);  % get the next character (with no timing info. and ascii codes only)
            end;
        end;
    end;

    Screen('FillRect', window, bgColor);
    Screen('Flip', window);

    for d=min(find(reply==8 | reply==10 | reply==13))-1 %#ok<MXFND>
        reply = reply(1:d);
    end;
    
    % Convert to char() string:
    reply=char(reply);
else
    reply=eval(replyFun);
end;

% Restore text size:
Screen('TextSize', window ,oldFontSize);

return;
