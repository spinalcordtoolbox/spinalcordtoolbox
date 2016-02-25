% now, with the help of mapKeys (3rd example could be improved)
%lum = KbKeysAction(keyCodes, 'a', 's', lum, -50, 50);
%depth = KbKeysAction(keyCodes, 'z', 'x', depth, -50, 50);
%quitTime = KbKeysAction(keyCodes, '', 'q', 0, 0, 1);
%end

function newval = KbKeysAction(keysPressed, keyDec, keyInc, current, minV, maxV)
% Returns an incremented or decremented value, depending on keys pressed.
%
% Usage: newval = KbKeysAction(keysPressed, keyDec, keyInc, currentV, minV, maxV);
%
% keysPressed = 0/1 vector of keys pressed
% keyDec = name of key to DECREASE value
% keyInc = name of key to INCREMENT value
% current = value
% min,max = keep value within this range
%

% History:
% 17.08.2011  Written: (c) 2011-8-17 Alan Robinson, UCSD

keyNames = KbName(find(keysPressed));

% for loop expects cell so it can handle simultaneous key presses.
if ~iscell(keyNames)
  keyNames = {keyNames};
end

for key = keyNames
  switch key{1}
    case keyDec
      current = current -1;
    case keyInc
      current = current + 1;
  end
end

newval = median([minV current maxV]);

return
