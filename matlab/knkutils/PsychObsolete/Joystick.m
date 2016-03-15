function result = Joystick(arg1, arg2, arg3)


if nargin>3
    error('Too many arguments');
elseif nargin==3
    tempResult=Gamepad(arg1,arg2,arg3);
elseif nargin==2
    tempResult=Gamepad(arg1,arg2);
elseif nargin==1
    tempResult=Gamepad(arg1);
elseif nargin==0
    Gamepad;
    tempResult=nan;
end

if ~isnan(tempResult)
    result=tempResult; 
end

% warning('"Joystick has been renamed to "Gamepad".  To avoid this warning message use the new name,  "Gamepad".');

