function rc = kPsychDontDoRotation
% kPsychDontDoRotation
%
% Returns a constant to be passed as part of the 'specialFlags' parameter
% of the Screen('DrawTexture') and Screen('DrawTextures') command.
%
% If this flag is set, the texture drawing functions will not apply texture
% Rotation by themselves, even if the 'rotationAngle' parameter is
% non-zero. Instead they will just pass the 'rotationAngle' parameter to a
% GLSL shader if any is enabled, so the shader can implement its own way of
% "rotating" the texture to be drawn. This allows for custom
% implementations of texture drawing.
%

% History:
% 28.10.2007 Written (MK).

rc = 2;
return;
