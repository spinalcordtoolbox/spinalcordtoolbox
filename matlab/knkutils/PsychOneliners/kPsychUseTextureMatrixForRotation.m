function rc = kPsychUseTextureMatrixForRotation
% kPsychUseTextureMatrixForRotation
%
% Returns a constant to be passed as part of the 'specialFlags' parameter
% of the Screen('DrawTexture') and Screen('DrawTextures') command.
%
% If this flag is set, the texture drawing functions will use the OpenGL
% TEXTURE_MATRIX instead of the MODELVIEW_MATRIX for application of a
% rotation transform - ie. for rotated drawing of textures.
%
% Conceptually you can think of the difference as follows:
%
% Normally, you pass a rectangular, upright 'dstRect' destination rectangle
% that defines the area of the window that should be overdrawn by the
% texture. If the 'rotationAngle' is non-zero, this 'dstRect' will be
% rotated around its center, thereby drawing a rotated texture onto the
% screen. The pixel data is read from the upright 'srcRect' source
% rectangle in the texture image matrix.
%
% If you set this flag, then the destination rectangle 'dstRect' stays at
% its defined upright position, whereas the 'srcRect' rectangular source
% area is rotated by 'rotationAngle' around its center, so a rotated area
% of the texture image matrix is read out, but drawn upright into the
% window. The advantage of this mode is that you can draw a lot rotated
% textures tightly packed/close to other parts of your stimulus without
% interference.
%
% If you use the 2nd mode of drawing, you'll need to make sure that the
% 'srcRect' source rectangle only occupies a fraction of the texture image
% and that you'll have at least d * sqrt(2) pixels space of defined texture
% data around the center of the 'srcRect', when d is the maximum of the
% width and height of the rectangle. Otherwise the "rotated reading" from
% the texture would actually try to read from areas outside the texture
% matrix - this will create undefined visual results, but most likely not
% what you want.

% History:
% 9.10.2007 Written (MK).

rc = 1;
return;
