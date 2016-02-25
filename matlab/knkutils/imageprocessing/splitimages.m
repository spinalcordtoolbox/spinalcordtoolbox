function splitimages(files,dsize)

% function splitimages(files,dsize)
%
% <files> is a pattern that matches image files (see matchfiles.m)
% <dsize> is the desired size of image chunks, i.e. [ROWS COLS]
% 
% split the images matched by <files> into image chunks and
% write these chunks out like FILE_01.ext, FILE_02.ext, etc.
%
% example:
% imwrite(uint8(255*rand(100,100)),'test.tif');
% splitimages('test.tif',[50 50]);

% do it
processimages(files,@(x) mat2cell(x,[repmat(dsize(1),[1 floor(size(x,1)/dsize(1))]) size(x,1)-dsize(1)*floor(size(x,1)/dsize(1))], ...
                                    [repmat(dsize(2),[1 floor(size(x,2)/dsize(2))]) size(x,2)-dsize(2)*floor(size(x,2)/dsize(2))],size(x,3)));
