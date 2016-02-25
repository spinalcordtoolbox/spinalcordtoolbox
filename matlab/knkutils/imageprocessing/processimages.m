function processimages(files,fun)

% function processimages(files,fun)
%
% <files> is a pattern that matches image files (see matchfiles.m)
% <fun> is a function to apply to each image.  the function should
%   return an image.  a special case is that the function can return
%   a cell matrix of images; in this case, we write out to files
%   named like FILE_01.ext, FILE_02.ext, etc.  if any element of the
%   cell matrix is empty, we just skip that one.
%
% apply <fun> to each image matched by <files> and save it.
% we automatically convert each image to double before calling
% <fun> and we automatically convert back to uint8 before saving.
%
% we issue a warning if no files are named by <files>.
%
% example:
% imwrite(uint8(255*rand(100,100)),'test.png');
% figure; subplot(1,2,1); imagesc(imread('test.png'),[0 255]);
% processimages('test.png',@(x) x/2);
% subplot(1,2,2); imagesc(imread('test.png'),[0 255]);

% transform
files = matchfiles(files);

% check sanity
if length(files)==0
  warning('no file matches');
  return;
end

% do it
for p=1:length(files)
  temp = feval(fun,double(imread(files{p})));
  if iscell(temp)
    [a,b,c,d] = fileparts(files{p});
    if ~isempty(a)
      a = [a '/'];
    end
    cnt = 0;
    for q=1:numel(temp)
      if ~isempty(temp{q})
        cnt = cnt + 1;
        imwrite(uint8(temp{q}),[a b sprintf('_%02d',cnt) c]);
      end
    end
  else
    imwrite(uint8(temp),files{p});
  end
end
