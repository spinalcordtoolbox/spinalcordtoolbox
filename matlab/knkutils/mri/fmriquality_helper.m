% this is a helper script used by fmriquality.m.

% <sl> is a slice number (or [])
% <fun> takes wh, ix, and sl and should return a 2D image
% <filename> should accept a figuredir string and a slice number (or [])

% make a temporary directory
tn = tempname;
mkdirquiet(tn);

% make the stack
tempimages = [];
for p=1:numtodo
  wh = firstel(find(todo(p) <= cumtotalnum));
  ix = todo(p) - sum(numinrun(1:wh-1));
  tempimages(:,:,p) = nanreplace(feval(fun,wh,ix,sl));  % just set NaNs to 0
  if p==1
    tempimages = placematrix2(zeros(size(tempimages,1),size(tempimages,2),numtodo),tempimages);
  end
end
  
% make the figure
parfor p=1:numtodo
    prev = warning('query'); warning('off');  % avoid "all pixels are constant" warning
  statusdots(p,numtodo);
  temp = tempimages(:,:,p);
  heightpixels = 700;
  widthpixels = round(heightpixels*.95*size(temp,2)/size(temp,1));  % .95 to ensure some room for title
  figureprep([100 100 widthpixels heightpixels]); hold on;
  [c,h] = contour(temp,[5000 5000],'k-');
  set(gca,'YDir','reverse');
  pxsz = size(temp,2)*volsize(2)/widthpixels;  % assume the width is limiting the size
  title(sprintf('figure pixel size is %.2f mm x %.2f mm',pxsz,pxsz));
  setaxispos(gca,[0 0 1 1]);
  axis off; axis equal tight; 
  axis([.5 size(temp,2)+.5 .5 size(temp,1)+.5]);
  figurewrite('contours%04d',p,[],tn);
    warning(prev);
end
              %   TOO SLOW set(figs(cnt),'DefaultPatchEdgeAlpha',0.2);

% average the images
im = 0;
for p=1:numtodo
  im = im + double(imread(sprintf([tn '/contours%04d.png'],p)))/255;
end
im = im / numtodo;  % range is 0 to 1
imwrite(uint8(255*im(:,:,1)),flipud(hot(256)),sprintf(filename,figuredir,sl));




%   if isoneslice
%     set(straightline(.5:3*1/volsize(2):size(temp,2)-1+.5,'v','k--'),'Color',[.7 .7 .7]);
%     set(straightline(.5:3*1/volsize(1):size(temp,1)-1+.5,'h','k--'),'Color',[.7 .7 .7]);
%   end
