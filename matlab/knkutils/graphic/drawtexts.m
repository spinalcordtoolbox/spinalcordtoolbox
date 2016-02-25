function f = drawtexts(res,x,y,font,sz,color,bg,word)

% function f = drawtexts(res,x,y,font,sz,color,bg,word)
%
% arguments are the same as to drawtext.m except for:
%   <res> is the number of pixels along one side
%   <word> can be a cell vector with different words
% for <x> and <y>, assume the standard coordinate frame.
%
% return a series of 2D images where values are in [0,1].
% the dimensions of the returned matrix are res x res x images
% note that we explicitly convert to grayscale.
%
% example:
% figure; imagesc(makeimagestack(drawtexts(100,0,0,'Helvetica',.5,[0 .5 0],[0 0 0],{'A' 'B' 'C'}))); axis equal tight;

% NOTE: see also drawclosedcontours.m.

% input
if ~iscell(word)
  word = {word};
end
numwords = length(word);

% do it
fig = figure;
f = zeros(res,res,numwords);
for p=1:numwords
  clf; drawtext(0,x,y,font,sz,color,bg,word{p});
  f(:,:,p) = renderfigure(res,1);
end  
close(fig);
