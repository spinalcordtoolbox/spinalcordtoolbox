function viewimage(im)

% function viewimage(im)
% 
% <im> is a 2D image
%
% in the current figure window, show the image and its
% central rows and columns (round((end+1)/2)), as well 
% as its amplitude spectrum and its central rows and columns.

% HRM: hanning window?

% do fft
im2 = fftshift(abs(fft2(im)));

% do it
setfigurepos([50 50 800 500]);
subplot(2,3,1); imagesc(im); colorbar; axis equal tight; title('image');
subplot(2,3,2); plot(im(round((end+1)/2),:),'r.-'); ax = axis; axis([1 size(im,2) ax(3:4)]); title('central row of image');
subplot(2,3,3); plot(im(:,round((end+1)/2)),'r.-'); ax = axis; axis([1 size(im,1) ax(3:4)]); title('central column of image');
subplot(2,3,4); imagesc(im2); colorbar; axis equal tight; title('magnitude spectrum');
subplot(2,3,5); plot(im2(round((end+1)/2),:),'r.-'); ax = axis; axis([1 size(im,2) ax(3:4)]); title('central row of magnitude spectrum');
subplot(2,3,6); plot(im2(:,round((end+1)/2)),'r.-'); ax = axis; axis([1 size(im,1) ax(3:4)]); title('central column of magnitude spectrum');
drawnow;
