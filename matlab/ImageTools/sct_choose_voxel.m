function [data1d,x,y,z]=sct_choose_voxel(data4d)

data4dmean=mean(data4d,4);
hsl = uicontrol('Style','slider','Min',1,'Max',size(data4d,3),...
                'SliderStep',[1 1]./size(data4d,3),'Value',round(size(data4d,3)/2),...
                'Position',[20 20 200 20]);
set(hsl,'Callback',@(hObject) display_image(hObject,data4dmean))
imagesc(data4dmean(:,:,get(hsl,'Value'))); axis image; colormap gray;
pause;
[y,x,~]=impixel;
z=get(hsl,'Value');
data1d=squeeze(data4d(x,y,z,:));





function display_image(hObject,data4dmean)
imagesc(data4dmean(:,:,round(get(hObject,'Value')))); axis image