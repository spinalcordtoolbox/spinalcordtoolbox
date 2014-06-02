% =========================================================================
% FUNCTION                                 
% j_displayMRI.m   
%
% INPUT
% D                 3d matrix
% clims				[min max]. Put [] for auto adjustment.
% (opt)             structure
%   rotate_angle        integer. 0,90,180,270 in trigo sense (default=90)
%   figure_name         string.
%   figure_title        string.
%   h_fig               integer. figure handle.
%
% OUTPUT
% h_axes            integer. axes handle
%
% COMMENTS
% emprunté à V. Perlbarg
% Julien Cohen-Adad, 2007-10-23
% =========================================================================
function h_axes = j_displayMRI(D,clims,opt)


% Default initialization
rotate_angle	= 0;
map				= 'gray';
figure_name     = 'Display MRI volume';
figure_title    = '';
h_fig           = 0;

% User initializations
if (nargin<1), help j_displayMRI; end
if (nargin<3), opt = []; end
if (isfield(opt,'map')), map = opt.map; end;
if (isfield(opt,'h_fig')), h_fig = opt.h_fig; end;


% rotate images
[nx ny nz] = size(D);
switch rotate_angle
    case(0)
        Drot = D;
    case(90)
        Drot = zeros(ny,nx,nz);
        for iz=1:nz
            Drot(:,:,iz) = rot90(D(:,:,iz));
        end
    case(270)
        Drot = permute(zeros(ny,nx,nz));
        for iz=1:nz
            Drot(:,:,iz) = D(:,:,iz)';
        end
end
clear D;

if ~h_fig
    h_fig = figure('name',figure_name,'color','white');
end
    
matrix_in=squeeze(Drot);
if length(size(Drot))==4 & size(Drot,4)>1
    fprintf('Cannot display more than one volume at once! Please use 3-D array as first argument to emumontage!\n');
    return
end
[siD1,siD2,siD3]=size(Drot);
%si=almquad(size(D,3));
si(1) = ceil(sqrt(siD3));
si(2) = ceil(siD3/si(1));

r=1;

M=zeros(si(1)*size(Drot,1),si(2)*size(Drot,2));
for i=1:si(1)
    for j=1:si(2)
        if r<=siD3
            M((i-1)*siD1+[1:siD1],(j-1)*siD2+[1:siD2])=Drot(:,:,r);
            r=r+1;
        end
    end
end


figure(h_fig)
if exist('clims')
	h_axes = imagesc(M,clims);
else
 	h_axes = imagesc(M);
end
title(figure_title)
axis image
colormap(map)
colorbar
set(gca,'Xtick',[])
set(gca,'Ytick',[])
 %noticks

 
 
 
 
 
 
 % OLD CODE
% =====================
% nb_slices = size(D,3);
% if (nb_slices>4)
%     nb_rows=3;
%     nb_columns=3;
% elseif (nb_slices>2)
%     nb_rows=2;
%     nb_columns=2;
% end
% 
% % display figure
% figure('name',figure_name)
% % colormap('jet');
% for j=1:nb_rows*nb_columns
%     if j>nb_slices, break; end;
%     subplot(nb_rows,nb_columns,j)
%     imagesc(imrotate(D(:,:,j),rotate_angle));
%     axis('image','off');
%     title(['Slice number = ',num2str(j)],'HorizontalAlignment','center');
% end


