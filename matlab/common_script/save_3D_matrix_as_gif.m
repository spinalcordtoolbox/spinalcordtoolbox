% save_3D_matrix_as_gif
% ---------------------
% Function that saves a 3D matrix in a gif image.
% 
% input:
%     filename  = the desired name of the output file 
%                 e.g.: 'C:\Users\John\Desktop\animation.gif' 
%                       or 'animation.gif' (saves in current folder)
%     matrix    = the 3D matrix, matrix(:,:,n) contains the n'th frame of the 
%                 desired gif animation
%     delaytime = optional argument to specify the delay time (in seconds)
%                 of the gif file (default = 0.1 sec)
% output:
%     /
%     
% example:
%     Im = zeros(100,100,20);
%     % create circles with decreasing radii
%     for ii=1:20
%         Im(:,:,ii) = phantom([1 1/ii 1/ii 0 0 0],100);
%     end
%     save_3D_matrix_as_gif('C:\Users\John\Desktop\name_of_gif_file.gif',Im,0.2)
%
% author:  Geert Van Eyndhoven
% contact: vegeert[at]hotmail[dot]com

function save_3D_matrix_as_gif(filename, matrix, delaytime)

    if(nargin<2 || nargin>3)
        error('incorrect number of input arguments')
    end
    
    if nargin==2
        delaytime = 0.1;
    end
    
    % adjust matrix to have entries between 1 and 256
    % first make range between 0 and 1
%     matrix = matrix - min(min(min(matrix)));
%     matrix = matrix/(max(max(max(matrix))));
%     % adjust range to be between 1 and 256
%     matrix = matrix*255 + 1;
    imwrite(squeeze(matrix(:,:,1)),filename,'gif', 'WriteMode','overwrite','DelayTime',delaytime,'LoopCount',Inf);
    for ii = 2:size(matrix,3)
        imwrite(squeeze(matrix(:,:,ii)),filename,'gif', 'WriteMode','append','DelayTime',delaytime);
    end

    [~,name,ext] = fileparts(filename);
    imwrite(squeeze(matrix(:,:,1)),hot(90),[name '_hot' ext],'gif', 'WriteMode','overwrite','DelayTime',delaytime,'LoopCount',Inf);
    for ii = 2:size(matrix,3)
        imwrite(squeeze(matrix(:,:,ii)),hot(90),[name '_hot' ext],'gif', 'WriteMode','append','DelayTime',delaytime);
    end

    imwrite(squeeze(matrix(:,:,1)),cool(90),[name '_cool' ext],'gif', 'WriteMode','overwrite','DelayTime',delaytime,'LoopCount',Inf);
    for ii = 2:size(matrix,3)
        imwrite(squeeze(matrix(:,:,ii)),cool(90),[name '_cool' ext],'gif', 'WriteMode','append','DelayTime',delaytime);
    end

    imwrite(squeeze(matrix(:,:,1)),jet(90),[name '_jet' ext],'gif', 'WriteMode','overwrite','DelayTime',delaytime,'LoopCount',Inf);
    for ii = 2:size(matrix,3)
        imwrite(squeeze(matrix(:,:,ii)),jet(90),[name '_jet' ext],'gif', 'WriteMode','append','DelayTime',delaytime);
    end
end