% =========================================================================
% FUNCTION
% j_dmri_gradientsDisplay.m
%
% INPUT
% gradient_vectors			nx3
%
% OUTPUT
% (-)
% 
% COMMENTS
% Julien Cohen-Adad 2009-10-02
% =========================================================================
function j_dmri_gradientsDisplay(gradient_vectors)


% get gradient file
% opt.file_selection = 'matlab';
% opt.windows_title = 'Select gradient file';
% opt.ext_filter = '*.txt';
% opt.output = 'array';
% fname = j_getfiles(opt);

% % open file
% fid_r = fopen(fname,'r');
% nb_directions = fscanf(fid_r,'%g',[1])-1;
% gradient_list = fscanf(fid_r,'%g %g %g',[3 inf]);
% fclose(fid_r);
% gradient_list = gradient_list';

% display gradients
figure('color','white'),
for i=1:size(gradient_vectors,1)
    plot3(gradient_vectors(i,1),gradient_vectors(i,2),gradient_vectors(i,3),'k.','MarkerSize',10)
    hold on
end
axis vis3d;
view(3), axis equal
axis on, grid
rotate3d on;





% % Voronoi transformation
X=gradient_vectors;
figure('name','Voronoi')
[V,C] = voronoin(X);
K = convhulln(X);
d = [1 2 3 1];       % Index into K
for i = 1:size(K,1)
   j = K(i,d);
   h(i)=patch(X(j,1),X(j,2),X(j,3),i,'FaceColor','white','FaceLighting','phong','EdgeColor','black');
end
hold off
view(2)
axis off
axis equal
colormap(gray);
% title('One cell of a Voronoi diagram')
axis vis3d;
rotate3d on;
% 
% 
% nb_directions = size(gradient_vectors,1);
% % Delaunay transformation
% X=gradient_vectors(2:end,:);
% figure('name',[num2str(nb_directions),' directions'])
% T = delaunayn(X);      % Generate Delaunay tessellation.
% d = [1 1 1 2; 2 2 3 3; 3 4 4 4];  % Index into T
% for i = 1:size(T,1)      % Draw each tetrahedron.
%    y = T(i,d);           % Get the ith T to make a patch.
%    x1 = reshape(X(y,1),3,4);
%    x2 = reshape(X(y,2),3,4);
%    x3 = reshape(X(y,3),3,4);
%    h(i)=patch(x1,x2,x3,(1:4)*i,'FaceColor','white','FaceLighting','none','SpecularColorReflectance',0.5,'EdgeColor','black');
% %    reducepatch(h(i),0.01)
% 
% %    h(i)=patch(x1,x2,x3,(1:4)*i,'FaceAlpha',0.9,'FaceLighting','phong','EdgeColor','black');
% end
% 
% hold off
% % view(3), 
% axis equal
% axis off
% % camorbit(65,120)         % To view it from another angle
% % title([num2str(nb_directions),' directions'])
% colormap(gray);
% axis vis3d;
% rotate3d on;




% % check normalization
% for i=2:nb_directions
%     gradient_norm(i) = norm(gradient_list(i,:));
% end
% 
% disp(gradient_norm)
% 
% for i=2:nb_directions
%     gradient_list_norm(i,:)=gradient_list(i,:)./gradient_norm(i);
% end
% for i=1:nb_directions
%     gradient_norm2(i,:) = norm(gradient_list_norm(i,:));
% end


% % generate triangulated gradients
% tri_gradients=voronoin(gradients(:,1),gradients(:,2),gradients(:,3));
% 
% % display mesh
% figure
% tri=trimesh(tri_gradients,gradients(:,1),gradients(:,2),gradients(:,3),'FaceColor','interp','FaceLighting','phong','EdgeColor','black');
% rotate3d on;
% axis equal
% colormap(gray);
% axis vis3d;
% % set(tri,'FaceLighting','phong','FaceColor','interp','AmbientStrength',0.98);
% % light('Position',[1 4 0],'Style','infinite');