function gradientsDisplay(gradient_vectors,display_voronoi)


display_3d = 1;


% display gradients
if display_3d
	h_fig = figure('color','white');
	for i=1:size(gradient_vectors,1)
		plot3(gradient_vectors(i,1),gradient_vectors(i,2),gradient_vectors(i,3),'k.','MarkerSize',10)
		hold on
	end
	xlabel('X')
	ylabel('Y')
	zlabel('Z')
	axis vis3d;
	view(3), axis equal
	axis on, grid
	rotate3d on;
end


% display gradients
h_fig = figure('color','white');
subplot(2,2,1)
for i=1:size(gradient_vectors,1)
    plot3(gradient_vectors(i,1),gradient_vectors(i,2),gradient_vectors(i,3),'k.','MarkerSize',10)
    hold on
end
xlabel('X')
ylabel('Y')
zlabel('Z')
axis vis3d;
view(3), axis equal
axis on, grid
rotate3d on;
view(0,0)

subplot(2,2,2)
for i=1:size(gradient_vectors,1)
    plot3(gradient_vectors(i,1),gradient_vectors(i,2),gradient_vectors(i,3),'k.','MarkerSize',10)
    hold on
end
xlabel('X')
ylabel('Y')
zlabel('Z')
axis vis3d;
view(3), axis equal
axis on, grid
rotate3d on;
view(90,0)

subplot(2,2,3)
for i=1:size(gradient_vectors,1)
    plot3(gradient_vectors(i,1),gradient_vectors(i,2),gradient_vectors(i,3),'k.','MarkerSize',10)
    hold on
end
xlabel('X')
ylabel('Y')
zlabel('Z')
axis vis3d;
view(3), axis equal
axis on, grid
rotate3d on;
view(0,90)


% % Voronoi transformation
if display_voronoi
	X=gradient_vectors;
	h_fig = figure('name','Voronoi');
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
	print(h_fig,'-dpng',strcat(['fig_voronoi.png']));
end
