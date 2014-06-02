function Tfit = j_dmri_eddyCorrect_fitTransfo(gx,gy,gz,T,Tlabel,nb_loops,display_fig)
% =========================================================================
% 
% Fit transformation coefficients.
% 
% CALLED BY
% -------------------------------------------------------------------------
% j_dmri_eddyCorrect.m
% 
% 
% INPUT
% -------------------------------------------------------------------------
% TODO
% -------------------------------------------------------------------------
% 
% OUTPUT
% -------------------------------------------------------------------------
% Tfit
% -------------------------------------------------------------------------
%
% Julien Cohen-Adad <jcohen@nmr.mgh.harvard.edu>
% 2011-12-24: Created
% 
% =========================================================================



coeff_fit = zeros(3,nb_loops);
Tfit = zeros(size(gx,1),nb_loops);

% fit 3D
ind_x = find(gx(:,1));
ind_y = find(gy(:,1));
ind_z = find(gz(:,1));
for iZ=1:nb_loops
	GX = gx(ind_x,iZ);
	GY = gy(ind_y,iZ); % TODO: no need to separate ind_x, ind_y and ind_z
	GZ = gz(ind_z,iZ);
	G = [GX GY GZ];
	Tind = T(ind_x,iZ);
	coeff_fit(:,iZ) = inv(G'*G)*G'*Tind;
end
% 
% % fit Gx
% ind_x = find(gx(:,1)); 
% for iZ=1:nb_loops
% 	coeff_fit(1,iZ) = inv(gx(ind_x,iZ)'*gx(ind_x,iZ))*gx(ind_x,iZ)'*T(ind_x,iZ);
% end
% % fit Gy
% ind_y = find(gy(:,1));
% for iZ=1:nb_loops
% 	coeff_fit(2,iZ) = inv(gy(ind_y,iZ)'*gy(ind_y,iZ))*gy(ind_y,iZ)'*T(ind_y,iZ);
% end
% % fit Gz
% ind_z = find(gz(:,1));
% for iZ=1:nb_loops
% 	coeff_fit(3,iZ) = inv(gz(ind_z,iZ)'*gz(ind_z,iZ))*gz(ind_z,iZ)'*T(ind_z,iZ);
% end

% Generate Tfit values
for iZ=1:nb_loops
	Tfit(ind_x,iZ) = gx(ind_x,iZ) * coeff_fit(1,iZ);
	Tfit(ind_y,iZ) = gy(ind_y,iZ) * coeff_fit(2,iZ);
	Tfit(ind_z,iZ) = gz(ind_z,iZ) * coeff_fit(3,iZ);
end


% display fit
if display_fig
	h_fig = figure;
	title('Transformation coefficients')

	subplot(2,2,1)
	for iZ=1:nb_loops
		plot(gx(ind_x,iZ),T(ind_x,iZ),'.','markersize',15,'color',[(iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5)]); xlabel('Gx'), ylabel('Ty (voxel)'), hold on
		plot(gx(ind_x,iZ),Tfit(ind_x,iZ),'.','markersize',10,'color',[1 (iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5)]); xlabel('Gx'), ylabel([Tlabel,' (voxel)']), hold on
	end
	grid
	subplot(2,2,2)
	for iZ=1:nb_loops
		plot(gy(ind_y,iZ),T(ind_y,iZ),'.','markersize',15,'color',[(iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5)]); xlabel('Gy'), ylabel('Ty (voxel)'), hold on
		plot(gy(ind_y,iZ),Tfit(ind_y,iZ),'.','markersize',10,'color',[1 (iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5)]); xlabel('Gy'), ylabel([Tlabel,' (voxel)']), hold on
	end
	grid
	subplot(2,2,3)
	for iZ=1:nb_loops
		plot(gz(ind_z,iZ),T(ind_z,iZ),'.','markersize',15,'color',[(iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5)]); xlabel('Gz'), ylabel('Ty (voxel)'), hold on
		plot(gz(ind_z,iZ),Tfit(ind_z,iZ),'.','markersize',10,'color',[1 (iZ-1)/(nb_loops*1.5) (iZ-1)/(nb_loops*1.5)]); xlabel('Gz'), ylabel([Tlabel,' (voxel)']), hold on
	end
	grid
	print(h_fig,'-dpng',['eddyCorr_',Tlabel,'.png'])
	close

	% Plot z-dependence
	h_fig = figure;
	plot(coeff_fit','linewidth',2), grid
	title('Z-dependence')
	legend('Gx','Gy','Gz')
	xlabel('Z')
	ylabel([Tlabel,'fit/G'])
	print(h_fig,'-dpng',['eddyCorr_ZDependence_',Tlabel,'.png'])
	close
end

Tfit = Tfit';
