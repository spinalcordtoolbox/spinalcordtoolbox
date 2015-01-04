
% =========================================================================
% FUNCTION
% j_dmri_gradients_writeBvecsBvals
%
% Write bvecs/bvals files for FSL
%
% INPUT
% bvecs					nx3 double
% 
% OUTPUT
% gradients				structure
%
% COMMENTS
% julien cohen-adad 2010-01-25
% =========================================================================
function gradients = j_dmri_gradients_writeBvecsBvals(grad_list)


bvalue = 1000;
gradient_format = 'fsl';
path_write = pwd;
file_bvecs = 'bvecs.txt';
file_bvals = 'bvals.txt';


% build file names
path_write = [path_write,filesep];
fname_bvecs = [path_write,file_bvecs];
fname_bvals = [path_write,file_bvals];

% open file and write header
fid = fopen(fname_bvecs,'w');

nb_directions = size(grad_list,1);

switch gradient_format

case 'fsl'

	for i_gradient = 1:nb_directions
	    fprintf(fid,'%1.10f %1.10f %1.10f\n',grad_list(i_gradient,1),grad_list(i_gradient,2),grad_list(i_gradient,3));
		% update bvals
		if grad_list(i_gradient,:) == [0 0 0]
			bvals(i_gradient) = 0;
		else
			bvals(i_gradient) =bvalue;
		end
	end
	
case 'medinria'

case 'siemens'

	fprintf(fid,[info,'\n']);
	fprintf(fid,['[Directions=',num2str(nb_directions),']\n']);
	fprintf(fid,['CoordinateSystem = ',CoordinateSystem,'\n']);
	fprintf(fid,['Normalisation = ',Normalisation,'\n']);
	for i=1:nb_directions
	% 	txt = ['vector[',num2str(i),'] = ( ',num2str()]
		fprintf(fid,'vector[ %i] = ( %f, %f, %f )\n',i-1,bvecs(i,1),bvecs(i,2),bvecs(i,3));
	end

end

% close files
fclose(fid);




% write bvals file
fid = fopen(fname_bvals,'w');
nb_directions = size(grad_list,1);
for i_gradient = 1:nb_directions
	fprintf(fid,'%i\n',bvals(i_gradient));
end
fclose(fid);


gradients.bvals = bvals;
gradients.bvecs = grad_list;

