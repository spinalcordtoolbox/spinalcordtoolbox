
% =========================================================================
% FUNCTION
% j_dmri_gradientsWrite
%
% Write gradient file for Siemens console
%
% INPUT
% bvecs					nx3 double
% fname_write			string
% gradient_format		string			'fsl', 'medinria', 'siemens' 
% 
% OUTPUT
% (-)
%
% COMMENTS
% julien cohen-adad 2009-08-31
% =========================================================================
function j_dmri_gradientsWrite(bvecs,fname_write,gradient_format)


info			= '# Contains B0 every 10 measurements. Generated from ''Product_80dir.txt'' on 2009-11-01 by jcohen@nmr.mgh.harvard.edu.';
CoordinateSystem = 'PRS';
Normalisation	= 'None';

% default initialization
% fname_dicom             = 'D:\data_irm\2007-01-23_catSpine-DTI\dicom\folder_01\08-DIFFUSION_b800_sag_grappa\Chat_Gx1_2-08-0001.dcm';
% fname_gradientRead      = 'D:\mes_documents\IRM\DTI\MedINRIA_12directions_axial.txt';
% fname_gradientWrite     = 'D:\data_irm\gradients_12dir-08.txt';
% disp_text               = 1;
% create_raw				= 1;
% 
% % user initialization 
% if ~exist('opt'), opt = []; end
% if isfield(opt,'fname_dicom'), fname_dicom = opt.fname_dicom; end
% if isfield(opt,'fname_gradientRead'), fname_gradientRead = opt.fname_gradientRead; end
% if isfield(opt,'fname_gradientWrite'), fname_gradientWrite = opt.fname_gradientWrite; end
% if isfield(opt,'disp_text'), disp_text = opt.disp_text; end

% open file and write header
fid = fopen(fname_write,'w');

nb_directions = size(bvecs,1);

switch gradient_format

case 'fsl'

	for i_gradient = 1:nb_directions
	    fprintf(fid,'%1.10f %1.10f %1.10f\n',bvecs(i_gradient,1),bvecs(i_gradient,2),bvecs(i_gradient,3));
	end
	
case 'medinria'

    fprintf(fid,'%i\n',nb_directions);
	for i_gradient = 1:nb_directions
	    fprintf(fid,'%1.10f %1.10f %1.10f\n',bvecs(i_gradient,1),bvecs(i_gradient,2),bvecs(i_gradient,3));
	end
	
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


