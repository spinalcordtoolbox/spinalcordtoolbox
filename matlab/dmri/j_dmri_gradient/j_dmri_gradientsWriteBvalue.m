% =========================================================================
% FUNCTION
% j_dmri_gradientsWriteBvalue()
%
% Write gradient file
%
% INPUT
% bvalue				nx1 integer
% fname_write			string
% gradient_format		string
% 
% OUTPUT
% (-)
%
% COMMENTS
% julien cohen-adad 2009-08-31
% =========================================================================
function j_dmri_gradientsWriteBvalue(bvalue,fname_write,gradient_format)


% open file and write header
fid_w = fopen(fname_write,'w');

nb_directions = max(size(bvalue));

switch gradient_format

	case 'fsl'

	for i_gradient = 1:nb_directions
	    fprintf(fid_w,'%i\n',bvalue(i_gradient));
	end
	
	case 'medinria'


end

% close files
fclose(fid_w);


