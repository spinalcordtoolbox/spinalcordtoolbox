function j_fsl_colormap()
% =========================================================================
% 
% 
% 
% INPUT
% -------------------------------------------------------------------------
% 
% 
% -------------------------------------------------------------------------
% OUTPUT
% -------------------------------------------------------------------------
% 
% -------------------------------------------------------------------------
% 
%   Example
%   j_fsl_colormap
%
%
% Julien Cohen-Adad <jcohen@nmr.mgh.harvard.edu>
% 2011-12-29: Created
%
% =========================================================================

% PARAMETERS
fname			= 'j_blue-black-red.lut';
nb_bits			= 128;


% START FUNCTION
% j_disp(fname_log,['\n\n\n=========================================================================================================='])
% j_disp(fname_log,['   Running: j_fsl_colormap'])
% j_disp(fname_log,['=========================================================================================================='])
% j_disp(fname_log,['.. Started: ',datestr(now)])


% =========================================================================
% COLORMAP ALGORITHMS
% =========================================================================

% black -> red
% c.red = (0:1/(nb_bits-1):1);
% c.blue = 0*(0:1/(nb_bits-1):1);
% c.green = 0*(0:1/(nb_bits-1):1);


% blue -> black -> red
red = cat(2,zeros(1,nb_bits/2),(0:2/(nb_bits-1):1));
green = 0*(0:1/(nb_bits-1):1);
blue = cat(2,(1:-2/(nb_bits-1):0),zeros(1,nb_bits/2));





% =========================================================================

% create file
fid = fopen(fname,'w');
fprintf(fid,'%!VEST-LUT \n');
fprintf(fid,'%%BeginInstance \n');
fprintf(fid,'<< \n');
fprintf(fid,'/SavedInstanceClassName /ClassLUT \n');
fprintf(fid,'/PseudoColorMinimum 0.00 \n');
fprintf(fid,'/PseudoColorMaximum 1.00 \n');
fprintf(fid,'/PseudoColorMinControl /Low  \n');
fprintf(fid,'/PseudoColorMaxControl /High \n');
fprintf(fid,'/PseudoColormap [ \n');
for i=1:nb_bits
	fprintf(fid,'<-color{%1.6f,%1.6f,%1.6f}-> \n',red(i),green(i),blue(i));
end
fprintf(fid,'] \n');
fprintf(fid,'>> \n');
fprintf(fid,'%%EndInstance \n');
fprintf(fid,'%%EOF \n');

fclose(fid);



disp(['File created: ',fname])


% END FUNCTION
% j_disp(fname_log,['\n.. Ended: ',datestr(now)])
% j_disp(fname_log,['==========================================================================================================\n'])