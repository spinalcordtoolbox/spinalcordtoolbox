

frequency		= 10; % frequency of B0 appearence (i.e. "every 'frequency' image you get a B0")
file_name		= '515dir_B0.txt';
info			= '# Contains B0 every 10 measurements. Generated from ''sphere514.txt'' on 2009-10-12 by jcohen@nmr.mgh.harvard.edu.';
CoordinateSystem = 'PRS';
Normalisation	= 'None';

% read standard Siemens gradient file
txt = textread('515dir.txt','%s');
j=1;
for i=5:8:4120
	grad_list(j,:) = [str2num(txt{i}(1:end-1)) str2num(txt{i+1}(1:end-1)) str2num(txt{i+2})];
	j=j+1;
end
nb_dirs = j-1;

% add B0
nb_dirs_new = nb_dirs+round(nb_dirs/(frequency-1));
grad_list_new = zeros(nb_dirs_new,3);
i_list = 1;
j=1;
k=1;
for i=1:size(grad_list_new,1)
	if j==frequency
		grad_list_new(k,:) = [0 0 0];
		k=k+1;
		j=1;
	else
		grad_list_new(k,:) = grad_list(i_list,:);
		i_list = i_list+1;
		k=k+1;
		j=j+1;
	end
end

% generate Siemens file
fid = fopen(file_name,'w');
fprintf(fid,[info,'\n']);
fprintf(fid,['[Directions=',num2str(nb_dirs_new),']\n']);
fprintf(fid,['CoordinateSystem = ',CoordinateSystem,'\n']);
fprintf(fid,['Normalisation = ',Normalisation,'\n']);
for i=1:nb_dirs_new
% 	txt = ['vector[',num2str(i),'] = ( ',num2str()]
	fprintf(fid,'vector[ %i] = ( %f, %f, %f )\n',i-1,grad_list_new(i,1),grad_list_new(i,2),grad_list_new(i,3));
end
fclose(fid);
% j_dmri_gradientsDisplay(grad_list)
