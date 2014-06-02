

function grad_list = j_dmri_gradient_siemens_get(file_read)



if nargin==0
	file_read					= 'DiffusionVectors_ElectrostaticRepulsion_200_b0every15.txt';
end

% read standard Siemens gradient file
txt = textread(file_read,'%s');
nb_chars = size(txt,1);

% find first vector string
for i=1:nb_chars
	if strcmp(txt{i},'vector[')
		index_vector = i;
		break
	end
end
index_vector = index_vector + 4;

% get the diffusion directions
j=1;
grad_list = [];
for i=index_vector:8:nb_chars
	grad_list(j,:) = [str2num(txt{i}(1:end-1)) str2num(txt{i+1}(1:end-1)) str2num(txt{i+2})];
	j=j+1;
end
nb_dirs = j-1;



