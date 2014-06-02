function j_dmri_gradient_siemens_create(opt)
% =========================================================================
% Function to generate gradient table from an existing Siemens table.
% 
% INPUT
% (opt)
%	% User Parameters:
% 	file_read					= '../DiffusionVectorsDSI_1418.txt';
% 	file_write					= 'jDSI1418_revGrad_Random_20b0';  Better not to put the extension
% 	input_format				= 'siemens'* | 'fsl'
% 	output_format				= 'siemens' | 'fsl'*
% 	info						= ['# Generated from ''DiffusionVectors_ElectrostaticRepulsion'' on ',datestr(now,29),' by jcohen@nmr.mgh.harvard.edu.'];
% 	CoordinateSystem			= 'PRS'; % 'XYZ' (absolute coordinates) | 'PRS' (patient coordinates)
% 	Normalisation				= 'None'; 
% 
% 	% vector lists:
% 	untouched					= 1; % 1: don't change anything to the gradient list, i.e. ALL FOLLOWING PARAMETERS ARE IGNORED.
% 	b0_at_beginning				= 0; % add one b0 at the beginning.
% 	interspersed_b0				= 20; % Frequency of b0 appearence (i.e. "every 'interspersed_b0' image you get a b0"). Set to 0 for keeping the matrix as it is. Note!! If using opposite gradient directions next to each other, then you should use an odd number (e.g. 15, not 14).
% 	last_chunck_at_beginning	= 0; % put the last 'last_chunck_at_beginning' directions at the beginning, to avoid screwing up the acquisition with high duty cycle at large b-values (only for DSI). Set to 0 for no change.
% 	randomize_directions		= 1; % randomize gradient directions to distribute duty cycle more equally in case of increasing b-values (only for DSI)
% 	opposite_direction			= 0; % add opposite directions to the existing ones at the end of the bvecs table (for eddy-current correction). It will effectively double the number of directions.
% 	opposite_direction_adj		= 1; % if the gradient table already contains opposite directions, identy them and put them next to each other (to minimize the amount of potential subject motion between the two images that will be subsequently used for eddy-current distortion correction)
%   multiple_shells				= [1]; % multiple shells can be prescribed as a q-ratio between 0 and 1. For single shell, use [1]. For two shells at 50% and 100%, use [0.5 1]. The max value will be defined by the b-value entered during the acquistion.
% 
% 	% misc:
% 	display_gradients			= 0; % display gradients at the end
%	display_voronoi				= 0; % display voronoi
% 	divide_bvecs				= 1; % divide bvecs into 2, 3 or more files if you want to divide your acquisition into multiple runs. For no division, put 1. Default=1.
% 	create_bvals				= 0; % create bvals for FSL.
% 	bvalue						= 1000; % enter b-value in mm2/s
% 	file_bvals					= 'bvals'; % file name for b-value file
% 
% 	% Default parameters:
% 	min_norm					= 0.001; % ONLY USE WITH opposite_direction_adj. Minimim norm of two opposite gradient directions (it should be 0 if they are in perfect opposition, but for some reasons, e.g., truncated values, it is note the case. Suggested value=0.001).
% 	fname_log					= 'log_j_dmri_gradient_siemens_create.txt';
% 
% OUTPUT
% -
%
% Julien Cohen-Adad <jcohen@nmr.mgh.harvard.edu>
% 2011-10-07: Created
% 2011-11-05: Fixed bug to identify "Vector[" string
% 2011-11-07: Allows to write FSL files
% 2011-11-19: Allows to have opposite gradient directions next to each other (for eddy-current correction)
% 2011-11-20: divide by two
% 2011-11-26: added an 'untouched' parameter.
% 2011-12-04: creates bvals file.
% 2012-01-16: enables input to make batch file and keep copy of parameters.
% 2012-03-19: enables to create multiple shells from single-shell sampling.
% 2012-11-18: changed default values
% =========================================================================




if ~exist('opt'), opt = []; end
if isfield(opt,'file_read'), file_read = opt.file_read; else file_read = ''; end
if isfield(opt,'file_write'), file_write = opt.file_write; else file_write = 'bvecs_new.txt'; end
if isfield(opt,'input_format'), input_format = opt.input_format; else input_format = 'siemens'; end
if isfield(opt,'output_format'), output_format = opt.output_format; else output_format = 'fsl'; end
if isfield(opt,'info'), info = opt.info; else info = ['# Generated from ''DiffusionVectors_ElectrostaticRepulsion'' on ',datestr(now,29),' by jcohen@nmr.mgh.harvard.edu.']; end
if isfield(opt,'CoordinateSystem'), CoordinateSystem = opt.CoordinateSystem; else CoordinateSystem = 'PRS'; end
if isfield(opt,'Normalisation'), Normalisation = opt.Normalisation; else Normalisation = 'None'; end
if isfield(opt,'untouched'), untouched = opt.untouched; else untouched = 1; end
if isfield(opt,'b0_at_beginning'), b0_at_beginning = opt.b0_at_beginning; else b0_at_beginning = 0; end
if isfield(opt,'interspersed_b0'), interspersed_b0 = opt.interspersed_b0; else interspersed_b0 = 20; end
if isfield(opt,'last_chunck_at_beginning'), last_chunck_at_beginning = opt.last_chunck_at_beginning; else last_chunck_at_beginning = 0; end
if isfield(opt,'randomize_directions'), randomize_directions = opt.randomize_directions; else randomize_directions = 0; end
if isfield(opt,'opposite_direction'), opposite_direction = opt.opposite_direction; else opposite_direction = 0; end
if isfield(opt,'opposite_direction_adj'), opposite_direction_adj = opt.opposite_direction_adj; else opposite_direction_adj = 1; end
if isfield(opt,'display_gradients'), display_gradients = opt.display_gradients; else display_gradients = 0; end
if isfield(opt,'divide_bvecs'), divide_bvecs = opt.divide_bvecs; else divide_bvecs = 1; end
if isfield(opt,'opposite_direction'), opposite_direction = opt.opposite_direction; else opposite_direction = 0; end
if isfield(opt,'create_bvals'), create_bvals = opt.create_bvals; else create_bvals = 0; end
if isfield(opt,'bvalue'), bvalue = opt.bvalue; else bvalue = 1000; end
if isfield(opt,'file_bvals'), file_bvals = opt.file_bvals; else file_bvals = 'bvals'; end
if isfield(opt,'min_norm'), min_norm = opt.min_norm; else min_norm = 0.001; end
if isfield(opt,'fname_log'), fname_log = opt.fname_log; else fname_log = 'log_j_dmri_gradient_siemens_create.txt'; end
if isfield(opt,'multiple_shells'), multiple_shells = opt.multiple_shells; else multiple_shells = [1]; end
if isfield(opt,'display_voronoi'), display_voronoi = opt.display_voronoi; else display_voronoi = 1; end


% check input file
if isempty(file_read)
	disp(['ERROR: no input file.']), help j_dmri_gradient_siemens_create, return
end


% delete log file
if exist(fname_log), delete(fname_log), end

% START FUNCTION
j_disp(fname_log,['\n\n\n=========================================================================================================='])
j_disp(fname_log,['   Running: j_dmri_gradient_siemens_create'])
j_disp(fname_log,['=========================================================================================================='])
j_disp(fname_log,['.. Started: ',datestr(now)])

% debug if error
dbstop if error


if strcmp(input_format,'siemens')

	% read standard Siemens gradient file
	j_disp(fname_log,['\nRead file...'])
	j_disp(fname_log,['.. File name: ',file_read])
	txt = textread(file_read,'%s');
	nb_strings = size(txt,1);


	% find first vector string
	j_disp(fname_log,['\nRegex search...'])
	for i=1:nb_strings
		if ~isempty(strfind(txt{i},'vector[')) || ~isempty(strfind(txt{i},'Vector['))
			j_disp(fname_log,['.. Identified vector string: "',txt{i},'" at i=',num2str(i)])
			index_vector = i;
			break
		end
	end
	iDir = 1;
	direction = [];
	for i=index_vector:nb_strings
		test_string = str2num(txt{i});
		if isnumeric(test_string) && ~isempty(test_string)
			direction(iDir) = test_string;
			iDir = iDir+1;
		end
	end
	nb_dirs = length(direction)/3;
	j_disp(fname_log,['.. Number of directions: ',num2str(nb_dirs)])
	% test number of directions
	if round(nb_dirs)~=nb_dirs
		error('Failed to identify gradient directions. Please check gradient table.')
	end
	% reshape
	grad_list = reshape(direction,[3 nb_dirs])';

elseif strcmp(input_format,'fsl')

	grad_list = textread(file_read);
	nb_dirs = size(grad_list,1);
	
end


% Identify b=0 and DWI
j_disp(fname_log,['\nIdentify b=0 and DWI...'])
index_b0 = [];
index_dwi = [];
for iT=1:nb_dirs
	if norm(grad_list(iT,:))==0
		index_b0 = cat(2,index_b0,iT);
	else
		index_dwi = cat(2,index_dwi,iT);
	end
end
j_disp(fname_log,['.. Index of b=0 images: ',num2str(index_b0)])
j_disp(fname_log,['.. Index of DWI images: ',num2str(index_dwi)])



% remove b=0 images
if ~untouched
	j_disp(fname_log,['\nRemove b=0 images...'])
	grad_list = grad_list(index_dwi,:);
	nb_dirs = size(grad_list,1);
	j_disp(fname_log,['.. Updated number of directions: ',num2str(nb_dirs)])
end



% put the X last in the beginning
if last_chunck_at_beginning && ~untouched
	j_disp(fname_log,['\nPut the last ',num2str(last_chunck_at_beginning),' directions at the beginning, to avoid screwing up the acquisition with high duty cycle (only for DSI)...'])
	grad_list_temp = grad_list;
	clear grad_list
	grad_list = cat(1,grad_list_temp(end-last_chunck_at_beginning+1:end,:),grad_list_temp(1:end-last_chunck_at_beginning,:));
	% update info header
	info = [info,' The last ',num2str(last_chunck_at_beginning),' directions are put at the beginning.'];
end



% randomize gradient directions to distribute duty cycle more equally in case of increasing b-values (only for DSI)
if randomize_directions && ~untouched
	j_disp(fname_log,['\nRandomize directions...'])
	index_random = randperm(nb_dirs);
	grad_list = grad_list(index_random,:);
	j_disp(fname_log,['.. New indices: ',num2str(index_random)])
	% update info header
	info = [info,' Gradient directions were randomized.'];
end
	


% generate opposite direction (for eddy-current correction)
if opposite_direction && ~untouched
	j_disp(fname_log,['\nGenerate opposite directions (for eddy-current distortion correction)...'])
	grad_list = cat(1,grad_list,-grad_list);
	nb_dirs = 2*nb_dirs;
end



% re-organize opposite directions so that they are next to each others (for eddy-current correction)
if opposite_direction_adj && ~untouched
	
	% Identify pairs of opposite gradient directions
	j_disp(fname_log,['\nIdentify pairs of opposite gradient directions...'])
	iN = 1;
	opposite_gradients = {};
	for iT=1:nb_dirs
		for jT=1:nb_dirs
			if norm(grad_list(iT,:)+grad_list(jT,:))<min_norm && norm(grad_list(iT,:))~=0 && iT<jT
				j_disp(fname_log,['.. Opposite gradient for #',num2str(iT),' is: #',num2str(jT)])
				opposite_gradients{iN} = [iT,jT];
				iN = iN+1;
				break
			end
		end
	end
	nb_oppositeGradients = length(opposite_gradients);
	j_disp(fname_log,['.. Number of opposite gradient directions: ',num2str(nb_oppositeGradients)])
	
	% Put opposite directions next to each other
	j_disp(fname_log,['\nPut opposite directions next to each other...'])
	grad_list_temp = zeros(nb_oppositeGradients*2,3);
	for iN=1:1:nb_oppositeGradients
		grad_list_temp(2*iN-1,:) = grad_list(opposite_gradients{iN}(1),:);
		grad_list_temp(2*iN,:) = grad_list(opposite_gradients{iN}(2),:);
	end
	grad_list = grad_list_temp;
	nb_dirs = size(grad_list,1);
	% update info header
	info = [info,' Opposite gradient directions were put next to each other (for eddy-current distortion correction).'];
end



% multiple shells
if ~untouched
	j_disp(fname_log,['\nBuild multiple shells...'])
	nb_shells = length(multiple_shells);
	grad_list_temp = [];
	for i_shell = 1:nb_shells
		j_disp(fname_log,['.. Shell #',num2str(i_shell),' --> ',num2str(multiple_shells(i_shell))])
		grad_list_scaled = grad_list .* multiple_shells(i_shell);
		grad_list_temp = cat(1,grad_list_temp,grad_list_scaled);
	end
	% update nb_dirs
	grad_list = grad_list_temp;
	nb_dirs = size(grad_list,1);
	j_disp(fname_log,['.. Updated number of directions: ',num2str(nb_dirs)])
	clear grad_list_temp
	% update info header
	txt_shell = [num2str(multiple_shells(1))];
	for i_shell=2:nb_shells
		txt_shell = [txt_shell,', ',num2str(multiple_shells(i_shell))];
	end
	info = [info,' Multiple shells: {',txt_shell,'}.'];
end



% add interspersed b=0
% interspersed_b0 = interspersed_b0-1;
if interspersed_b0 && ~untouched
	j_disp(fname_log,['\nAdd interspersed b=0...'])
	j_disp(fname_log,['.. every ',num2str(interspersed_b0),' volumes (start with one at the beginning)'])
	% add b0
	grad_list_temp = [];
	count_b0 = interspersed_b0;
	iDir = 1;
% 	shifted = 0;
	while iDir<=nb_dirs
		% don't add b=0 if the previous direction is the opposite as the current one (because eddy-current correction will be applied before motion correction)
		if count_b0 >= interspersed_b0
			if iDir ~= 1
% 				if norm(grad_list(iDir-1,:)+grad_list(iDir,:)) > min_norm
% 					if shifted
% 						j_disp(fname_log,['.. Insert b=0 at direction #',num2str(iDir),' (shifted by one to keep opposite gradient directions next to each other)'])
% 					else
 						j_disp(fname_log,['.. Insert b=0 at direction #',num2str(iDir)])
% 					end
 					grad_list_temp = cat(1,grad_list_temp,[0 0 0]);
 					count_b0 = 1;
%  					shifted = 0;
% 				else
					grad_list_temp = cat(1,grad_list_temp,grad_list(iDir,:));
					iDir = iDir + 1;
					count_b0 = count_b0 + 1;
% 					shifted = 1;
% 				end
			else
				j_disp(fname_log,['.. Insert b=0 at direction #',num2str(iDir)])
				grad_list_temp = cat(1,grad_list_temp,[0 0 0]);
				count_b0 = 1;
			end				
		else
			grad_list_temp = cat(1,grad_list_temp,grad_list(iDir,:));
			iDir = iDir + 1;
			count_b0 = count_b0 + 1;
		end
	end
	% update nb_dirs
	grad_list = grad_list_temp;
	nb_dirs = size(grad_list,1);
	j_disp(fname_log,['.. Updated number of directions: ',num2str(nb_dirs)])
	clear grad_list_temp
	% update info header
	info = [info,' Contains b=0 every ',num2str(interspersed_b0),' measurements.'];
end



% add b=0 at the beginning
if b0_at_beginning && ~untouched
	j_disp(fname_log,['\nAdd b=0 at the beginning...'])
	grad_list = cat(1,[0 0 0],grad_list);
	nb_dirs = size(grad_list,1);
	j_disp(fname_log,['.. Updated number of directions: ',num2str(nb_dirs)])
	% update info header
	info = [info,' Added one b=0 at the beginning.'];
end



% Write file(s)
if untouched, divide_bvecs = 1; end
j_disp(fname_log,['\nWrite output file (divide into ',num2str(divide_bvecs),' chunks)...'])
ind_last = 1;
for i_bvecs = 1:divide_bvecs

	% since division may not be integer, make sure the last chuck has all the directions
	if i_bvecs == divide_bvecs
		nb_dirs_div = nb_dirs - (i_bvecs-1)*round(nb_dirs/divide_bvecs);
	else
		nb_dirs_div = round(nb_dirs/divide_bvecs);
	end
	j_disp(fname_log,['\n.. Chunk #',num2str(i_bvecs),': ',num2str(nb_dirs_div),' directions'])
	
	% get sub-grad list
	grad_list_div = grad_list(ind_last:ind_last+nb_dirs_div-1,:);
	ind_last = i_bvecs*nb_dirs_div+1;
	
	% get file name (in case division, add _X at the end)
	if divide_bvecs ~= 1
		file_write_div = [file_write,'_',num2str(i_bvecs),'of',num2str(divide_bvecs)];
		info_div = [info,' Direction set was divided into several chunks --> This file number is: ',num2str(i_bvecs),'/',num2str(divide_bvecs)];
	else
		file_write_div = file_write;
		info_div = [info];
	end
	
	% delete file (if exist)
	if exist(file_write_div), delete(file_write_div), end
	
	% Write output file
% 	j_disp(fname_log,['\nWrite output file...'])
	j_disp(fname_log,['.. Output format: ',output_format])
	fid = fopen(file_write_div,'w');
	if strcmp(output_format,'siemens')
		% generate Siemens file
		fprintf(fid,[info_div,'\n']);
		fprintf(fid,['[Directions=',num2str(nb_dirs_div),']\n']);
		fprintf(fid,['CoordinateSystem = ',CoordinateSystem,'\n']);
		fprintf(fid,['Normalisation = ',Normalisation,'\n']);
		for i=1:nb_dirs_div
			fprintf(fid,'vector[ %i] = ( %1.10f, %1.10f, %1.10f )\n',i-1,grad_list_div(i,1),grad_list_div(i,2),grad_list_div(i,3));
		end

	elseif strcmp(output_format,'fsl')
		% Generate FSL file
		for i=1:nb_dirs_div
			fprintf(fid,'%1.10f %1.10f %1.10f\n',grad_list_div(i,1),grad_list_div(i,2),grad_list_div(i,3));
		end
	end
	fclose(fid);
	j_disp(fname_log,['.. File written: ',file_write_div])

end



% Identify b=0 and DWI
j_disp(fname_log,['\nSummary of the file...'])
index_b0 = [];
index_dwi = [];
for iT=1:nb_dirs
	if norm(grad_list(iT,:))==0
		index_b0 = cat(2,index_b0,iT);
	else
		index_dwi = cat(2,index_dwi,iT);
	end
end
nb_b0 = length(index_b0);
nb_dwi = length(index_dwi);
j_disp(fname_log,['.. Number of directions: ',num2str(nb_dirs)])
j_disp(fname_log,['.. Index of b=0 images: (',num2str(nb_b0),'): ',num2str(index_b0)])
j_disp(fname_log,['.. Index of DWI images: (',num2str(nb_dwi),'): ',num2str(index_dwi)])



% Create bvals
if create_bvals
	j_disp(fname_log,['\nCreate bvals...'])
	fid = fopen(file_bvals,'w');
	for i=1:nb_dirs
		if ~isempty(find(index_b0==i))
			fprintf(fid,'%i\n',0);
		elseif ~isempty(find(index_dwi==i))
			if length(multiple_shells)>1
				% multiple shells
				bvalue_shell = int16(bvalue*(norm(grad_list(i,:)))^2);
				fprintf(fid,'%i\n',bvalue_shell);
			else
				% single shell
				fprintf(fid,'%i\n',bvalue);
			end
		end
	end
	fclose(fid);
	j_disp(fname_log,['.. File written: ',file_bvals])

end

% grad_list_new = grad_list^2;

for iDir=1:nb_dirs
	for iDim = 1:3
 		grad_list_new(iDir,iDim) = sign(grad_list(iDir,iDim)) * (15000*grad_list(iDir,iDim))^2;
% 		grad_list_new(iDir,iDim) = sign(grad_list(iDir,iDim)) * sqrt(abs(grad_list(iDir,iDim)));
	end
end

% display gradients
if display_gradients
	j_disp(fname_log,['\nDisplay and print gradient vectors...'])
	gradientsDisplay(grad_list(index_dwi',:),display_voronoi);
end


% end
j_disp(fname_log,['\n.. Ended: ',datestr(now)])
j_disp(fname_log,['==========================================================================================================\n'])








% =========================================================================
% FUNCTION
% gradientsDisplay.m
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
	print(h_fig,'-dpng',strcat(['fig_bvecs.png']));
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

print(h_fig,'-dpng',strcat(['fig_bvecs_3axis.png']));


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
