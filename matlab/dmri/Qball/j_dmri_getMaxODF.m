% =========================================================================
% FUNCTION
% -------------------------------------------------------------------------
% j_dmri_getMaxODF.m
% 
%
% DESCRIPTION
% -------------------------------------------------------------------------
% Find maxima of the ODF based on a given sampling of the sphere (e.g.,
% 181, 362, ...). The algorithm searches for local maxima on the ODF.
% 
% To improve the robustness of the algorithm, it takes the mean of both
% antipodal local maxima, in a parametric way (i.e., the definition of the 
% principal directions does NOT depend on the sampling of the sphere). Of
% course, the better the original sampling of the sphere, the more accurate
% the definition of the principal direction is.
% 
% 
% INPUTS
% -------------------------------------------------------------------------
% odf					(1xn) float		values in amplitudes on the sphere
% scheme				struct.
% 
% *** OPTIONS: (to be specified like: 'option_name','option_value').
% nearest_neighb		cell			neighrest neighbours each sample on the sphere
% output				string			'discreet','parametric'*. Definition of the maxima is either discreet (depends on the sampling of the ODF and defined as a nx1 matrix) or parametric (defined as a 3x1 vector, this option is way better)
% 
% 
% OUTPUTS
% -------------------------------------------------------------------------
% if output is discreet:
%	max_odf				nxm	binary		n=nb of maxima, m=sampling of the ODF
%	nb_maxima			integer
%	sorted_maxima		1xn integer
%
% if output is parametric:
%	max_odf_param		nx3 float		vector coordinates of each maximum
%	nb_maxima			integer			number of maxima
%
% 
% COPYRIGHT
% Julien Cohen-Adad 2009-10-29
% Massachusetts General Hospital
% =========================================================================
function varargout = j_dmri_getMaxODF(odf,scheme,varargin)


% default parameters
maxima_threshold		= 0.33;
output					= 'parametric';

for i = 1:nargin-2
	if strcmp(varargin{i},'nearest_neighb'), nearest_neighb = varargin{i+1}; end
	if strcmp(varargin{i},'maxima_threshold'), maxima_threshold = varargin{i+1}; end
	if strcmp(varargin{i},'output'), output = varargin{i+1}; end
end

% Get the maxima of the ODF (to apply subsequent threshold on the maxima)
peak_odf = max(odf);

% odf = squeeze(dmri.bootstrap.odf_mean(1,1,1,:));
% scheme = dmri.bootstrap.scheme_visu;
% el = scheme.el;
% az = scheme.az;
vert = scheme.vert;

nb_samples=size(vert,1);
% TODO: don't need the whole sphere

if ~exist('nearest_neighb')
	% Find each sample's neighbour on the sphere
	nearest_neighb = {};
	distvect = [];
	for isample=1:nb_samples

		% compute the euclidean distance between that guy and all other guys
		for i=1:nb_samples
			distvect = vert(isample,:)-vert(i,:);
			normdist(i) = norm(distvect);
		end
		% sort the distance in the increasing order
		[val nearest_neighb_tmp] = sort(normdist);
		% remove the first guy
		nearest_neighb_tmp = nearest_neighb_tmp(2:end);
		% save the first nb_neighbours
		nearest_neighb{isample} = nearest_neighb_tmp(1:nb_neighbours);
	end
	clear nearest_neighb_tmp
end

nb_neighbours			= length(nearest_neighb{1}); % make it depends on the sampling


% Loop on each sample on the half-sphere to find local maxima
max_odf = zeros(nb_samples,1);
dont_test = [];
j=1;
for isample = 1:nb_samples

	% check if this guy is not inferior to an already found local maxima
	if isempty(find(dont_test==isample))
	
		% check if this guy is superior to the given threshold
		if (odf(isample)/peak_odf)>maxima_threshold
			
			% check if this guy is superior to all its neighbours
			sup_neighb = 0;
			for ineighb = 1:nb_neighbours

% fprintf('icol=%d, ineighb=%d, odf[icol]=%f, odf[index_neighbour]=%f\n',isample,ineighb,odf(isample),odf(nearest_neighb{isample}(ineighb)));

				if odf(isample)>odf(nearest_neighb{isample}(ineighb))
					sup_neighb = sup_neighb+1;
				end		
			end
			if sup_neighb==nb_neighbours
% fprintf('icol=%d, odf[icol]=%f\n',isample,odf(isample));
				% assign this guy as a superhero
				max_odf(isample) = 1;
				i_max_odf(j)=isample;
				j=j+1;
				% prevent neighbours to be tested (for obvious computational reasons)
				dont_test = cat(1,dont_test,nearest_neighb{isample});
			end
 		end
	end
end
% length(i_max_odf)
% get the whole sphere
% max_odf(182:362) = zeros(181,1);
% max_odf = cat(1,max_odf,max_odf);

% reshape
% max_odf = max_odf';

% get maxima indices
% i_max_odf = find(max_odf);

% Get the number of maxima, assuming symmetry of the ODF. Also, if the
% number of maxima is odd, then rounds by the inferior number.
nb_maxima = floor(length(i_max_odf)/2);

switch output

case 'discreet'
	
	% divide the number of maxima by 2, due to the symmetry of the ODF
% 	nb_maxima = floor(length(find(max_odf))/2);
	
	% order the maxima
	[a b]=sort(odf(i_max_odf),'descend');
	sorted_maxima = i_max_odf(b);

case 'parametric'
	
	% find the antipod of the first direction by computing the dot
	% procuct between the first direction and the other ones to get the
	% angle. Two antipodal directions are assumed to form an angle of about
	% 180°, so the maximum angle between each direction serves as a basis
	% to select two antipodal directions.
	for i = 1:length(i_max_odf)
		for j = 1:length(i_max_odf)
%  			acos(vert(i_max_odf(i),:)*vert(i_max_odf(j),:)')
			angle_dir(i,j) = abs(acos(vert(i_max_odf(i),:)*vert(i_max_odf(j),:)'))*180/pi;
		end
	end
	% find the antipod for each line of the angular matrix. It only look at
	% half of the matrix since it is defined symmetrical. 
	antipod = {};
	[a max_angle] = max(angle_dir);
	for i = 1:nb_maxima
		antipod{i} = [i max_angle(i)];
	end
	% compute the mean for each maxima
	max_odf_param = [];
	for i = 1:nb_maxima
		% compute the mean vector
		vect1 = vert(i_max_odf(antipod{i}(1)),:);
		vect2 = -vert(i_max_odf(antipod{i}(2)),:);
		max_odf_param(i,:) = mean([vect1; vect2],1);
		% compute the mean value on the ODF for each each antipodal pair
		value1 = odf(i_max_odf(antipod{i}(1)));
		value2 = odf(i_max_odf(antipod{i}(2)));
		max_odf_val(i) = mean([value1 value2]);
	end
	% order ODF maxima based on the mean value of each antipodal pair
	[a index_sort_max] = sort(max_odf_val,'descend');
	for i = 1:nb_maxima
		max_odf_param_tmp(i,:) = max_odf_param(index_sort_max(i),:);
		% put the vectors into +Z plane
%  		if max_odf_param(i,3)<0, max_odf_param(i,:) = -max_odf_param(i,:); end
	end
	max_odf_param = max_odf_param_tmp;
	% outputs
	varargout{1} = max_odf_param;
	varargout{2} = nb_maxima;
end
