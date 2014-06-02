function h_plot = j_errorbar(x,y,sd,varargin)
% =========================================================================
% FUNCTION
% j_errorbar
%
% Replace the Matlab errorbar function which has horrible bugs I could not
% handle any more.
%
% INPUTS
% x					integer. x location.
% y					integer. y location.
% sd				1x2 integer. standard deviation.
% (color)			string. Example: 'k'
% (width_sd)		integer.
% (sd_sampling)		integer. Display error bars every 'sd_sampling' sample
%
% OUTPUTS
% (-)
%
% COMMENTS
% Julien Cohen-Adad
% 2008-03-01: Created
% 2011-11-25: Modifs for SD
% =========================================================================



% default initialization
color_plot		= '';
marker_size		= 5;
line_width		= 0.5;
line_width_sd	= 1.5;
index_nargin	= 3;
sd_sampling		= 1;


if ~isempty(varargin)
	color_plot = varargin{1};
end

for i=1:length(varargin)
	% check field using case insensitive function
	if isstr(varargin{i})
		if regexpi(varargin{i},'MarkerSize');
			marker_size = varargin{i+1};
		elseif regexpi(varargin{i},'LineWidth');
			line_width = varargin{i+1};
		elseif regexpi(varargin{i},'width_sd');
			line_width_sd = varargin{i+1};
		elseif regexpi(varargin{i},'sd_sampling');
			sd_sampling = varargin{i+1};
		end
	end
end

% display y
h_plot = plot(x,y,color_plot,'MarkerSize',marker_size,'LineWidth',line_width);
struct_plot = get(h_plot);

% check if sd is non-symmetrical
hold on
if length(sd)==1
	sd(2)=sd;
end

% display sd in both directions (up and down)
sd = sd';
line([x(1,1:sd_sampling:end);x(1,1:sd_sampling:end)],[y(1,1:sd_sampling:end)+sd(1,1:sd_sampling:end);y(1,1:sd_sampling:end)],'LineWidth',line_width_sd,'Color',struct_plot.Color);
line([x(1,1:sd_sampling:end);x(1,1:sd_sampling:end)],[y(1,1:sd_sampling:end)-sd(2,1:sd_sampling:end);y(1,1:sd_sampling:end)],'LineWidth',line_width_sd,'Color',struct_plot.Color);

