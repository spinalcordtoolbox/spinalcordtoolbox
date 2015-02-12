% =========================================================================
% FUNCTION
% j_numbering.m
%
% Create a cell containing ascending numbers (e.g. '001', '002', ...)
%
% INPUTS
% max_numbering     integer.
% (nb_digits)       integer. By default, it ajusts it automatically
% (starting_value)  integer. starting value (default=1)
% (output_format)   string. 'cell'*, 'array'
%
% OUTPUTS
% out_numbering     cell or array
%
% DEPENDANCES
% (-)
%
% COMMENTS
% Julien Cohen-Adad 2006-11-26
% =========================================================================
function varargout = m_numbering(varargin)


% initialization
if (nargin<1) help m_numbering; return; end
max_numbering = varargin{1};
if (nargin<2)
    nb_digits = length(num2str(max_numbering));
else
    nb_digits = varargin{2};
    % check number of digits
    if (nb_digits<length(num2str(max_numbering)))
        error('Number of digits too small!!!');
        return
    end
end
if (nargin<3)
    starting_value = 1;
else
    starting_value = varargin{3};
end
if (nargin<4)
    output_format = 'cell';
else
    output_format = varargin{4};
end

% generate numbering
out_numbering = cell(max_numbering,1);
number = starting_value;
for iNumber=1:max_numbering
    % write number
    number_string = num2str(number);
    % fill with zeros
    for iDigits=1:nb_digits-length(number_string)
        number_string = strcat('0',number_string);
    end
    out_numbering{iNumber} = number_string;
    number = number + 1;
end

if strcmp(output_format,'array')
    out_numbering_tmp = out_numbering;
    clear out_numbering
    for i=1:size(out_numbering_tmp,1)
		out_numbering(i,:) = out_numbering_tmp{i};
    end
	clear out_numbering_tmp
end

% output
varargout{1} = out_numbering;
