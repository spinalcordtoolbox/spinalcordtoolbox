% =========================================================================
% FUNCTION
% j_cell2array.m
%
% Convert cell to array (fill by blanks)
%
% INPUT
% cell_in           cell
% 
% OUTPUT
% array_out         array
%
% COMMENTS
% julien cohen-adad 2007-01-29
% =========================================================================
function array_out = j_cell2array(cell_in)


% initialization


% find size cell
size_cell = size(cell_in,2);

% find file max chars
max_chars=0;
for i=1:size_cell
    if (max_chars<size(cell_in{i},2))
        max_chars = size(cell_in{i},2);
    end
end

% transform cell into char array (fill with blanks when necessary)
array_out = char(zeros(size_cell,max_chars));
for i=1:size_cell
    array_out(i,:) = [cell_in{i},blanks(max_chars-size(cell_in{i},2))];
end

