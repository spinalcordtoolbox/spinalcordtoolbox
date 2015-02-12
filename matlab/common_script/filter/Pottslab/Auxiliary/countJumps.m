function j = countJumps( f )
%COUNTJUMPS Counts the jumps of a vector f
j = sum( diff(f) ~= 0);

end

