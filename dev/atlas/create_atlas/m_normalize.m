function out = m_normalize(X)
% Normalizes input array values [0,1]
% The class of the output is double
% If all values in input are equal, they are set to zero

X = double(X);

if( max(X(:)) == min(X(:)) )
    out = zeros(size(X));
else
    out = ( X - min(X(:)) ) / ( max(X(:)) - min(X(:)) );
end


end

