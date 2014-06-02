% =========================================================================
% FUNCTION
% j_transpose
%
% Compute the symmetic of a discrete function with the bissectrisse y=x.
% Works with a table lookup.
%
% INPUT
% data_in			1xn float.
% (interpolate)		integer. Interpolate for more accuracy (default = 1).
% 
% OUTPUT
%
% COMMENTS
% Julien Cohen-Adad 2008-03-05
% =========================================================================
function data_out = j_transpose(data_in,interpolate)

if nargin<1, help j_transpose; return; end
if nargin<2, interpolate = 2; end


% interpolate data_in
data_in = interp(data_in,interpolate);
n = length(data_in);
data_in = round(data_in/max(data_in)*n);

% table lookup
for i=1:n
	if (data_in(i)==0)
		data_out(i) = 0;
	else
		data_out(data_in(i)) = i;
	end
end

% replace zeros by previous value
for i=1:n-1
	if (data_out(i+1)==0)
		data_out(i+1) = data_out(i);
	end
end

% correct edges
index_min = min(find(data_out~=0));
data_out(1:index_min-1) = data_out(index_min);
index_max = max(find(data_out~=n));
data_out(index_max+1:n) = data_out(index_max);


% figure, plot(data_in,'*');
% grid, hold on, axis square
% x = (1:n);
% plot(x,'k*')
% plot(data_out,'*r')	
