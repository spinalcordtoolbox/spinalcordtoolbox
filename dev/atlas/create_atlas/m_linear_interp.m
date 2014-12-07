function out = m_linear_interp(X,p,q)
% out = m_linear_interp(X,p,q)
% Linear interpolation for a 3D data along z (3rd) direction
% The input data is interpolated between z = n and z = m

X = double(X);
out = X;
n = min(p,q);
m = max(p,q);
delta = m - n;

if (delta <= 1)
    out = X;
else
    for k = 1:delta-1
        out(:,:,n+k) = (delta-k)/delta * X(:,:,n) + k/delta * X(:,:,m);
    end
end


end