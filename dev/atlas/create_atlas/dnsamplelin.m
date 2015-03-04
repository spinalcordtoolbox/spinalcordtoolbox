function out = dnsamplelin(I,n)
% out = dnsamplelin(I,n)
% Downsample image I by factor n by computing mean value for each region

I = double(I);
I = m_normalize(I);

[nx,ny] = size(I);
nxd = ceil(nx/n);
nyd = ceil(ny/n);
out = zeros(nx,ny);

downsamplinggrid = false(size(I));
downsamplinggrid(1:n:end,1:n:end) = true;
ind = find(downsamplinggrid);
[indx,indy] = ind2sub(size(I),ind);

indx = reshape(indx,nxd,nyd);
indy = reshape(indy,nxd,nyd);
indx = indx(:,1); indx = indx(:);
indy = indy(1,:); indy = indy(:);
xf = indx(end);
yf = indy(end);
indx = indx(1:end-1)';
indy = indy(1:end-1)';


for i = indx
    for j = indy
        out(i,j) = mean( mean( I(i:i+n-1,j:j+n-1) ) );
    end
end

for i = indx
    out(i,yf) = mean( mean( I(i:i+n-1,yf:end) ) );
end

for j = indy
    out(xf,j) = mean( mean( I(xf:end,j:j+n-1) ) );
end

out(xf,yf) = mean( mean( I(xf:end,yf:end) ) );

out = downsample(out,n);
out = out';
out = downsample(out,n);
out = out';



end