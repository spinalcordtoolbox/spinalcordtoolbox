% =========================================================================
% FUNCTION
% j_mask
%
% Mask data based on intensity threshold retrived using Otsu algorithm.
%
% INPUT
% data              double.
%
% OUTPUT
% (-)
%
% DEPENDENCES
%
% COMMENTS
% Otsu implementation based on st_segment_brain (Perlbarg/Bellec).
% Julien Cohen-Adad 2007-07-17
% =========================================================================
function [mask threshold] = j_mask(data)


% default initialization
if (nargin<1), help j_otsu; return; end

% smooth with gaussian filter
size_se = 3;
type_se = 'disk';
data_s = j_morpho(data,'smooth',size_se,type_se);

% compute histogram
[Y,X] = hist(data_s(:),256);

Y = Y/sum(Y);
ngr = length(Y);
somme = sum((1:(ngr)).*Y);

eps = 10^(-10);
index_seuil = 0;


smax = 0;
p = 0;
a = 0;
for i = 1:(ngr-1)
    a = a+(i)*Y(i);
    p = p+Y(i);
    s = somme*p-a;
    d = p*(1-p);
    if (d>=eps)
        s = s*s/d;        
        if (s >= smax)
            smax = s;
            amax = a;
            pmax = p;
            index_seuil = i;
        end
    end
end

index_seuil = index_seuil-1;

threshold = X(index_seuil);
mask = data>threshold;

% data_m = mean(abs(data),4);
% 
% nx=size(data_m,1);
% ny=size(data_m,2);
% nz=size(data_m,3);
% 
% filtre_gaussien=[9.541757E-6  4.53885E-5 1.526681E-4 3.63108E-4 6.106724E-4 7.26216E-4 6.106724E-4 3.63108E-4 1.526681E-4  4.53885E-5 9.541757E-6
% 4.53885E-5 2.159053E-4  7.26216E-4 .001727243  .002904864 .003454485  .002904864 .001727243  7.26216E-4 2.159053E-4  4.53885E-5
% 1.526681E-4  7.26216E-4   .00244269 .005809728  .009770759  .01161946  .009770759 .005809728   .00244269  7.26216E-4 1.526681E-4
% 3.63108E-4  .001727243  .005809728  .01381794   .02323891  .02763588   .02323891  .01381794  .005809728  .001727243  3.63108E-4
% 6.106724E-4  .002904864  .009770759  .02323891   .03908303  .04647782   .03908303  .02323891  .009770759  .002904864 6.106724E-4
% 7.26216E-4  .003454485   .01161946  .02763588   .04647782  .05527176   .04647782  .02763588   .01161946  .003454485  7.26216E-4
% 6.106724E-4  .002904864  .009770759  .02323891   .03908303  .04647782   .03908303  .02323891  .009770759  .002904864 6.106724E-4
% 3.63108E-4  .001727243  .005809728  .01381794   .02323891  .02763588   .02323891  .01381794  .005809728  .001727243  3.63108E-4
% 1.526681E-4  7.26216E-4   .00244269 .005809728  .009770759  .01161946  .009770759 .005809728   .00244269  7.26216E-4 1.526681E-4
% 4.53885E-5 2.159053E-4  7.26216E-4 .001727243  .002904864 .003454485  .002904864 .001727243  7.26216E-4 2.159053E-4  4.53885E-5
% 9.541757E-6  4.53885E-5 1.526681E-4 3.63108E-4 6.106724E-4 7.26216E-4 6.106724E-4 3.63108E-4 1.526681E-4  4.53885E-5 9.541757E-6];
% 
% for num_coupe=1:nz
%     data_m(:,:,num_coupe)=conv2(data_m(:,:,num_coupe),filtre_gaussien,'same');
% end
% 
% 
% 
