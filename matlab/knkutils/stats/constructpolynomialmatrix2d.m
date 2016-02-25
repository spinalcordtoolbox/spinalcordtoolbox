function [f,str] = constructpolynomialmatrix2d(matrixsize,locs,degree)

% function [f,str] = constructpolynomialmatrix2d(matrixsize,locs,degree)
%
% <matrixsize> is a 2D matrix size like [100 50]
% <locs> is a row or column vector of indices into that matrix size
% <degree> is the maximum polynomial degree desired.
%   should be no greater than 10.
% 
% return <f>, a matrix of dimensions length(<locs>) x N
% with polynomial basis functions evaluated at <locs> in
% the columns.  the polynomial basis functions are evaluated
% over the range [-1,1] which is presumed to correspond to
% the beginning and ending element along each of the two dimensions.
% (if a dimension has only one element, the values are all set to 1.)
% also, return <str>, the algebraic expression that corresponds to
% the columns of <f>.  'x' refers to the first matrix dimension; 'y'
% refers to the second matrix dimension.
%
% note that there may be various gain factors on the basis functions
% (e.g. don't assume that they are all unit-length).
%
% also, beware of numerical precision issues for high degrees...
%
% see also constructpolynomialmatrix3d.m.
%
% history:
% - 2013/05/23 - hard code the basis expansion to ensure consistent results across platforms.
%                this changes previous behavior.
%
% example:
% [f,str] = constructpolynomialmatrix2d([30 30],find(ones(30,30)),2);
% str
% f = reshape(f,30,30,[]);
% figure; imagesc(makeimagestack(f));

% prep
x = sym('x');
y = sym('y');

% do the algebra
%str = char(expand((x+y+1)^degree));
switch degree
case 0
  str = '1';
case 1
  str = 'x+y+1';
case 2
  str = 'x^2+2*x*y+2*x+y^2+2*y+1';
case 3
  str = 'x^3+3*x^2*y+3*x^2+3*x*y^2+6*x*y+3*x+y^3+3*y^2+3*y+1';
case 4
  str = '1+4*x+4*y+6*x^2+12*x*y+6*y^2+4*x^3*y+6*x^2*y^2+12*x^2*y+4*x*y^3+12*x*y^2+x^4+4*x^3+y^4+4*y^3';
case 5
  str = '1+5*x+5*y+10*x^2+20*x*y+10*y^2+20*x^3*y+30*x^2*y^2+30*x^2*y+20*x*y^3+30*x*y^2+5*x^4+10*x^3+5*y^4+10*y^3+y^5+5*x*y^4+10*x^2*y^3+10*x^3*y^2+5*x^4*y+x^5';
case 6
  str = '1+6*x+6*y+15*x^2+30*x*y+15*y^2+60*x^3*y+90*x^2*y^2+60*x^2*y+60*x*y^3+60*x*y^2+15*x^4+20*x^3+15*y^4+20*y^3+6*x*y^5+y^6+6*y^5+30*x*y^4+15*x^2*y^4+60*x^2*y^3+60*x^3*y^2+20*x^3*y^3+15*x^4*y^2+30*x^4*y+6*x^5+x^6+6*x^5*y';
case 7
  str = '1+7*x+7*y+21*x^2+42*x*y+21*y^2+140*x^3*y+210*x^2*y^2+105*x^2*y+140*x*y^3+105*x*y^2+35*x^4+35*x^3+35*y^4+35*y^3+7*x*y^6+42*x*y^5+7*y^6+21*y^5+y^7+105*x*y^4+105*x^2*y^4+210*x^2*y^3+21*x^2*y^5+210*x^3*y^2+35*x^3*y^4+140*x^3*y^3+105*x^4*y^2+105*x^4*y+21*x^5+7*x^6+x^7+35*x^4*y^3+42*x^5*y+21*x^5*y^2+7*x^6*y';
case 8
  str = '1+8*x+8*y+28*x^2+56*x*y+28*y^2+280*x^3*y+420*x^2*y^2+168*x^2*y+280*x*y^3+168*x*y^2+70*x^4+56*x^3+70*y^4+56*y^3+56*x*y^6+168*x*y^5+8*x*y^7+28*y^6+56*y^5+8*y^7+y^8+280*x*y^4+28*x^2*y^6+420*x^2*y^4+560*x^2*y^3+168*x^2*y^5+560*x^3*y^2+280*x^3*y^4+560*x^3*y^3+56*x^3*y^5+420*x^4*y^2+280*x^4*y+56*x^5+28*x^6+8*x^7+x^8+70*x^4*y^4+280*x^4*y^3+168*x^5*y+168*x^5*y^2+56*x^5*y^3+56*x^6*y+28*x^6*y^2+8*x^7*y';
case 9
  str = '1+9*x+9*y+36*x^2+72*x*y+36*y^2+504*x^3*y+756*x^2*y^2+252*x^2*y+504*x*y^3+252*x*y^2+126*x^4+84*x^3+126*y^4+84*y^3+252*x*y^6+504*x*y^5+72*x*y^7+84*y^6+126*y^5+36*y^7+9*y^8+630*x*y^4+252*x^2*y^6+1260*x^2*y^4+1260*x^2*y^3+756*x^2*y^5+1260*x^3*y^2+1260*x^3*y^4+1680*x^3*y^3+504*x^3*y^5+1260*x^4*y^2+630*x^4*y+126*x^5+84*x^6+36*x^7+9*x^8+630*x^4*y^4+1260*x^4*y^3+504*x^5*y+756*x^5*y^2+504*x^5*y^3+252*x^6*y+252*x^6*y^2+72*x^7*y+x^9+y^9+9*x^8*y+36*x^7*y^2+84*x^6*y^3+126*x^5*y^4+126*x^4*y^5+84*x^3*y^6+36*x^2*y^7+9*x*y^8';
case 10
  str = '1+10*x+10*y+45*x^2+90*x*y+45*y^2+840*x^3*y+1260*x^2*y^2+360*x^2*y+840*x*y^3+360*x*y^2+210*x^4+120*x^3+210*y^4+120*y^3+840*x*y^6+1260*x*y^5+360*x*y^7+210*y^6+252*y^5+120*y^7+45*y^8+1260*x*y^4+1260*x^2*y^6+3150*x^2*y^4+2520*x^2*y^3+2520*x^2*y^5+2520*x^3*y^2+4200*x^3*y^4+4200*x^3*y^3+2520*x^3*y^5+3150*x^4*y^2+1260*x^4*y+252*x^5+210*x^6+120*x^7+45*x^8+3150*x^4*y^4+4200*x^4*y^3+1260*x^5*y+2520*x^5*y^2+2520*x^5*y^3+840*x^6*y+1260*x^6*y^2+360*x^7*y+10*x^9+10*y^9+x^10+90*x^8*y+360*x^7*y^2+840*x^6*y^3+1260*x^5*y^4+1260*x^4*y^5+840*x^3*y^6+360*x^2*y^7+90*x*y^8+10*x^9*y+45*x^8*y^2+y^10+120*x^7*y^3+210*x^6*y^4+252*x^5*y^5+210*x^4*y^6+120*x^3*y^7+45*x^2*y^8+10*x*y^9';
otherwise
  die;
end

%REMOVE since hard-coded now
%str = sort(strsplit(str,'+'));%% sort the stuff in between + signs to try to ensure consistent ordering!!!
str = strsplit(str,'+');
str = cat(1,str,repmat({'+'},[1 length(str)]));
str = cat(2,str{:});
str = str(1:end-1);

% add a little padding so the 1 step below will work for degree 0
str = [' ' str ' '];

% change * to .*
  old = pwd;  % THIS IS A TOTAL HACK TO AVOID FUNCTION NAME CONFLICTS!
  cd(fullfile(matlabroot,'toolbox','matlab','funfun'));
str = vectorize(str);
  cd(old);

% remove +
str(str=='+') = ' ';

% change 1 to ones(size(x),1)
str0 = strrep(str,' 1 ',' ones(size(x)) '); assert(length(str0) ~= length(str));
str = str0;

% add brackets
str = [ '[' str ']' ];

% prep the linear coordinates
[x,y] = ind2sub(matrixsize,locs(:));
if matrixsize(1)~=1
  x = normalizerange(x,-1,1,1,matrixsize(1));
end
if matrixsize(2)~=1
  y = normalizerange(y,-1,1,1,matrixsize(2));
end

% do it
f = eval(str);
