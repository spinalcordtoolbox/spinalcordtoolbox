function [f,str] = constructpolynomialmatrix3d(matrixsize,locs,degree,weights)

% function [f,str] = constructpolynomialmatrix3d(matrixsize,locs,degree,weights)
%
% <matrixsize> is a 3D matrix size like [100 50 100]
% <locs> is a row or column vector of indices into that matrix size
% <degree> is the maximum polynomial degree desired.  should be no greater than 10.
% <weights> (optional) is a 1 x N vector of values. 
%   if supplied, we automatically weight and sum the basis functions.
%   the point of this input is to avoid having to explicitly create
%   all the basis functions (and therefore we save on memory requirements).
%   default is [] which means do nothing special.
% 
% if <weights> is not supplied, then:
%   return <f>, a matrix of dimensions length(<locs>) x N
%   with polynomial basis functions evaluated at <locs> in
%   the columns.  the polynomial basis functions are evaluated
%   over the range [-1,1] which is presumed to correspond to
%   the beginning and ending element along each of the three dimensions.
%   (if a dimension has only one element, the values are all set to 1.)
%   also, return <str>, the algebraic expression that corresponds to
%   the columns of <f>.  'x' refers to the first matrix dimension; 'y'
%   refers to the second matrix dimension; 'z' refers to the third 
%   matrix dimension.
%
% if <weights> is supplied, then:
%   return <f>, a vector of dimensions length(<locs>) x 1 with the weighted
%   sum of the polynomial basis functions.  also, return <str>, a cell 
%   vector of algebraic expressions describing the various basis functions.
%
% note that there may be various gain factors on the basis functions
% (e.g. don't assume that they are all unit-length).
%
% also, beware of numerical precision issues for high degrees...
%
% see also constructpolynomialmatrix2d.m.
%
% history:
% - 2013/05/23 - hard code the basis expansion to ensure consistent results across platforms.
%                this changes previous behavior.
%
% example:
% [f,str] = constructpolynomialmatrix3d([30 30 30],find(ones(30,30,30)),2);
% str
% f = reshape(f,30,30,30,[]);
% for p=1:size(f,4)
%   drawnow; figure; imagesc(makeimagestack(f(:,:,:,p)));
% end

% input
if ~exist('weights','var') || isempty(weights)
  weights = [];
end

% prep
x = sym('x');
y = sym('y');
z = sym('z');

% do the algebra
%str = char(expand((x+y+z+1)^degree));
switch degree
case 0
  str = '1';
case 1
  str = 'x+y+z+1';
case 2
  str = 'x^2+2*x*y+2*x*z+2*x+y^2+2*y*z+2*y+z^2+2*z+1';
case 3
  str = '1+3*x+3*y+3*z+6*x*y*z+3*x^2+6*x*y+6*x*z+3*y^2+6*y*z+3*z^2+3*x^2*y+3*x^2*z+3*x*y^2+3*x*z^2+x^3+y^3+z^3+3*y^2*z+3*y*z^2';
case 4
  str = '1+4*x+4*y+4*z+24*x*y*z+6*x^2+12*x*y+12*x*z+6*y^2+12*y*z+6*z^2+4*x^3*y+4*x^3*z+6*x^2*y^2+12*x^2*y+6*x^2*z^2+12*x^2*z+4*x*y^3+12*x*y^2+4*x*z^3+12*x*z^2+12*x^2*y*z+12*x*y^2*z+12*x*y*z^2+x^4+4*x^3+y^4+4*y^3+z^4+4*z^3+4*y^3*z+6*y^2*z^2+12*y^2*z+4*y*z^3+12*y*z^2';
case 5
  str = '1+5*x+5*y+5*z+20*x*y^3*z+30*x*y^2*z^2+60*x*y*z+10*x^2+20*x*y+20*x*z+10*y^2+20*y*z+10*z^2+20*x^3*y+20*x^3*z+30*x^2*y^2+30*x^2*y+30*x^2*z^2+30*x^2*z+20*x*y^3+30*x*y^2+20*x*z^3+30*x*z^2+60*x^2*y*z+60*x*y^2*z+60*x*y*z^2+5*x^4+10*x^3+5*y^4+10*y^3+5*z^4+10*z^3+20*y^3*z+30*y^2*z^2+30*y^2*z+20*y*z^3+30*y*z^2+20*x*y*z^3+5*y^4*z+10*y^3*z^2+10*y^2*z^3+5*y*z^4+y^5+z^5+5*x*y^4+5*x*z^4+30*x^2*y^2*z+30*x^2*y*z^2+10*x^2*y^3+10*x^2*z^3+20*x^3*y*z+10*x^3*y^2+10*x^3*z^2+5*x^4*z+5*x^4*y+x^5';
case 6
  str = '1+6*x+6*y+6*z+30*x^4*y*z+120*x*y^3*z+180*x*y^2*z^2+120*x*y*z+15*x^2+30*x*y+30*x*z+15*y^2+30*y*z+15*z^2+60*x^3*y+60*x^3*z+90*x^2*y^2+60*x^2*y+90*x^2*z^2+60*x^2*z+60*x*y^3+60*x*y^2+60*x*z^3+60*x*z^2+180*x^2*y*z+180*x*y^2*z+180*x*y*z^2+15*x^4+20*x^3+15*y^4+20*y^3+15*z^4+20*z^3+60*y^3*z+90*y^2*z^2+60*y^2*z+60*y*z^3+60*y*z^2+120*x*y*z^3+30*x*y^4*z+60*x*y^3*z^2+60*x*y^2*z^3+30*x*y*z^4+6*y^5*z+30*y^4*z+60*y^3*z^2+20*y^3*z^3+15*y^4*z^2+60*y^2*z^3+30*y*z^4+6*x*y^5+6*x*z^5+y^6+6*y^5+6*z^5+z^6+15*y^2*z^4+6*y*z^5+30*x*y^4+30*x*z^4+60*x^2*y^3*z+90*x^2*y^2*z^2+180*x^2*y^2*z+60*x^2*y*z^3+180*x^2*y*z^2+15*x^2*y^4+60*x^2*y^3+15*x^2*z^4+60*x^2*z^3+120*x^3*y*z+60*x^3*y^2*z+60*x^3*y*z^2+60*x^3*y^2+60*x^3*z^2+20*x^3*y^3+20*x^3*z^3+30*x^4*z+15*x^4*y^2+15*x^4*z^2+30*x^4*y+6*x^5+x^6+6*x^5*y+6*x^5*z';
case 7
  str = '1+7*x+7*y+7*z+210*x^4*y*z+420*x*y^3*z+42*x*y^5*z+105*x*y^2*z^4+630*x*y^2*z^2+210*x*y*z+21*x^2+42*x*y+42*x*z+21*y^2+42*y*z+21*z^2+140*x^3*y+140*x^3*z+210*x^2*y^2+105*x^2*y+210*x^2*z^2+105*x^2*z+140*x*y^3+105*x*y^2+140*x*z^3+105*x*z^2+420*x^2*y*z+420*x*y^2*z+420*x*y*z^2+35*x^4+35*x^3+35*y^4+35*y^3+35*z^4+35*z^3+140*y^3*z+210*y^2*z^2+105*y^2*z+140*y*z^3+105*y*z^2+420*x*y*z^3+210*x*y^4*z+420*x*y^3*z^2+140*x*y^3*z^3+105*x*y^4*z^2+420*x*y^2*z^3+210*x*y*z^4+42*x*y*z^5+21*y^5*z^2+42*y^5*z+35*y^4*z^3+105*y^4*z+210*y^3*z^2+35*y^3*z^4+140*y^3*z^3+105*y^4*z^2+210*y^2*z^3+105*y*z^4+7*y^6*z+7*x*y^6+42*x*y^5+42*x*z^5+7*x*z^6+7*y^6+21*y^5+y^7+21*z^5+7*z^6+z^7+105*y^2*z^4+21*y^2*z^5+42*y*z^5+7*y*z^6+105*x*y^4+105*x*z^4+420*x^2*y^3*z+630*x^2*y^2*z^2+210*x^2*y^3*z^2+210*x^2*y^2*z^3+105*x^2*y*z^4+105*x^2*y^4*z+630*x^2*y^2*z+420*x^2*y*z^3+630*x^2*y*z^2+105*x^2*y^4+210*x^2*y^3+105*x^2*z^4+210*x^2*z^3+21*x^2*y^5+21*x^2*z^5+420*x^3*y*z+140*x^3*y^3*z+210*x^3*y^2*z^2+420*x^3*y^2*z+140*x^3*y*z^3+420*x^3*y*z^2+105*x^4*y^2*z+105*x^4*y*z^2+42*x^5*y*z+210*x^3*y^2+210*x^3*z^2+35*x^3*y^4+140*x^3*y^3+35*x^3*z^4+140*x^3*z^3+105*x^4*z+105*x^4*y^2+105*x^4*z^2+105*x^4*y+21*x^5+7*x^6+x^7+35*x^4*y^3+35*x^4*z^3+42*x^5*y+42*x^5*z+21*x^5*y^2+21*x^5*z^2+7*x^6*y+7*x^6*z';
case 8
  str = '1+8*x+8*y+56*x*y*z^6+8*z+840*x^4*y*z+1120*x*y^3*z+336*x*y^5*z+280*x^4*y^3*z+840*x*y^2*z^4+56*x^6*y*z+1680*x*y^2*z^2+336*x*y*z+28*x^2+56*x*y+56*x*z+28*y^2+56*y*z+28*z^2+280*x^3*y+280*x^3*z+420*x^2*y^2+168*x^2*y+420*x^2*z^2+168*x^2*z+280*x*y^3+168*x*y^2+280*x*z^3+168*x*z^2+840*x^2*y*z+840*x*y^2*z+840*x*y*z^2+70*x^4+56*x^3+70*y^4+56*y^3+70*z^4+56*z^3+280*y^3*z+420*y^2*z^2+168*y^2*z+280*y*z^3+168*y*z^2+1120*x*y*z^3+168*x*y^5*z^2+280*x*y^4*z^3+840*x*y^4*z+1680*x*y^3*z^2+280*x*y^3*z^4+1120*x*y^3*z^3+840*x*y^4*z^2+1680*x*y^2*z^3+840*x*y*z^4+56*x*y^6*z+168*x*y^2*z^5+336*x*y*z^5+168*y^5*z^2+168*y^5*z+280*y^4*z^3+280*y^4*z+560*y^3*z^2+280*y^3*z^4+560*y^3*z^3+420*y^4*z^2+560*y^2*z^3+280*y*z^4+28*y^6*z^2+56*y^5*z^3+70*y^4*z^4+56*y^3*z^5+8*y^7*z+56*y^6*z+56*x*y^6+168*x*y^5+8*x*y^7+168*x*z^5+56*x*z^6+8*x*z^7+28*y^6+56*y^5+8*y^7+56*z^5+28*z^6+z^8+8*z^7+y^8+420*y^2*z^4+28*y^2*z^6+168*y^2*z^5+168*y*z^5+8*y*z^7+56*y*z^6+280*x*y^4+280*x*z^4+1680*x^2*y^3*z+2520*x^2*y^2*z^2+168*x^2*y^5*z+1680*x^2*y^3*z^2+560*x^2*y^3*z^3+420*x^2*y^4*z^2+1680*x^2*y^2*z^3+840*x^2*y*z^4+840*x^2*y^4*z+1680*x^2*y^2*z+1680*x^2*y*z^3+1680*x^2*y*z^2+28*x^2*y^6+420*x^2*y^4+560*x^2*y^3+420*x^2*z^4+560*x^2*z^3+168*x^2*y^5+168*x^2*z^5+28*x^2*z^6+420*x^2*y^2*z^4+168*x^2*y*z^5+1120*x^3*y*z+1120*x^3*y^3*z+1680*x^3*y^2*z^2+1680*x^3*y^2*z+1120*x^3*y*z^3+1680*x^3*y*z^2+280*x^3*y^4*z+560*x^3*y^3*z^2+560*x^3*y^2*z^3+280*x^3*y*z^4+420*x^4*y^2*z^2+840*x^4*y^2*z+280*x^4*y*z^3+840*x^4*y*z^2+168*x^5*y*z^2+336*x^5*y*z+168*x^5*y^2*z+560*x^3*y^2+560*x^3*z^2+280*x^3*y^4+560*x^3*y^3+280*x^3*z^4+560*x^3*z^3+56*x^3*y^5+56*x^3*z^5+280*x^4*z+420*x^4*y^2+420*x^4*z^2+280*x^4*y+56*x^5+28*x^6+8*x^7+x^8+70*x^4*y^4+280*x^4*y^3+70*x^4*z^4+280*x^4*z^3+168*x^5*y+168*x^5*z+168*x^5*y^2+56*x^5*z^3+168*x^5*z^2+56*x^5*y^3+56*x^6*y+28*x^6*z^2+56*x^6*z+28*x^6*y^2+8*x^7*z+8*x^7*y';
case 9
  str = '1+72*z*x^7*y+9*x+9*y+504*x*y*z^6+9*z+2520*x^4*y*z+2520*x*y^3*z+1512*x*y^5*z+72*z^7*x*y+2520*x^4*y^3*z+3780*x*y^2*z^4+504*x^6*y*z+3780*x*y^2*z^2+504*x*y*z+36*x^2+72*x*y+72*x*z+36*y^2+72*y*z+36*z^2+504*x^3*y+504*x^3*z+756*x^2*y^2+252*x^2*y+756*x^2*z^2+252*x^2*z+504*x*y^3+252*x*y^2+504*x*z^3+252*x*z^2+1512*x^2*y*z+1512*x*y^2*z+1512*x*y*z^2+126*x^4+84*x^3+126*y^4+84*y^3+126*z^4+84*z^3+504*y^3*z+756*y^2*z^2+252*y^2*z+504*y*z^3+252*y*z^2+2520*x*y*z^3+1512*x*y^5*z^2+2520*x*y^4*z^3+2520*x*y^4*z+5040*x*y^3*z^2+2520*x*y^3*z^4+5040*x*y^3*z^3+3780*x*y^4*z^2+5040*x*y^2*z^3+2520*x*y*z^4+504*x*y^6*z+1512*x*y^2*z^5+1512*x*y*z^5+756*y^5*z^2+504*y^5*z+1260*y^4*z^3+630*y^4*z+1260*y^3*z^2+1260*y^3*z^4+1680*y^3*z^3+1260*y^4*z^2+1260*y^2*z^3+630*y*z^4+252*y^6*z^2+504*y^5*z^3+630*y^4*z^4+504*y^3*z^5+72*y^7*z+252*y^6*z+252*x*y^6+504*x*y^5+72*x*y^7+504*x*z^5+252*x*z^6+72*x*z^7+84*y^6+126*y^5+36*y^7+126*z^5+84*z^6+9*z^8+36*z^7+9*y^8+1260*y^2*z^4+252*y^2*z^6+756*y^2*z^5+504*y*z^5+72*y*z^7+252*y*z^6+630*x*y^4+630*x*z^4+5040*x^2*y^3*z+7560*x^2*y^2*z^2+1512*x^2*y^5*z+7560*x^2*y^3*z^2+5040*x^2*y^3*z^3+3780*x^2*y^4*z^2+7560*x^2*y^2*z^3+3780*x^2*y*z^4+3780*x^2*y^4*z+3780*x^2*y^2*z+5040*x^2*y*z^3+3780*x^2*y*z^2+252*x^2*y^6+1260*x^2*y^4+1260*x^2*y^3+1260*x^2*z^4+1260*x^2*z^3+756*x^2*y^5+756*x^2*z^5+252*x^2*z^6+3780*x^2*y^2*z^4+1512*x^2*y*z^5+2520*x^3*y*z+5040*x^3*y^3*z+7560*x^3*y^2*z^2+5040*x^3*y^2*z+5040*x^3*y*z^3+5040*x^3*y*z^2+2520*x^3*y^4*z+5040*x^3*y^3*z^2+5040*x^3*y^2*z^3+2520*x^3*y*z^4+3780*x^4*y^2*z^2+3780*x^4*y^2*z+2520*x^4*y*z^3+3780*x^4*y*z^2+1512*x^5*y*z^2+1512*x^5*y*z+1512*x^5*y^2*z+1260*x^3*y^2+1260*x^3*z^2+1260*x^3*y^4+1680*x^3*y^3+1260*x^3*z^4+1680*x^3*z^3+504*x^3*y^5+504*x^3*z^5+630*x^4*z+1260*x^4*y^2+1260*x^4*z^2+630*x^4*y+126*x^5+84*x^6+36*x^7+9*x^8+630*x^4*y^4+1260*x^4*y^3+630*x^4*z^4+1260*x^4*z^3+504*x^5*y+504*x^5*z+756*x^5*y^2+504*x^5*z^3+756*x^5*z^2+504*x^5*y^3+252*x^6*y+252*x^6*z^2+252*x^6*z+252*x^6*y^2+72*x^7*z+72*x^7*y+x^9+y^9+9*x^8*y+36*x^7*y^2+84*x^6*y^3+126*x^5*y^4+126*x^4*y^5+84*x^3*y^6+36*x^2*y^7+9*x*y^8+9*z*x^8+9*z*y^8+36*z^2*x^7+36*z^2*y^7+84*z^3*x^6+84*z^3*y^6+252*z*x^6*y^2+504*z*x^5*y^3+630*z*x^4*y^4+504*z*x^3*y^5+252*z*x^2*y^6+72*z*x*y^7+252*z^2*x^6*y+756*z^2*x^5*y^2+1260*z^2*x^4*y^3+1260*z^2*x^3*y^4+756*z^2*x^2*y^5+252*z^2*x*y^6+504*z^3*x^5*y+1260*z^3*x^4*y^2+1680*z^3*x^3*y^3+1260*z^3*x^2*y^4+504*z^3*x*y^5+630*z^4*x^4*y+1260*z^4*x^3*y^2+1260*z^4*x^2*y^3+630*z^4*x*y^4+126*z^4*x^5+126*z^4*y^5+126*z^5*x^4+126*z^5*y^4+84*z^6*x^3+84*z^6*y^3+36*z^7*x^2+36*z^7*y^2+504*z^5*x^3*y+756*z^5*x^2*y^2+504*z^5*x*y^3+252*z^6*x^2*y+252*z^6*x*y^2+z^9+9*z^8*x+9*z^8*y';
case 10
  str = '1+720*z*x^7*y+10*x+10*y+2520*x*y*z^6+1260*z*x^4*y^5+10*z+6300*x^4*y*z+5040*x*y^3*z+5040*x*y^5*z+720*z^7*x*y+360*z^7*x^2*y+1260*z^4*x^5*y+12600*x^4*y^3*z+12600*x*y^2*z^4+90*z*x^8*y+2520*x^6*y*z+7560*x*y^2*z^2+360*z*x^7*y^2+720*x*y*z+1260*z^5*x^4*y+45*x^2+90*x*y+90*x*z+45*y^2+90*y*z+45*z^2+840*x^3*y+840*x^3*z+1260*x^2*y^2+360*x^2*y+1260*x^2*z^2+360*x^2*z+840*x*y^3+360*x*y^2+840*x*z^3+360*x*z^2+2520*x^2*y*z+2520*x*y^2*z+2520*x*y*z^2+210*x^4+120*x^3+210*y^4+120*y^3+210*z^4+120*z^3+840*y^3*z+1260*y^2*z^2+360*y^2*z+840*y*z^3+360*y*z^2+5040*x*y*z^3+7560*x*y^5*z^2+12600*x*y^4*z^3+6300*x*y^4*z+12600*x*y^3*z^2+12600*x*y^3*z^4+16800*x*y^3*z^3+12600*x*y^4*z^2+12600*x*y^2*z^3+6300*x*y*z^4+2520*x*y^6*z+7560*x*y^2*z^5+5040*x*y*z^5+2520*y^5*z^2+1260*y^5*z+4200*y^4*z^3+1260*y^4*z+2520*y^3*z^2+4200*y^3*z^4+4200*y^3*z^3+3150*y^4*z^2+2520*y^2*z^3+1260*y*z^4+1260*y^6*z^2+2520*y^5*z^3+3150*y^4*z^4+2520*y^3*z^5+360*y^7*z+840*y^6*z+840*x*y^6+1260*x*y^5+360*x*y^7+1260*x*z^5+840*x*z^6+360*x*z^7+210*y^6+252*y^5+120*y^7+252*z^5+210*z^6+45*z^8+120*z^7+45*y^8+3150*y^2*z^4+1260*y^2*z^6+2520*y^2*z^5+1260*y*z^5+360*y*z^7+840*y*z^6+1260*x*y^4+1260*x*z^4+12600*x^2*y^3*z+18900*x^2*y^2*z^2+7560*x^2*y^5*z+25200*x^2*y^3*z^2+25200*x^2*y^3*z^3+18900*x^2*y^4*z^2+25200*x^2*y^2*z^3+12600*x^2*y*z^4+12600*x^2*y^4*z+7560*x^2*y^2*z+12600*x^2*y*z^3+7560*x^2*y*z^2+1260*x^2*y^6+3150*x^2*y^4+2520*x^2*y^3+3150*x^2*z^4+2520*x^2*z^3+2520*x^2*y^5+2520*x^2*z^5+1260*x^2*z^6+18900*x^2*y^2*z^4+7560*x^2*y*z^5+5040*x^3*y*z+16800*x^3*y^3*z+25200*x^3*y^2*z^2+12600*x^3*y^2*z+16800*x^3*y*z^3+12600*x^3*y*z^2+12600*x^3*y^4*z+25200*x^3*y^3*z^2+25200*x^3*y^2*z^3+12600*x^3*y*z^4+18900*x^4*y^2*z^2+12600*x^4*y^2*z+12600*x^4*y*z^3+12600*x^4*y*z^2+7560*x^5*y*z^2+5040*x^5*y*z+7560*x^5*y^2*z+840*z*x^6*y^3+1260*z*x^5*y^4+840*z*x^3*y^6+360*z*x^2*y^7+90*z*x*y^8+360*z^2*x^7*y+1260*z^2*x^6*y^2+2520*z^2*x^5*y^3+3150*z^2*x^4*y^4+2520*z^2*x^3*y^5+2520*x^3*y^2+2520*x^3*z^2+4200*x^3*y^4+4200*x^3*y^3+4200*x^3*z^4+4200*x^3*z^3+2520*x^3*y^5+2520*x^3*z^5+1260*x^4*z+3150*x^4*y^2+3150*x^4*z^2+1260*x^4*y+252*x^5+210*x^6+120*x^7+45*x^8+3150*x^4*y^4+4200*x^4*y^3+3150*x^4*z^4+4200*x^4*z^3+1260*x^5*y+1260*x^5*z+2520*x^5*y^2+2520*x^5*z^3+2520*x^5*z^2+2520*x^5*y^3+840*x^6*y+1260*x^6*z^2+840*x^6*z+1260*x^6*y^2+360*x^7*z+360*x^7*y+10*x^9+10*y^9+x^10+90*x^8*y+360*x^7*y^2+840*x^6*y^3+1260*x^5*y^4+1260*x^4*y^5+840*x^3*y^6+360*x^2*y^7+90*x*y^8+10*x^9*y+45*x^8*y^2+y^10+120*x^7*y^3+210*x^6*y^4+252*x^5*y^5+210*x^4*y^6+120*x^3*y^7+45*x^2*y^8+10*x*y^9+10*z*x^9+10*z*y^9+45*z^2*x^8+45*z^2*y^8+90*z*x^8+90*z*y^8+120*z^3*x^7+120*z^3*y^7+360*z^2*x^7+360*z^2*y^7+210*z^4*x^6+210*z^4*y^6+840*z^3*x^6+840*z^3*y^6+1260*z^2*x^2*y^6+360*z^2*x*y^7+2520*z*x^6*y^2+5040*z*x^5*y^3+6300*z*x^4*y^4+5040*z*x^3*y^5+2520*z*x^2*y^6+720*z*x*y^7+840*z^3*x^6*y+2520*z^3*x^5*y^2+4200*z^3*x^4*y^3+4200*z^3*x^3*y^4+2520*z^3*x^2*y^5+840*z^3*x*y^6+2520*z^2*x^6*y+7560*z^2*x^5*y^2+12600*z^2*x^4*y^3+12600*z^2*x^3*y^4+7560*z^2*x^2*y^5+2520*z^2*x*y^6+3150*z^4*x^4*y^2+4200*z^4*x^3*y^3+3150*z^4*x^2*y^4+1260*z^4*x*y^5+5040*z^3*x^5*y+12600*z^3*x^4*y^2+16800*z^3*x^3*y^3+12600*z^3*x^2*y^4+5040*z^3*x*y^5+2520*z^5*x^3*y^2+2520*z^5*x^2*y^3+1260*z^5*x*y^4+6300*z^4*x^4*y+12600*z^4*x^3*y^2+12600*z^4*x^2*y^3+6300*z^4*x*y^4+252*z^5*x^5+252*z^5*y^5+1260*z^4*x^5+1260*z^4*y^5+210*z^6*x^4+210*z^6*y^4+1260*z^5*x^4+1260*z^5*y^4+120*z^7*x^3+120*z^7*y^3+840*z^6*x^3+840*z^6*y^3+45*z^8*x^2+45*z^8*y^2+360*z^7*x^2+360*z^7*y^2+840*z^6*x^3*y+1260*z^6*x^2*y^2+840*z^6*x*y^3+5040*z^5*x^3*y+7560*z^5*x^2*y^2+5040*z^5*x*y^3+90*z^8*x*y+360*z^7*x*y^2+2520*z^6*x^2*y+2520*z^6*x*y^2+10*z^9+z^10+10*z^9*x+10*z^9*y+90*z^8*x+90*z^8*y';
otherwise
  die;
end

%REMOVE since hard-coded now
%str = sort(strsplit(str,'+'));% sort the stuff in between + signs to try to ensure consistent ordering!!!
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

% prep the linear coordinates
[x,y,z] = ind2sub(matrixsize,locs(:));
if matrixsize(1)~=1
  x = normalizerange(x,-1,1,1,matrixsize(1));
end
if matrixsize(2)~=1
  y = normalizerange(y,-1,1,1,matrixsize(2));
end
if matrixsize(3)~=1
  z = normalizerange(z,-1,1,1,matrixsize(3));
end

% handle regular case
if isempty(weights)

  % add brackets
  str = [ '[' str ']' ];
  
  % do it
  f = eval(str);

% handle special case
else

  % divide them up
  str = strsplit(str,' ');
  
  % throw away empty elements
  str = str(cellfun(@(x) ~isempty(x),str));
  
  % weight and sum
  f = 0;
  for p=1:length(str)
    f = f + eval(str{p})*weights(p);
  end
  
end









%   switch degrees(p)
%   case 0
%     f = [f ones(size(x))];
%   case 1
%     f = [f x y z];
%   case 2
%     f = [f x.^2 y.^2 z.^2 x.*y x.*z y.*z];
%   case 3
%     f = [f x.^3 y.^3 z.^3 x.^2.*y x.*y.^2 x.^2.*z x.*z.^2 y.^2.*z y.*z.^2 x.*y.*z];
%   case 4
%     f = [f x.^4 y.^4 z.^4 x.^3.*y x.*y.^3 x.^3.*z x.*z.^3 y.^3.*z y.*z.^3 x.^2.*y.^2 x.^2.*z.^2 y.^2.*z.^2 x.^2.*y.*z x.*y.^2.*z x.*y.*z.^2];
%   otherwise
%     die;
%   end

% FOURIER STUF
% % <maxcpfov> is the maximum allowable cycles per FOV (measured
% %   along the first dimension)
% 
% % all basis functions
% % have unit length and have mean 0 (except for the DC basis
% % function).
% 
%         % not handling nyquist
%         % get rid of bogus basis functions/ DC special
%         % normalize basis functions
% 
% % 
% % % do it
% % f = [];
% % for a=0:maxcpfov
% %   for b=0:maxcpfov
% %     for c=0:maxcpfov
% %       cpfov = sqrt(a.^2 + b.^2 + c.^2);
% %       if cpfov <= maxcpfov
% % 
% %         tt = a*xx + b*yy + c*zz;
% %         if a==0 & b==0 & c==0
% %           f(:,end+1) = cos(-2*pi*tt) / sqrt(2);
% %         else
% %           f(:,end+1) = cos(-2*pi*tt);
% %           f(:,end+1) = sin(-2*pi*tt);
% %         end
% %         
% %       end
% %     end
% %   end
% % end
