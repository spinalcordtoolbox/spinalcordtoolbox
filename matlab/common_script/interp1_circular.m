function valq=interp1_circular(theta,val,thetaq)
% valq=interp1_circular(theta,val,thetaq)
% theta and thetaq in radian
theta=theta(:); val=val(:); thetaq=thetaq(:);
val2 = [val' val' val'];
theta2 = [theta'-2*pi, theta', theta'+2*pi];
thetaq2 = [thetaq'-2*pi, thetaq', thetaq'+2*pi];
valq2 = interp1(theta2,val2,thetaq2);
valq=valq2(end/3+1:2*end/3);