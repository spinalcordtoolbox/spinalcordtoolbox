function [coord,binary,S] = minimalPath3d(nC,factx,homogeneousmod , display)

% MINIMALPATH Recherche du chemin minimum de Haut vers le bas et de
% bas vers le haut tel que décrit par Luc Vincent 1998
% [sR,sC,S] = MinimalPath(I,factx)
%                                                     
%   I     : Image d'entrï¿½e dans laquelle on doit trouver le      
%           chemin minimal                                       
%   factx : Poids de linearite [1 10]                            
%                                                                
% Programme par : Ramnada Chav                                   
% Date : 22 février 2007                                         
% Modifié le 16 novembre 2007

if nargin < 3
    display = false;
end

if nargin==1
    factx=sqrt(2);
end

% load MP
% nC=ModPlage(nC,-Inf,Inf,0,1);
[m,n,p]=size(nC);
mask=isinf(nC) | isnan(nC) | nC==0;
nC(mask)=0;

cPixel = nC;

vectx=2:m-1;
vecty=2:n-1;


J1=ones(m,n,p).*Inf;
J1(:,:,1)=0;
for row=2:p
    pJ=squeeze(J1(:,:,row-1));
%     pP=cPixel(i-1,:);
    cP=squeeze(cPixel(:,:,row));
    cPm=squeeze(cPixel(:,:,row-1));
%     Iq=[pP(vect-1);pP(vect);pP(vect+1)];
    VI=repmat(cP(vectx,vecty),1,1,5);
    VIm=repmat(cPm(vectx,vecty),1,1,5);
%     VI=Ip;
    VI(:,:,1:2)=VI(:,:,1:2).*factx; VIm(:,:,1:2)=VIm(:,:,1:2).*factx;
    VI(:,:,4:5)=VI(:,:,4:5).*factx; VIm(:,:,4:5)=VIm(:,:,4:5).*factx;
    Jq=cat(3,pJ(vectx-1,vecty),pJ(vectx,vecty-1),pJ(vectx,vecty),pJ(vectx,vecty+1),pJ(vectx+1,vecty));
    if homogeneousmod
        J1(2:end-1,2:end-1,row)=min(Jq+abs(VI-VIm)./min(VI,VIm),[],3);
    else
        J1(2:end-1,2:end-1,row)=min(Jq+VI,[],3);
    end
end


J2=ones(m,n,p).*Inf;
J2(:,:,p)=0;
for row=p-1:-1:1
    pJ=squeeze(J2(:,:,row+1));
%     pP=cPixel(i-1,:);
    cP=squeeze(cPixel(:,:,row));
    cPm=squeeze(cPixel(:,:,row+1));
%     Iq=[pP(vect-1);pP(vect);pP(vect+1)];
    VI=repmat(cP(vectx,vecty),1,1,5);
    VIm=repmat(cPm(vectx,vecty),1,1,5);
%     VI=Ip;
    VI(:,:,1:2)=VI(:,:,1:2).*factx; VIm(:,:,1:2)=VIm(:,:,1:2).*factx;
    VI(:,:,4:5)=VI(:,:,4:5).*factx; VIm(:,:,4:5)=VIm(:,:,4:5).*factx;
    Jq=cat(3,pJ(vectx-1,vecty),pJ(vectx,vecty-1),pJ(vectx,vecty),pJ(vectx,vecty+1),pJ(vectx+1,vecty));
    if homogeneousmod
        J2(2:end-1,2:end-1,row)=min(Jq+abs(VI-VIm)./min(VI,VIm),[],3);
    else
        J2(2:end-1,2:end-1,row)=min(Jq+VI,[],3);
    end
end


S=J1+J2;
[Cx,mx]=min(S,[],1);
[Cy,my]=min(Cx,[],2);

coord=[mx(:,my)',my(:),[1:p]'];

binary=false(m,n,p);
for row=1:p
    binary(mx(1,my(row),row),my(row),row)=true;
end

if display
    figure(46)
    imagesc(squeeze(S(floor(end/2),:,:))',[prctile(S(:),0) prctile(S(:),10)]); axis image
    figure(47)
    imagesc(squeeze(S(:,floor(end/2),:))',[prctile(S(:),0) prctile(S(:),10)]); axis image
    figure(48)
     imagesc(squeeze(nC(floor(end/2),:,:))',[prctile(nC(:),10) prctile(nC(:),90)]); axis image
end
% % [mv2,mi2]=min(fliplr(S),[],2);]
% sR=(1:m)';
% sC=round(mi1);
% 
% if display
%     figure(47)
%     sc(nC);
%     hold on
%     plot(sC,sR,'r')
% end