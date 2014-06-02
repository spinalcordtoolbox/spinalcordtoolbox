function PO = j_prepend(PI,pre)
for i=1:size(PI,1)
    [pth,nm,xt,vr] = fileparts(deblank(PI(i,:)));
    PO(i,:)        = fullfile(pth,[pre nm xt vr]);
end
return
