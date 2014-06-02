
txt = textread('DiffusionVectorsDSI_257_b0.txt','%s');
j=1;
for i=5:8:4120
	grad_list(j,:) = [str2num(txt{i}(1:end-1)) str2num(txt{i+1}(1:end-1)) str2num(txt{i+2})];
	j=j+1;
end
j_dmri_gradientsDisplay(grad_list)
