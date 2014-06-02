% =========================================================================
% Script
% 
% Flip gradients. Gradient file should be in FSL format. 
% 
%
% Julien Cohen-Adad <jcohen@nmr.mgh.harvard.edu>
% 2011-12-04
% 
% =========================================================================


% parameters
flip = [1 -2 3]; % example: [1 3 2], or [-1 2 3]
fname_bvecs		= 'bvecs';
fname_bvecs_new = ''; % leave empty for automatic name




% start stuff

if isempty(fname_bvecs_new)
	fname_bvecs_new = [fname_bvecs,'_flip',num2str(flip(1)),num2str(flip(2)),num2str(flip(3))];
end

gradient = textread(fname_bvecs);
% opt.read_method		= 'linePerLine';
% gradient = j_readFile(fname_bvecs,opt);
% flipmat_rep = repmat(flip,size(gradient,1),1);
% gradient_new = gradient.*flipmat_rep;
fid = fopen(fname_bvecs_new,'w');
for i=1:size(gradient,1)
	G = [sign(flip(1))*gradient(i,abs(flip(1))),sign(flip(2))*gradient(i,abs(flip(2))),sign(flip(3))*gradient(i,abs(flip(3)))];
	fprintf(fid,'%1.10f %1.10f %1.10f\n',G(1),G(2),G(3));
end
fclose(fid);
[a b c] = fileparts(fname_bvecs_new);
disp(['-> File generated: ',b])
