function sct_reslice(src,dest)
% sct_reslice(src,dest)
% EXAMPLE: sct_reslice('highQ_mean.nii','template_roi.nii')
unix(['isct_antsRegistration --dimensionality 3 --transform syn[0.5,3,0] --metric MI[' dest ',' src ',1,32] --convergence 0 --shrink-factors 1 --smoothing-sigmas 0mm --restrict-deformation 1x1x0 --output [step0,' sct_tool_remove_extension(src,1) '_reslice.nii] --interpolation BSpline[3] -r [' dest ',' src ',0]'])
mkdir step0
unix('mv step00GenericAffine.mat step0/')
unix('mv step01InverseWarp.nii.gz step0/')
unix('mv step01Warp.nii.gz step0/')
