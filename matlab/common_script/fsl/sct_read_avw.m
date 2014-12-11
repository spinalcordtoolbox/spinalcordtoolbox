function [img,dims,scales,bpp,endian] = sct_read_avw(fname)
% Read NIFTI file. Based on read_avw.m from FSL.
%
% Read in an Analyze or nifti file into either a 3D or 4D
% Note: automatically detects - unsigned char, short, long, float
%        double and complex formats
% 
%
% INPUTS
% ========================================================================
% fname					string. File name of input image. Don't put the extension!
%
%  
% OUTPUTS
% ========================================================================
% img					array of data
% dims					4 dimensions.
% scales				scaling of the data
% bpp					bits per pixel
% endian				char. 'l' for little-endian or 'b' for big-endian
% 
% 
% EXAMPLE
% ========================================================================
% [img,dims,scales,bpp,endian] = sct_read_avw('data/my_file')
%
% 
% DEPENDENCES
% ========================================================================
% none.
%
%
% COMMENTS
% ========================================================================  
% Based on read_avw.m from FSL.
%
% Copyright (c) 2013  NeuroPoly, Polytechnique Montreal <www.neuropoly.info>
% Created by: Julien Cohen-Adad
%
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included in
% all copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
% THE SOFTWARE.
% ========================================================================


%% convert to uncompressed nifti pair (using FSL)
tmpname = tempname;

command = sprintf('FSLOUTPUTTYPE=NIFTI_PAIR; export FSLOUTPUTTYPE; $FSLDIR/bin/fslmaths %s %s', fname, tmpname);
sct_call_fsl(command);

[dims,scales,bpp,endian,datatype] = sct_read_avw_hdr(tmpname);
if (datatype==32),
	% complex type
	img = sct_read_avw_complex(tmpname);
else
	img = sct_read_avw_img(tmpname);
end
  
% cross platform compatible deleting of files
delete([tmpname,'.hdr']);
delete([tmpname,'.img']);