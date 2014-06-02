function [status,output] = sct_call_fsl(cmd)
% Wrapper around calls to FSL binaries
% clears LD_LIBRARY_PATH and ensures the FSL envrionment variables have been set up.
% Debian/Ubuntu users should uncomment as indicated
%
%
% INPUTS
% ========================================================================
% cmd					string. 
%
%  
% OUTPUTS
% ========================================================================
% status
% output
% 
% 
% EXAMPLE
% ========================================================================
% [status, output] = call_fsl(cmd)
%
% 
% DEPENDENCES
% ========================================================================
% none
%
%
% COMMENTS
% ========================================================================  
% Based on call_fsl.m from FSL.
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

fsldir = getenv('FSLDIR');

% Debian/Ubuntu - uncomment the following
%fsllibdir=sprintf('%s/%s', fsldir, 'bin');

if ismac
	dylibpath=getenv('DYLD_LIBRARY_PATH');
	setenv('DYLD_LIBRARY_PATH');
else
  ldlibpath=getenv('LD_LIBRARY_PATH');
  setenv('LD_LIBRARY_PATH');
  % Debian/Ubuntu - uncomment the following
  % setenv('LD_LIBRARY_PATH',fsllibdir);
end

command = sprintf('/bin/sh -c ". ${FSLDIR}/etc/fslconf/fsl.sh; %s"\n', cmd);
[status,output] = system(command);

if ismac
  setenv('DYLD_LIBRARY_PATH', dylibpath);
else
  setenv('LD_LIBRARY_PATH', ldlibpath);
end

