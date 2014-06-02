% =========================================================================
% SCRIPT
% j_parameters
%
% param to setup depending on the MATLAB and toolbox installation. The
% used structure is 'param'. It is defined as global variable.
% This file SHOULD be visible in the matlab path.
%
% In each script or function, call this file if necessary.
%
% COMMENTS
% Julien Cohen-Adad 2008-04-08
% =========================================================================

global	param

% Figure properties
param.figure.font_name			= 'Arial';
param.figure.font_size.axes		= 16; % FontSize
param.figure.font_size.label	= 20;
param.figure.font_size.title	= 18;
param.figure.line_width			= 2;
param.figure.line_width_sd		= 1;
param.figure.bar_width			= 0.5;
param.figure.bar_line_width		= 0.5;
param.figure.position			= [10 5 15 8];

% path
param.fsl.version				= '5.98';
param.fsl.path					= '/usr/local/fsl';
param.spm.version				= 'spm5'; % 'spm2' or 'spm5'
param.syngo						= 'VB15'; % version of console in Siemens scanner


