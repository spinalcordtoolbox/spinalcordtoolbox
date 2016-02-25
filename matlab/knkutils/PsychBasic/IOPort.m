function IOPort(varargin)
% IOPort is a MEX file for precise control of input/output hardware, e.g.,
% Serial ports (or emulated serial ports like Serial-over-USB etc.),
% parallel ports, network ports, and special digital I/O boxes.
%
% Goal: ___________________________________________________________________
%
% It provides a unified cross-platform interface to such devices and tries
% to bundle functionality in one MEX file that is common to all those
% devices, but implemented differently on each of them. An example would be
% sending of trigger signals: The step to send a trigger signal is always
% the same. Your code wants to send a trigger signal immediately (with
% lowest possible delay), at a scheduled point in time, or automatically in
% response to some event like stimulus onset. However, the mechanism to
% send triggers is different for different devices. IOPort tries to provide
% a unified interface for such cases, so you need to code only once and
% IOPort takes care of the nitty gritty differences between different
% devices in how they send trigger signals.
%
% So far the theory. The current implementation of IOPort only provides
% unified support for accessing the serial ports of your computer. All
% other functions and device classes will be added in future releases of
% the driver.
%
% Usage: __________________________________________________________________
%
% IOPort has many functions; type "IOPort" for a list:
% 	IOPort
%
% For explanation of any particular IOPort function, just add a question
% mark "?". E.g. for 'OpenSerialPort', try either of these equivalent forms:
% 	IOPort('OpenSerialPort?')
% 	IOPort OpenSerialPort?
%

% History:
% 06/22/08 mk Wrote initial revision of this help text.

AssertMex('IOPort.m');
