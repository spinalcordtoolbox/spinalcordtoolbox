% gestaltBits = Gestalt(selector)
%
% OS X: ___________________________________________________________________
%
% Gestalt calls the eponymous Macintosh system function Gestalt, a
% general-purpose function that reports on particular features of your 
% computer.  Gestalt accepts "selector", a four-character string identifying
% the feature you wish to query and returns "gestaltBits", a 32-element
% logical array holding the result of the query. 
%
% For example:
%
%   gestaltbits = gestalt('sysa')
%
%      gestaltbits(32) will be 1 if run from a 680x0-based Macintosh, while
%      gestaltbits(31) will be 1 if run from a PowerPC-based Macintosh. 
%
% For a list of four-character selector codes, see the Carbon Gestalt
% Manager Reference 
% web http://developer.apple.com/ ;
%
% If the call to Carbon Gestalt returns an error then MATLAB Gestalt returns
% the error code in a double instead of gestaltBits in a logical array.
% Gestalt error codes are:
%
%   gestaltUnknownErr       -5550   An unknown error.
%   gestaltUndefSelectorErr -5551   An undefined selector was 
%                                     passed to the Gestalt Manager.
%   gestaltDupSelectorErr   -5552   You tried to add an entry 
%                                     that already existed.
%   gestaltLocationErr      -5553   The gestalt function ptr was not in the
%                                     system heap.
%
% In MATLAB 6.0 and greater the Psychtoolbox supplies Gestalt. Gestalt is a 
% work-alike implementation of the identically-named function previously
% provided by MATLAB.  The only differences between Psychtoolbox Gestalt and 
% MATLAB Gestalt are:
%
%   1. Psychtoolbox Gestalt returns a struct holding information about
%      itself when passed 'Version', for example:
%     >> Gestalt('Version')
% 
%     ans = 
% 
%          version: '1.0.3.62269826'
%            major: 1
%            minor: 0
%            point: 3
%            build: 62269826
%             date: 'Dec  7 2004'
%             time: '17:10:26'
%           module: 'Gestalt'
%          project: 'OpenGL Psychtoolbox'
%               os: 'Apple OS X'
%         language: 'MATLAB'
%          authors: [1x1 struct]
%
%   2. Psychtoolbox Gestalt will return the error code in the event of any 
%      Gestalt error.  MATLAB Gestalt will return the error code in the
%      event of error code -5551.  Its behavior for other error codes is
%      unknown.
%
% OS 9: ___________________________________________________________________
%
% In MATLAB versions below 6.0 MATLAB supplies Gestalt. 
%
% WINDOWS: ________________________________________________________________
% 
% Gestalt does not exist in Windows.  
% 
% _________________________________________________________________________
%
% See also: Screen('Computer?'), MacModelName, AppleVersion

% HISTORY
% 12/6/04  awi     Wrote it.  Behavior based on Gestalt in MATLAB 5.0
%                  Documentation derived from Apple's description of
%                  Gestalt.  Example borrowed from MATLAB 5.0 help.
% 12/7/04  awi     Improved documentation: mentioned error codes, listed
%                  differences.  Added AssertMex call. 
% 10/4/05	  awi	 Note here cosmetic changes by dgp between 12/7/04 and 10/4/05.  
                 
% This file should not execute on OS X or OS 9 because MATLAB should
% execute the corresponding mex file instead.  
AssertMex('OSX','OS9');
