function bytesName=NameBytes(numBytes, abbreviateFlag)
% bytesName=NameBytes(numBytes [,abbreviateFlag])
%
% NameBytes accepts a quantity of bytes and returns a string naming that number
% of bytes in more readable form.  For example:
%
%     >> c=Screen('Computer');
%     >> c.hw.physmem
% 
%     ans =
% 
%        1.3422e+09
% 
%     >> NameBytes(c.hw.physmem)
% 
%     ans =
% 
%     1.25 GB
% 
%     >>
%
% Numbers of bytes have the following names and abbreviations:
%
%   Byte        B       2^0                                     1      
%   KiloByte    KB      2^10                                1,024
%   MegaByte    MB      2^20                            1,048,576
%   GigaByte    GB      2^30                        1,073,741,824
%   TeraByte    TB      2^40                    1,099,511,627,776   
%   PetaByte    PB      2^50                1,125,899,906,842,624
%   ExaByte     EB      2^60            1,152,921,504,606,846,976
%   ZettaByte   ZB      2^70        1,180,591,620,717,411,303,424
%   YottaByte   YB      2^80    1,208,925,819,614,629,174,706,176
%
% By default NameBytes uses the abbeviated byte quantity label, for 
% example "GB" instead of "GigaBytes".  For the full name, pass
% 1 in the optional abbreviateFlag argument.  
%
% see also: NameFrequency, DescribeComputer

% HISTORY
% 12/17/04     awi      Wrote it to use in DescribeComputer.  It is
%                       little more general than required for that 
%                       purpose.  An obvious extension would be to 
%                       specify the number of places after the decimal.  
%                      




byteNames={
'Bytes',
'KiloBytes',
'MegaBytes',
'GigaBytes',
'TeraBytes',
'PetaBytes',
'ExaBytes',
'ZettaBytes',
'YottaBytes'
};

byteNamesAb={
'B',
'KB',
'MB',
'GB',
'TB',
'PB',
'EB',
'ZB',
'YB'
};

numByteNames=length(byteNames);

foundIndex=0;
for i=1:numByteNames
    if round(numBytes/2^(10*(i-1)) * 100) / 100 < 1000
        namesIndex=i;
        foundIndex=1;
        break;
    end %if
end %for
if ~foundIndex
    error('Failed to find a name for the value');
end
if nargin==2 && ~abbreviateFlag 
    bytesName=[sprintf('%3.2f', numBytes / 2^(10*(namesIndex-1))) ' ' byteNames{namesIndex}];
else
    bytesName=[sprintf('%3.2f', numBytes / 2^(10*(namesIndex-1))) ' ' byteNamesAb{namesIndex}];
end %if ..else








