function list = sct_dir(string)
% LIST FILES LIKE DIR SCRIPT.... BUT SORT BY FILENAME!!
% dir sort like this     : 1.txt 10.txt 2.txt
% sct_dir sort like this : 1.txt 2.txt  10.txt
list=textscan(ls(string),'%s');list=sort_nat(list{1});
end