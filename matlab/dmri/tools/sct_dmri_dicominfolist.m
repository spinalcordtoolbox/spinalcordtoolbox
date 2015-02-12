dbstop if error
list=dir('MR*');
a={};
for i={list.name}
u=dicominfo(i{:});
a{end+1}=u.SequenceName;
end
sort_nat(a)