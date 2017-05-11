class SliceData(object):
    def __init__(self,image,labels,num_slice):
        self.image=image
        self.labels=labels
        self.num_slice=num_slice


def get_txt_num_slice(s):
    while s[0]!='_':
        s=s[1:]
    s=s.replace('_labels_slice_','')
    s=s.replace('.txt','')
    return s

def get_png_num_slice(s):
    while s[0]!='_':
        s=s[1:]
    s=s.replace('_image_slice_','')
    s=s.replace('.png','')
    return s

def find_image_at_slice(list_png,num_slice):
    for png_file in list_png:
        if get_png_num_slice(png_file)==num_slice:
            return png_file

def extract_txt_info(path,txt_file):
    txt_file = open (path+txt_file,'r')
    output=''
    for line in txt_file:
        output+=line
    return output


import os
from scipy import misc
directory_path='/home/apopov/Bureau/t2_ground_truth/'
list_txt=list(filter(lambda x: '.txt' in x, os.listdir(directory_path)))
list_png=list(filter(lambda x: '.png' in x, os.listdir(directory_path)))

list_SliceData=[]
for txt_file in list_txt:
    slice_num=get_txt_num_slice(txt_file)
    png_file=find_image_at_slice(list_png,slice_num)
    list_SliceData.append(SliceData(misc.imread(directory_path+png_file),extract_txt_info(directory_path,txt_file),slice_num))

print(list_SliceData)

