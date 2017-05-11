


#TODO: prevoir si ce nest pas T2 au debut


class SliceData(object):
    def __init__(self,image,labels):
        self.image=image
        self.labels=labels


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





import os
directory_path='/home/apopov/Bureau/t2_ground_truth'
list_txt=list(filter(lambda x: '.txt' in x, os.listdir(directory_path)))
list_png=list(filter(lambda x: '.png' in x, os.listdir(directory_path)))

for txt_file in list_txt:
    slice_num=get_txt_num_slice(txt_file)
    png_file=find_image_at_slice(list_png,slice_num)
    print(txt_file,png_file)

