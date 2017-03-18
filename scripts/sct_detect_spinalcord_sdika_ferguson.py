import os
import time
import pickle

####################################################################################################################
#   User Case

path_ferguson_work = '/home/neuropoly/code/spine-ms-tmi/'
path_config = path_ferguson_work + 'ferguson_config.pkl'
with open(path_config) as outfile:    
	config = pickle.load(outfile)
	outfile.close()
contrast = config['contrast']
nb_image_train = config['nb_image_train']
valid_subj = config['valid_subj']
path_ferguson_input_img = path_ferguson_work + 'input_img/' + contrast + '/'
path_ferguson_train = path_ferguson_work + contrast + '_'+ str(nb_image_train) + '/'
path_ferguson_res_img = path_ferguson_work + 'output_img_' + contrast + '_'+ str(nb_image_train) + '/'
if config['svm_hog_alone']:
	cmd_line_test = './spine_detect -ctype=maxslice -lambda=1 '
else:
	cmd_line_test = './spine_detect -ctype=dpdt -lambda=1 '

# subj_id = []
# for f in valid_subj:
# 	if f.endswith('.img') and not '_seg' in f:
# 		subj_id.append(f.split('.')[0])

# path_sub_train = [path_ferguson_train + f + '/' for f in os.listdir(path_ferguson_train) if os.path.isdir(path_ferguson_train + f)] 

path_sub_train = path_ferguson_train

if os.path.exists(path_ferguson_res_img):
	os.system('rm -r ' + path_ferguson_res_img)
os.makedirs(path_ferguson_res_img)

txt_name = [f for f in os.listdir(path_sub_train) if f.endswith('.txt') and not '_ctr' in f]
for zz,tt in enumerate(txt_name):
	path_txt = path_sub_train + tt
	path_txt_ctr = path_sub_train + tt.split('.')[0] + '_ctr.txt'
	
	t0 = time.time()
	os.system('./spine_train_svm -hogsg -incr=20 ' + tt.split('.')[0] + ' ' + path_txt + ' ' + path_txt_ctr + ' --list True')
	os.system('mv ' + tt.split('.')[0] + '.yml ' + path_sub_train + tt.split('.')[0] + '.yml')

	id_train_subj = [line.rstrip('\n').split('/')[-1] for line in open(path_txt)]

	# path_res_cur = path_ferguson_res_img + '__'.join(id_train_subj) + '/'
	path_res_cur = path_ferguson_res_img +tt.split('.')[0]+ '/'

	try:
		os.makedirs(path_res_cur)
	except Exception, e:
		print e
		path_res_cur = path_ferguson_res_img + str(zz).zfill(3) + '/'
		os.makedirs(path_res_cur)


	for ss_test in valid_subj:
		if not ss_test in id_train_subj:
			t_test_init = time.time()
			os.system(cmd_line_test + path_sub_train + tt.split('.')[0] + ' ' + path_ferguson_input_img + ss_test + ' ' + path_res_cur + ss_test)
			if nb_image_train == 666 or nb_image_train == 5228 or nb_image_train == 11:
				with open(path_res_cur + ss_test + '.txt', 'w') as text_file:
					text_file.write(str(time.time() - t_test_init))
					text_file.close()
			if os.path.isfile(path_res_cur + ss_test + '_ctr.txt'):
				os.remove(path_res_cur + ss_test + '_ctr.txt')
			if os.path.isfile(path_res_cur + ss_test + '_svm.hdr'):
				os.remove(path_res_cur + ss_test + '_svm.hdr')
			if os.path.isfile(path_res_cur + ss_test + '_svm.img'):
				os.remove(path_res_cur + ss_test + '_svm.img')

	delta_t = time.time() - t0
	with open(path_res_cur + 'time.txt', 'w') as text_file:
            text_file.write(str(delta_t))