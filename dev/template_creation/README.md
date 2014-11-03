PREPROCESSING
=============

- Preprocessing is done for each subject. Preprocessing is first done for T2, and then for T1. Preprocessing for T1 involves registration to T2.

- the list of commands is included in files:
  - preprocess_t2.sh
  - preprocess_t1.sh

N.B. preprocessing for T1 has less steps because some are made in T2. E.g., generation of centerlines. 

List of steps is below. Functions involved are in brackets ():

1. Crop image a little above the brainstem and a little under L2/L3 vertebral disk (sct_crop_image). 
2. Generate a centerline with Propseg and correct/finish/improve it manually (sct_propseg, sct_erase_centerline, sct_generate_centerline, fslmaths -add). Start by generating a centerline with propseg (you may need to use a mask to initialize it), then you can erase the parts that you dont like using sct_erase_centerline -s start -e end. Then you have to create a mask to generate centerline parts that are missing (typically in the brainstem and in lumbar levels): put landmarks all along the centerline part and then use sct_generate_centerline. Then you have to add all those parts using fslmaths -add. Then, you have to binarize the resulting volume. The centerline must cover ALL cropped image, i.e., Z=[0..Zmax].
3. Straighten volume using this centerline (sct_straighten_spinalcord)
4. Apply warping field curve to straight the the centerline  ( sct_WarpImageMultiTransform )
5. Crop volume one more time to erase the blank spaces (sct_detect_extrema, sct_crop_image ). To do this use sct_detect_extrema with your straight centerline as input it will return you two arrays [a,b,c] [d,e,f] containning the coordinates of the upper and lower nonzero points. use c and f to crop your volume. 
  - todo: replace detect_extrema with label_utils
6. Create a cross of 5 mm at the top center of the volume and a point at the bottom center of the volume ( sct_create_cross ). Use sct_create_cross with your straightened-cropped volume with flag -x a -y b. Usually a=d b=e which is normal if the straightening is good. If they are not equal then make a choice…
  - todo: use sct_label_utils
7. Push the straightened volume into the template space. The template space has crosses in it for registration. If you want to use another template or cross landmarks, there is a flag. You have to use ``sct_push_into_template_space``. Input: previous volume, mask created at previous step.
8. Create a mask in which you put 5 labels with following values: 1: PMJ, 2: C3, 3: T1, 4: T7, 5: L1. 
9. (ALREADY DONE: Use sct_average_levels to create the same landmarks in the template space. This scripts take the folder containing all the masks created in previous step and for a given landmark it averages values across all subjects and put a landmark at this averaged value. You only have to do this once for a given preprocessing process. If you change to preprocessing or if you had subjects 2 choices : assume that it will not change the average too much and use the previous mask, or generate a new one.)
10. Use sct_align_vertebrae -t affine (transformation) -w spline (interpolation) to align the vertebrae using affine transformation along Z.
11. Crop the straight centerline the same way you've cropped the volume the second time and push this straight cropped centerline into the template space (sct_crop_image, sct_create_cross, sct_push_into_template_space)
12. use this centerline and the volume to normalize intensity (sct_normalize). Before you should apply the transformation outputed in 10 to the centerline generated in 11


IMPORTANT : 
normalize.sh does 10 and 12 once 11 is done

For T1 volumes, do the same as: preprocess_t1/TR.

All data are located in:
~~~
users_hd2/jtouati/data/template_data
~~~


TEMPLATE CREATION
=================

Connect to Guillimin
--------------------

- First register yourself. Choose Guillimin server
- To connect to the server :  
	``ssh <username>@guillimin.clumeq.ca``
  - To transfer files from your system to the server (run this from your system): 
  - To transfer files from the server to your system:
    ``scp <username>@ guillimin.clumeq.ca:<path to file >`
- You have a .bash_profile and .bashrc on the server. Use .bahsrc rather than .bash_profile.
Add this to your .bashrc:
~~~
PATH=${PATH}:/gs/project/rrp-355-aa/bin/ants/bin
export ANTSPATH=/gs/project/rrp-355-aa/bin/ants/bin/
~~~

- There are pre-installed modules that you'll need to load in order to use (e.g. cmake). To see all modules available :

~~~
module avail
~~~

- To load module (have to reload every time you login) (you can put this in your .bashrc if you need the module all the time):

~~~
module load <module_name>
~~~

- You have to build everything from source because you don't have root permission to install anything yourself. You can send an email to guillimin@calculquebec.ca if you need them to install something on your session. They are quite responsive.

- In terms of Disk space you have a home folder /home/<username> -> 10GB, a project space /gs/project/<id> id is the one shared by all the people in the group (login to calculquebec website and you’ll find it) –> 1TB
	
- The folder common to the lab (where you need to work) is:
~~~
/gs/project/rrp-355-aa
~~~

See https://wiki.calculquebec.ca/w/Accueil

Read the next one carefully everything’s in there :
http://www.hpc.mcgill.ca/index.php/starthere 

- To see if you have jobs running or pending enter :
showq –u <username>

- To run a job : you need to create a .sh file with the correct header (see bellow for examples), submit.sh e.g. Then just enter :
qsub submit.sh

Example : 
~~~
#!/bin/bash
#PBS -l nodes=1:ppn=16,pmem=31700m,walltime=48:00:00
#PBS -A rrp-355-aa
#PBS -o output.txt
#PBS -e error.txt
#PBS -V
#PBS -N build_template
~~~
cd /gs/project/rrp-355-aa/final_data_T2

bash buildtemplateparallel.sh -d 3 -o AVT -n 0 -c 2 -j 16 *.nii.gz

To create template, run: ``qsub submit_template_ants.sh``


Connect to Magma 
----------------

- To connect use ssh
~~~
ssh <username>@magma.criugm.qc.ca
~~~
- Add this to your ``~/.bashrc`` (only need to do it once):
~~~
PATH=${PATH}:/usr/local/ants/bin
export ANTSPATH=/usr/local/ants/bin/
~~~
- go to data folder
~~~
cd /data/neuropoly<username>
~~~
- Launch script to create template (based on qsub):
~~~
./buildtemplateparallel.sh -d 3 -n 0 -o AVT *.nii.gz
~~~

- N.B. if we do not want the normalization, you need to put: 0 at line 918
- N.B. Things are installed in ``/usr/local/`` (ants, fsl …)
- N.B. be sure to set the right path to waitForSGEQjobs.pl line 1205
- Typically you’ll want to do screen then run your script. You can close the terminal, it will continue to run.

- To see all the jobs currently running type :
~~~
qstat
~~~


OUTPUT OT TEMPLATE CREATION
---------------------------

GR_iteration_X --> within this folder, the generated template is called: AVTtemplate.nii.gz.

To find the final version of the template, go to the highest iteration.
