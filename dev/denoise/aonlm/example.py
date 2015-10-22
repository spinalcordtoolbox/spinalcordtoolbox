from aonlm import mabonlm3d
from aonlm import aonlm
import numpy as np
import nibabel as nib
from mixingsubband import mixingsubband

def test_filters():
    #Load ground truth
    nii_matlab_filtered1=nib.load('data/filtered_naonlm_3_1_1.nii');
    matlab_filtered1=nii_matlab_filtered1.get_data().astype(np.double)
    nii_matlab_filtered2=nib.load('data/filtered_naonlm_3_2_1.nii');
    matlab_filtered2=nii_matlab_filtered2.get_data().astype(np.double)
    nii_matlab_filtered3=nib.load('data/filtered_naonlm_final.nii');
    matlab_filtered3=nii_matlab_filtered3.get_data().astype(np.double)
    #Load data
    name ='data/fibercup-averaged_b-1500.nii';
    nib_image=nib.load(name)
    image=nib_image.get_data().astype(np.double)
    ##################
    fimau=np.zeros_like(image)
    fimaucpp=np.zeros_like(image)
    fimao=np.zeros_like(image)
    fima=np.zeros_like(image)
    for i in xrange(image.shape[3]):
        print "Filtering volume",i+1,"/",image.shape[3]
        fimau[:,:,:,i]=mabonlm3d(image[:,:,:,i], 3, 1, 1)
        fimaucpp[:,:,:,i]=aonlm(image[:,:,:,i], 3, 1, 1)
        #diffu=abs(fimau[:,:,:,i]-matlab_filtered1[:,:,:,i])
        #diffu=abs(fimaucpp[:,:,:,i]-matlab_filtered1[:,:,:,i])
        diffu=abs(fimau[:,:,:,i]-fimaucpp[:,:,:,i])
        diff_check=abs(fimau[:,:,:,i]-image[:,:,:,i])
        print "Maximum error [aonlm (block size= 3x3)]: ", diffu.max(),". Check: ",diff_check.max()
#        fimao[:,:,:,i]=mabonlm3d(image[:,:,:,i], 3, 2, 1)
#        diffo=abs(fimao[:,:,:,i]-matlab_filtered2[:,:,:,i])
#        diff_check=abs(fimao[:,:,:,i]-image[:,:,:,i])
#        print "Maximum error [aonlm (block size= 5x5)]: ", diffo.max(),". Check: ",diff_check.max()
#        fima[:,:,:,i]=mixingsubband(fimau[:,:,:,i],fimao[:,:,:,i])
#        diff=abs(fima[:,:,:,i] - matlab_filtered3[:,:,:,i])
#        diff_check=abs(fima[:,:,:,i]-image[:,:,:,i])
#        print "Maximum error [mixed]: ", diff.max(),". Check: ",diff_check.max()

def testAverageBlock():
    from aonlm import Average_block_cpp
    from aonlm import _average_block
    sz=(5,6,7)
    ll=sz[0]*sz[1]*sz[2]
    X=np.array(range(ll), dtype=np.float64).reshape(sz,order='F')
    #X=np.array(np.random.uniform(0,1,ll), dtype=np.float64).reshape(sz)
    error=0
    for mask in range(8):
        px=(mask&1>0)*(sz[0]-1)
        py=(mask&2>0)*(sz[1]-1)
        pz=(mask&4>0)*(sz[2]-1)
        av=np.zeros((3,3,3), order='F')
        _average_block(X, px, py, pz, av, 1)
        av_import=np.zeros((3,3,3), order='F')
        Average_block_cpp(X, px, py, pz, av_import, 1, 1)
        error+=(av-av_import).std()
    print 'Average_block error: ', error

def testValueBlock():
    from aonlm import Value_block_cpp
    from aonlm import _value_block
    sz=(5,6,7)
    ll=sz[0]*sz[1]*sz[2]
    #average=np.array(np.random.uniform(0,1,27), dtype=np.float64).reshape((3,3,3), order='F')
    average=np.array(range(27), dtype=np.float64).reshape((3,3,3), order='F')
    error=0
    variation=0
    for mask in range(8):
        px=(mask&1>0)*(sz[0]-1)
        py=(mask&2>0)*(sz[1]-1)
        pz=(mask&4>0)*(sz[2]-1)
        #E=np.array(np.random.uniform(0,1,ll), dtype=np.float64).reshape(sz, order='F')
        E=np.array(range(ll), dtype=np.float64).reshape(sz, order='F')
        L=2*np.array(range(ll), dtype=np.float64).reshape(sz, order='F')
        Value_block_cpp(E, L, px, py, pz, average, 1)
        EE=np.array(range(ll), dtype=np.float64).reshape(sz, order='F')
        #EE=E.copy()
        LL=2*np.array(range(ll), dtype=np.float64).reshape(sz, order='F')
        variation+=(E-EE).std()
        variation+=(L-LL).std()
        _value_block(EE, LL, px, py, pz, average, 1)
        error+=(E-EE).std()
        error+=(L-LL).std()
    print 'Value_block error: ', error

def testDistance():
    from aonlm import distance_cpp
    from aonlm import _distance
    ima=np.array(np.random.uniform(0,1,125), dtype=np.float64).reshape((5,5,5), order='F')
    err=0.0
    for p in np.ndindex(ima.shape):
        for q in np.ndindex(ima.shape):
            x,y,z=(p[1], p[0], p[2])
            nx,ny,nz=(q[1], q[0], q[2])
            d=_distance(ima, x,y,z, nx,ny,nz, 1)
            d_import=distance_cpp(ima, x,y,z, nx,ny,nz, 1)
            err+=(d-d_import)**2
            print d, d_import
    print 'distance error: ', err

def testDistance2():
    from aonlm import distance2_cpp
    from aonlm import _distance2
    ima=np.array(np.random.uniform(0,1,125), dtype=np.float64).reshape((5,5,5), order='F')
    means=np.array(np.random.uniform(0,1,125), dtype=np.float64).reshape((5,5,5), order='F')
    err=0.0
    for p in np.ndindex(ima.shape):
        for q in np.ndindex(ima.shape):
            x,y,z=(p[1], p[0], p[2])
            nx,ny,nz=(q[1], q[0], q[2])
            d=_distance2(ima, means, x,y,z, nx,ny,nz, 1)
            d_import=distance2_cpp(ima, means, x,y,z, nx,ny,nz, 1)
            err+=(d-d_import)**2
            print d, d_import
    print 'distance error: ', err

if __name__=='__main__':
    test_filters()
    #testAverageBlock()