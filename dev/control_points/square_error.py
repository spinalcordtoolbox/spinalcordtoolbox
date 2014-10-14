import nibabel
import splines_approximation_v2 as spapp
import matplotlib.pyplot as plt



def main():
	
	cmd1 = 'sct_sctraighten -i '+file+' -o '+straight

	cmd2 = 'sct_propseg -i'+straight+' -o '+file+'straight'

	file = nibabel.load(file_name)
    data = file.get_data()
    
    nx, ny, nz = spapp.get_dim(file_name)
    
    x = [0 for iz in range(0, nz, 1)]
    y = [0 for iz in range(0, nz, 1)]
    z = [iz for iz in range(0, nz, 1)]
    
    for iz in range(0, nz, 1):
            x[iz], y[iz] = ndimage.measurements.center_of_mass(numpy.array(data[:,:,iz]))

    plot(x, y, z)


def plot( x, y, z):
    plt.subplot(2,2,1)
    plt.plot(x_centerline_fit,y_centerline_fit,'r-')
    plt.plot(x,y,'b:')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(2,2,2)
    plt.plot(x_centerline_fit,z_centerline_fit,'r-')
    plt.plot(x,z,'b:')
    plt.xlabel('x')
    plt.ylabel('z')

    plt.subplot(2,2,3)
    plt.plot(y_centerline_fit,z_centerline_fit,'r-')
    plt.plot(y,z,'b:')
    plt.xlabel('y')
    plt.ylabel('z')