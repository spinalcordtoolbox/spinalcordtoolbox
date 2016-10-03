class QcPatches:

    def save(self, im_data):

        from msct_image import Image
        image_input = Image(im_data)
        nx, ny, nz, nt, px, py, pz, pt = image_input.dim

        patch = image_input.data[:,:,nz/2]

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.imshow(patch)
        fig1 = plt.gcf()
        fig1.savefig('test.png', format='png')
        plt.close()

