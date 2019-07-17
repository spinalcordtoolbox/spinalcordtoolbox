import torch
import numpy as np
import os
import nipy
from spinalcordtoolbox.image import Image
from spinalcordtoolbox import resampling
from spinalcordtoolbox.modality_prediction import model as M
import sct_utils as sct


class Acquisition(object):
    
    def __init__(self, axial_slices=None):
        self.slices = axial_slices

    def loadFromImage(self, image):
        """
        This method loads the slices in the slices attribute from a spinalcordtoolbox.image.Image object.
        At this point it is only a list of arrays and will only be converted
        to tensor after transformations.
        """
        image_original = image.data
        if image_original.size == 0 :
            raise RuntimeError(f"Empty slice in subject {image._path}.")
        axial_slices = []
        for i in range(image_original.shape[2]):
            axial_slices.append(image_original[:,:,i])
        self.slices = axial_slices

    def StandardizeTransform(self):
        """
        This method standardizes each slices individually
        """
        for i in range(len(self.slices)):
            mean, std = self.slices[i].mean(), self.slices[i].std() 
            self.slices[i] = (self.slices[i] - mean) / std
    
    def CenterCropTransform(self, size=128):
        """
        This method centers the image around the center
        """
        for i in range(len(self.slices)):
            y, x = self.slices[i].shape
            
            startx = x // 2 - (size // 2)
            starty = y // 2 - (size // 2)
            
            if startx < 0 or starty < 0:
                # in case the image is too small we need to pad with 0s
                padded = np.pad(self.slices[i], size//2, 'constant')
                y, x = padded.shape

                startx = x // 2 - (size // 2)
                starty = y // 2 - (size // 2)

                self.slices[i] = padded[starty:starty + size,
                                        startx:startx + size]

            else:
                self.slices[i] = self.slices[i][starty:starty + size,
                                           startx:startx + size]

    def ToTensor(self):
        """
        This method returns the tensor in the correct shape to feed the network
        ie. torch.Size([16, 1, 128, 128]) with dtype = float
        """        
        slices = np.asarray(self.slices, dtype=np.float32)
        slices = np.expand_dims(slices, axis=1)
        tensor = torch.FloatTensor(slices)
        return(tensor)


def classify_acquisition(input_image, model=None):
    """
    This function takes as input an object from spinalcordtoolbox.image object and the
    model and outputs the predicted modality.

    :param input_image: the loaded and correctly oriented Image object
    :param model: the loaded pre-trained model
    :return (string): name of the class
    """

    # We preprocess the slices within our Acquisition Class
    acq = Acquisition()
    acq.loadFromImage(input_image)
    acq.CenterCropTransform()
    acq.StandardizeTransform()
    
    input_slices = acq.ToTensor()

    with torch.no_grad():
        
        input_slices = input_slices

        outputs = model(input_slices)   
        _, preds = torch.max(outputs, 1)
        preds = preds.tolist()

    # We consider the mode of the predictions for each slice
    numeral=[[preds.count(nb), nb] for nb in preds]
    numeral.sort(key=lambda x:x[0], reverse=True)
    modality = numeral[0][1]

    class_names = ["t1", "t2s", "t2"]
    return(class_names[modality])


def classify_from_path(input_path):
    """
    This is our main function that will be called from the sct scripts inside the parser.
    :param input_path: raw path to the nifti acquisition
    :return: the predicted modality
    """

    # We use the Image module to load a resampled and well oriented image
    input_image = Image(input_path)
    input_image.change_orientation('RPI')

    logger.info("Resample the image to 0.5x0.5 mm in-plane resolution...\n")
    input_resolution = input_image.dim[4:7]
    new_resolution = 'x'.join(['0.5', '0.5', str(input_resolution[2])])

    input_nipy = nipy.load_image(input_path)
    res_nipy = resampling.resample_nipy(img=input_nipy, new_size=new_resolution,
                                        new_size_type='mm', interpolation='linear', verbose=0)
    input_image.data = res_nipy.get_data()

    # We load the model
    model = M.Classifier()
    model_path = os.path.join(sct.__sct_dir__, 'data', 'modality_pred_model/modality_pred_model.pt')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    modality = classify_acquisition(input_image, model)
    
    return(modality)

