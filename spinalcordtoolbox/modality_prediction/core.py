import numpy as np
import torch
from spinalcordtoolbox.image import Image
from spinalcordtoolbox.modality_prediction import model as M


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
                raise RuntimeError("Negative crop.")
            
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

    # We load the acquisitions from the image module in order to benefit from all existing methods
    input_image = Image(input_path)
    input_image.change_orientation('RPI')

    # We load the model
    # Here we have to specify the path from which the model can be found when we are in the
    # actual scripts/ folder. It is probably not the most elegant way to proceed so it might require
    # to load it somewhere else.
    model = M.Classifier()
    model.load_state_dict(torch.load("../spinalcordtoolbox/modality_prediction/model.pt", map_location='cpu'))
    model.eval()

    modality = classify_acquisition(input_image, model)
    
    return(modality)


def classify_from_image(input_image):
    """
    This is our main function that will be called from the sct scripts inside the parser.
    :param input_image: loaded image of the acquisition
    :return: the predicted modality
    """

    # We load the acquisitions from the image module in order to benefit from all existing methods
    input_image.change_orientation('RPI')

    # We load the model
    # Here we have to specify the path from which the model can be found. It is obviously not the most elegant
    #  way to proceed so it might require to load it somewhere else.
    model = M.Classifier()
    model.load_state_dict(torch.load("/home/bsauty/sct/spinalcordtoolbox/modality_prediction/model.pt", map_location='cpu'))
    model.eval()

    modality = classify_acquisition(input_image, model)

    return (modality)

"""
DELETE THIS TEST BEFORE MERGING


print(classifier("/Volumes/projects/ivado-medical-imaging/spineGeneric_201907041011/result/sub-amu01/anat/sub-amu01_T1w.nii.gz"))

"""
