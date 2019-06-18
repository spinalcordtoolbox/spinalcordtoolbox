from nicolas_scripts.functions_sym_rot import *
from scipy import misc
image = misc.imread("/home/nicolas/Pictures/chat.png")
import matplotlib.pyplot as plt
import numpy as np


image = np.mean(image[:, :, 0:3], axis=2)

image_chat = ImageSplit(image, (255, 205), pi/3)

plt.figure()
plt.subplot(221)
plt.imshow(image)
plt.title("image")
plt.subplot(222)
plt.imshow(image_chat.image_half1)
plt.title("first half")
plt.subplot(223)
plt.imshow(image_chat.image_half1)
plt.title("second half")
plt.show()

pass

