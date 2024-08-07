import numpy as np
import matplotlib.pyplot as plt
import os

from utils import apply_fresnel_propagation_np, normalize

IMAGE_SIZE = 64
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "train_results.npy")

data = np.load(DATA_DIR)

for i in range(data.shape[0]):
    imgs = data[i]
    original, pred, reconstructed = imgs[:, :IMAGE_SIZE], imgs[:, IMAGE_SIZE:2*IMAGE_SIZE], imgs[:, 2*IMAGE_SIZE:]
    print("Hologram pixel sum: ", np.sum(pred))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(pred, cmap='gray')
    plt.title('Predicted')
    plt.axis('off')

    # reconstructed[reconstructed>0.3] = 0
    plt.subplot(1, 3, 3)
    plt.imshow(reconstructed, cmap='gray')
    plt.title('Reconstructed')
    plt.axis('off')

    plt.show()
