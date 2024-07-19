import numpy as np

loaded_dataset = np.load('dataset.npy')
print("loaded_dataset shape:", loaded_dataset.shape)

print("np min: ", np.min(loaded_dataset))
print("np max: ", np.max(loaded_dataset))
# # Iterate over each image in the dataset
# for i in range(loaded_dataset.shape[0]):
#     image = loaded_dataset[i]
#     plt.imshow(image, cmap='gray')
#     plt.show()
