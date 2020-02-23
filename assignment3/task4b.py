
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch
import numpy as np
image = Image.open("images/zebra.jpg")
print("Image shape:", image.size)

model = torchvision.models.resnet18(pretrained=True)
print(model)
first_conv_layer = model.conv1
print("First conv layer weight shape:", first_conv_layer.weight.shape)
print("First conv layer:", first_conv_layer)

# Resize, and normalize the image with the mean and standard deviation
image_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image = image_transform(image)[None]
print("Image shape:", image.shape)

activation = first_conv_layer(image)
print("Activation shape:", activation.shape)


def torch_image_to_numpy(image: torch.Tensor):
    """
    Function to transform a pytorch tensor to numpy image
    Args:
        image: shape=[3, height, width]
    Returns:
        iamge: shape=[height, width, 3] in the range [0, 1]
    """
    # Normalize to [0 - 1.0]
    image = image.detach().cpu()  # Transform image to CPU memory (if on GPU VRAM)
    image = image - image.min()
    image = image / image.max()
    image = image.numpy()
    if len(image.shape) == 2:  # Grayscale image, can just return
        return image
    assert image.shape[0] == 3, "Expected color channel to be on first axis. Got: {}".format(
        image.shape)
    image = np.moveaxis(image, 0, 2)
    return image


indices = [14, 26, 32, 49, 52]

for index in range(0, len(indices)):
    # wish output 2 subplots(2,5,..) where one should be weights of kernels while the other the greyscaled activation shape
    desired_image = indices[index]

    # weights have the dim (image,3,7,7); meaning (image,:,:,:) is what we want out. PS: need to convert
    kernel_weight_image = torch_image_to_numpy(
        first_conv_layer.weight[desired_image, :, :, :])
    print(kernel_weight_image.shape)
    plt.subplot(2, 5, index+1)
    plt.title('%s %d' % ("Weights of Kernel nr:", (desired_image)))
    plt.imshow(kernel_weight_image)

    # greyscaled activation shape dim (1,image,112,112)
    activation_image = torch_image_to_numpy(activation[0, desired_image, :, :])
    plt.subplot(2, 5, index + 6)
    plt.title('%s %d' % ("Greyscale Activation nr:", (desired_image)))
    plt.imshow(activation_image, cmap="gray")

plt.savefig("weight_and_activationImages")
plt.show()
