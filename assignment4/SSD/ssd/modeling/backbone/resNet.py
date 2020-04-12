import torchvision
import torch

model = torchvision.models.resnet18(pretrained=True, progress=True)#, **kwargs)


#NOTE: (usikker på om dette allerede er håndtert)
#-------------------------------
#Images applied to the models need to be normalized in the following way before feed into the network:
#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
#-------------------------------


# Example RGB image with size 320x240
example_image = torch.zeros((10, 3, 320, 240))

# You can print the structure of the model to inspect it!
# print(model)
# If you print the structure, you will notice that resnet use this structure
print(model)
sequential_layers = [
    model.conv1,
    model.bn1,
    model.relu,
    model.layer1,
    model.layer2,
    model.layer3,  # Ouput 256 x 40 x 30
    model.layer4,
]
x = example_image
for layer in sequential_layers:
    x = layer(x)
    print(
        f"Output shape of layer: {x.shape}"
    )
# The output will be:
'''
Output shape of layer: torch.Size([10, 64, 160, 120])
Output shape of layer: torch.Size([10, 64, 160, 120])
Output shape of layer: torch.Size([10, 64, 160, 120])
Output shape of layer: torch.Size([10, 64, 160, 120])
Output shape of layer: torch.Size([10, 128, 80, 60])
Output shape of layer: torch.Size([10, 256, 40, 30])
Output shape of layer: torch.Size([10, 512, 20, 15])

'''

#problems:
#1. Hvordan kontrollerer jeg antall outputs
