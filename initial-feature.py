import os
import torch
import torchvision
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Define the image transform pipeline
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load the image datasets using torchvision.datasets.ImageFolder
dataset = torchvision.datasets.ImageFolder(root='dataset/', transform=transform)

# Define the ResNet-18 feature extractor
# You can change the pretrained model to any other models in torchvision.models
# or manually load a pretrained model using torch.load()
model = torchvision.models.resnet18(pretrained=True)
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])

# Extract features from the images using the ResNet-18 feature extractor
features = []
labels = []
for image, label in dataset:
    with torch.no_grad():
        feature = feature_extractor(image.unsqueeze(0))
        feature = feature.reshape(feature.size(0), -1)
        features.append(feature.numpy())
        labels.append(label)
features = np.vstack(features)
labels = np.array(labels)

# Use t-SNE to reduce the dimensionality of the feature vectors to 2D for visualization
# change the n_components parameter to change the number of output dimensions (default: 2)
# change the perplexity parameter to change the level of clustering in the output
# change the learning_rate parameter to change the learning rate of the optimization
# Please check more t-sne parameters at https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html  
tsne = TSNE(n_components=2)
embeddings = tsne.fit_transform(features)

# Visualize the t-SNE embeddings as a scatter plot with different colors for each class
plt.figure(figsize=(10, 10))
colors = plt.cm.get_cmap('tab10', len(dataset.classes))
for c in range(len(dataset.classes)):
    mask = labels == c
    label = dataset.classes[c]
    plt.scatter(embeddings[mask, 0], embeddings[mask, 1], c=colors(c), label=label, alpha=0.5, edgecolors='none')
plt.legend()
plt.title('t-SNE Embeddings of Image Features')
plt.show()