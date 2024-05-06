
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import h5py

def modify_model(model):
    # Remove the last layer (classifier) to get the features
    return torch.nn.Sequential(*(list(model.children())[:-1]))

def save_hdf5(filename, features, labels):
    with h5py.File(filename, 'w') as h5file:
        h5file.create_dataset('features', data=features)
        h5file.create_dataset('labels', data=labels)
        print(f"Data saved to {filename}")

def load_hdf5(filename):
    with h5py.File(filename, 'r') as h5file:
        features = h5file['features'][:]
        labels = h5file['labels'][:]
        return features, labels

def extract_features(data_loader, model, device):
    model = modify_model(model).to(device)
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for images, lbls in data_loader:
            images = images.to(device)
            output = model(images)
            features.append(output.view(images.size(0), -1).cpu())
            labels.extend(lbls)

    features = torch.cat(features, dim=0).numpy()
    labels = torch.tensor(labels).numpy()
    return features, labels

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    batch_size = 128  # Matched to your initial notebook

    # Setup Datasets and DataLoaders
    train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, pin_memory=True)

    test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

    # Setup Models
    vgg16 = models.vgg16(pretrained=True)
    alexnet = models.alexnet(pretrained=True)

    vgg16_features, vgg16_labels = extract_features(train_loader, vgg16, device)
    alexnet_features, alexnet_labels = extract_features(train_loader, alexnet, device)

    save_hdf5('vgg16_train_features.h5', vgg16_features, vgg16_labels)
    save_hdf5('alexnet_train_features.h5', alexnet_features, alexnet_labels)

    print("Features extracted and saved successfully for training data.")

if __name__ == "__main__":
    main()
