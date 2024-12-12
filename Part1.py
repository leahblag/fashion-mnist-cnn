import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Set random seed for reproducibility
torch.manual_seed(42)

# Device configuration (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data transformations: Convert to Tensor and normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load Fashion MNIST dataset
train_data = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Split training data into train and validation sets
train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_data, val_data = random_split(train_data, [train_size, val_size])

# Create data loader for training data
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# CNN Model definition
class FashionMNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5),  # Convolutional layer 1
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(8, 16, kernel_size=5),  # Convolutional layer 2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))  # Adaptive pooling for flexible input size
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),  # Fully connected layer 1
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),          # Fully connected layer 2
            nn.ReLU(inplace=True),
            nn.Linear(84, 10)            # Output layer (10 classes for Fashion MNIST)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Initialize weights
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Model instantiation
model = FashionMNISTCNN().to(device)
model.apply(weights_init)

# Save the model architecture
torch.save(model.state_dict(), 'fashion_mnist_cnn_part1.pt')
print("Model saved successfully.")

# Optional training loop for extra points :)
num_epochs = 5
learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

print("Training complete.")
