import torch
import os
import numpy as np
import torch.optim as optim
from torchvision.models import resnet18
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

__dirname__ = "/src/MotorMovement"
checkpoint_path = (os.getcwd() + __dirname__ + '/checkpoints')

def clear_checkpoint(path):
    """
    Clear (delete) the checkpoint file.
    """
    if os.path.isfile(path):
        os.remove(path)
        print(f"Checkpoint cleared at {path}")
    else:
        print(f"No checkpoint found at {path}")

def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Save the model and optimizer state.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at {path}")

def load_checkpoint(model, optimizer, path):
    """
    Load the model and optimizer state.
    """
    if os.path.isfile(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Checkpoint loaded from {path}")
        return start_epoch, loss
    else:
        print(f"No checkpoint found at {path}")
        return 0, None
    
def set_seed():
    torch.cuda.manual_seed_all(448)
    torch.manual_seed(448)
    np.random.seed(448)

def split_channels(batch: np.ndarray) -> np.ndarray:
    """
    Reshapes the batch data to match the input shape expected by ResNet.
    Reshaping the EEG data from (N, 9, 160) to (N, 3, 3, 160) to fit ResNet's input requirement.
    """
    return batch.reshape(-1, 3, 3, 160)

def main(retrain, rank):
    set_seed()
    
    # Load and initialize ResNet model (ResNet18 in this case)
    resnet = resnet18(pretrained=False)
    resnet.fc = nn.Linear(resnet.fc.in_features, 5)

    if retrain:
        clear_checkpoint(checkpoint_path)
        # Load EEG data and labels
        eeg_data = np.load(os.getcwd() + '/data/X_eeg.npy')
        labels = np.load(os.getcwd() + '/data/y_labels.npy') 
        
        # Slice data (optional)
        eeg_data = eeg_data[0:len(eeg_data)]
        labels = labels[0:len(labels)]
        
        # Reshape the data for ResNet input
        eeg_data = split_channels(eeg_data)

        # Split into train, validation, and test sets
        split_ratio = int(0.8 * len(eeg_data))
        train_data, test_data = eeg_data[:split_ratio], eeg_data[split_ratio:]
        train_labels, test_labels = labels[:split_ratio], labels[split_ratio:]

        split_ratio = int(0.85 * len(train_data))
        train_data, val_data = train_data[:split_ratio], train_data[split_ratio:]
        train_labels, val_labels = train_labels[:split_ratio], train_labels[split_ratio:]

        # Convert data to PyTorch tensors
        train_data = torch.tensor(train_data, dtype=torch.float32)
        train_labels = torch.tensor(train_labels, dtype=torch.long)
        val_data = torch.tensor(val_data, dtype=torch.float32)
        val_labels = torch.tensor(val_labels, dtype=torch.long)
        test_data = torch.tensor(test_data, dtype=torch.float32)
        test_labels = torch.tensor(test_labels, dtype=torch.long)

        # Create DataLoaders
        batch_size = 32
        train_dataset = TensorDataset(train_data, train_labels)
        val_dataset = TensorDataset(val_data, val_labels)
        test_dataset = TensorDataset(test_data, test_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Set loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(resnet.parameters(), lr=0.001, momentum=0.9)
        
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu") TODO FIGURE OUT WHY GPU ACCELERATION ISNT WORKING
        device = "cpu"
        resnet = resnet.to(device)

        for epoch in range(7):
            resnet.train()
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = resnet(inputs)
                loss = criterion(outputs, torch.argmax(labels, dim=1)) 
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if (i + 1) % 200 == 0:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                    running_loss = 0.0

            # Validation loop
            resnet.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = resnet(inputs)
                    loss = criterion(outputs, torch.argmax(labels, dim=1)) 
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(torch.argmax(labels, dim=1)).sum().item()
            
            print(f'Validation Loss: {val_loss / len(val_loader):.3f}, '
                f'Accuracy: {100 * correct / total:.2f}%')
            save_checkpoint(resnet, optimizer, epoch + 1, running_loss, checkpoint_path)
    else:
        pass # TODO LOAD FROM CHECKPOINT
    print('Finished Training')

if __name__ == "__main__":
    retrain = True  # Set retrain flag as needed
    main(retrain, 0)
    print('GG')