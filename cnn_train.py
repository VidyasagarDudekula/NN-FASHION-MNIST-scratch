import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import random
from data_preprocessor import preprocess_line, preprocess_output, preprocess_input
import torch.optim as optim
import json


device = torch.device('cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
if torch.cuda.is_available():
    device = torch.device('cuda')


def get_processed_data(file_name, shuffle, validation_set):
    with open(file_name, 'r') as file:
        total_data_lines = file.readlines()
    total_data_lines = [preprocess_line(line) for line in total_data_lines[1:]]
    if shuffle:
        random.shuffle(total_data_lines)
    x_data = []
    y_data = []
    for line in total_data_lines:
        x_data.append(preprocess_input(line[0]))
        y_data.append(line[1])
    split_length = 0
    if validation_set:
        split_length = int(len(x_data)*0.9)
    x_train_data, y_train_data = x_data[:split_length], y_data[:split_length]
    x_val_data, y_val_data = x_data[split_length:], y_data[split_length:]

    x_train_data_tensor = torch.tensor(x_train_data).view(-1, 28, 28).unsqueeze(1)
    y_train_data_tensor = torch.tensor(y_train_data)

    x_val_data_tensor = torch.tensor(x_val_data).view(-1, 28, 28).unsqueeze(1)
    y_val_data_tensor = torch.tensor(y_val_data)
    
    return x_train_data_tensor, y_train_data_tensor, x_val_data_tensor, y_val_data_tensor
    
file_name = "fashion-mnist_train.csv"
x_train_data_tensor, y_train_data_tensor, x_val_data_tensor, y_val_data_tensor = get_processed_data(file_name, shuffle=True, validation_set=True)
test_file_name = "fashion-mnist_test.csv"
x_test_data_tensor, y_test_data_tensor, _, _ = get_processed_data(test_file_name, shuffle=True, validation_set=True)

train_dataset = TensorDataset(x_train_data_tensor, y_train_data_tensor)
val_dataset = TensorDataset(x_val_data_tensor, y_val_data_tensor)
test_dataset = TensorDataset(x_test_data_tensor, y_test_data_tensor)

train_dataloader_obj = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader_obj = DataLoader(val_dataset, batch_size=64)
test_dataloader_obj = DataLoader(test_dataset, batch_size=64)

class ConvolutionNN(nn.Module):
    def __init__(self):
        super().__init__()
        # h, w = floor((h + 2 * padding - (k-1)-1)/stride + 1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1) # out dim -> [batch, 128, 28, 28]
        self.relu1 = nn.ReLU()
        self.pooling1 = nn.MaxPool2d(2, stride=2) # out dim -> [batch, 128, 14, 14]
        self.bn1 = nn.BatchNorm2d(64)
        self.dp1 = nn.Dropout(0.1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0) # out dim -> [batch, 256, 12, 12]
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(128)
        self.pooling2 = nn.MaxPool2d(2, stride=2) # out dim -> [batch, 256, 6, 6]
        self.dp2 = nn.Dropout(0.1)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1) # [batch, 256*6*6]
        self.ln1 = nn.Linear(in_features=128*6*6, out_features=1024)
        self.relu3 = nn.ReLU()
        self.layer_norm1 = nn.LayerNorm(1024)
        self.dp3 = nn.Dropout(0.1)
        self.ln2 = nn.Linear(in_features=1024, out_features=1024)
        self.relu4 = nn.ReLU()
        self.layer_norm2 = nn.LayerNorm(1024)
        self.dp4 = nn.Dropout(0.1)
        self.proj = nn.Linear(in_features=1024, out_features=10)
        
    
    def forward(self, inputs):
        inputs = self.conv1(inputs)
        inputs = self.relu1(inputs)
        inputs = self.bn1(inputs)
        inputs = self.pooling1(inputs)
        inputs = self.dp1(inputs)
        inputs = self.conv2(inputs)
        inputs = self.relu2(inputs)
        inputs = self.bn2(inputs)
        inputs = self.pooling2(inputs)
        inputs = self.dp2(inputs)
        inputs = self.flatten(inputs)
        inputs = self.ln1(inputs)
        inputs = self.relu3(inputs)
        inputs = self.layer_norm1(inputs)
        inputs = self.dp3(inputs)
        inputs = self.ln2(inputs)
        inputs = self.relu4(inputs)
        inputs = self.layer_norm2(inputs)
        inputs = self.dp4(inputs)
        inputs = self.proj(inputs)
        return inputs


model = ConvolutionNN()
model = model.to(device)

model.train()
optimizer = optim.AdamW(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()


def eval_loss():
    with torch.no_grad():
        model.eval()
        losses = 0.0
        n = 1
        for xb, yb in val_dataloader_obj:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            losses += loss.item()
            n += 1
        model.train()
    return losses/n

stepi = []
lossi = []
evali = []

step = 0
for epoch in range(10):
    for xb, yb in train_dataloader_obj:
        xb = xb.to(device)
        yb = yb.to(device)
        model.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        if step % 50 == 0:
            stepi.append(step)
            lossi.append(loss.item())
            evali.append(eval_loss())
            print(f"Epoch:- {epoch}, step:= {step}, train_loss:- {lossi[-1]:.4}, val_loss:- {evali[-1]:.4}")
        xb = xb.to('cpu')
        yb = yb.to('cpu')
        step += 1
        loss.backward()
        optimizer.step()
        

plt.figure(figsize=(10, 6))


plt.plot(stepi, lossi, label='Training Loss', color='blue')


plt.plot(stepi, evali, label='Validation Loss', color='orange')

plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)


plt.savefig('training_loss_plot.png')
print("Plot saved as training_loss_plot.png")

plt.show()


def test_model():
    model.eval() 
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for xb, yb in test_dataloader_obj:
            xb = xb.to(device)
            yb = yb.to(device)
            
            outputs = model(xb)
            loss = criterion(outputs, yb)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += yb.size(0)
            correct += (predicted == yb).sum().item()

    avg_test_loss = test_loss / len(test_dataloader_obj)
    accuracy = 100 * correct / total
    
    print(f'Test Loss: {avg_test_loss:.4f}')
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    stats = {
        'test_loss': avg_test_loss,
        "test_accuracy": accuracy
    }
    return stats


stats = test_model()
with open('cnn_model_stats.json', 'w') as file:
    json.dump(stats, file)