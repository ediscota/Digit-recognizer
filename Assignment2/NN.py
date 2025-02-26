import random, os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import torch
import torch.nn as nn
import torch.nn.functional as F


data_dir = 'data'
save_dir = 'models'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#caricamento dati
data = fetch_openml('mnist_784', version=1, as_frame=False, data_home=data_dir)
x = data.data #array numpy di 784 colonne, ogni riga pixel di singola immagine
y = data.target.astype(float)  # label 0-9 come float
num_samples = len(y)
num_features = x.shape[1] #saranno i successivi input della nn, corrisponde a una immagine
num_labels = len(set(y))


#dataset split training/validation/test set
random_state = 0
val_size = 0.2
test_size = 0.2
x_train, x_val_test, y_train, y_val_test = train_test_split(x / 255.0, y, random_state=random_state,
                                                            test_size=test_size) # scikit-learn utility
x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, random_state=random_state, test_size=test_size) # scikit-learn utility



#iperparametri
batch_size = 32
lr = 0.0001
num_epochs = 10
hidden_dim = 256

seed = 10
log_every = 1
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)

class SimpleNN(nn.Module):


 def __init__(self, num_features, num_labels, hidden_dim):
    super(SimpleNN, self).__init__()
    self.fc1 = nn.Linear(num_features, hidden_dim)  # primo hidden layer
    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    self.fc3 = nn.Linear(hidden_dim, hidden_dim)
    self.fc4 = nn.Linear(hidden_dim, num_labels)  # Output layer
    self.dropout = nn.Dropout(p=0.2)


 def forward(self, x):
    x = self.fc1(x)  # Primo layer
    x = F.relu(x)  # ReLU dopo il primo layer
    x = self.dropout(x)

    x = self.fc2(x)  # Secondo layer
    x = F.relu(x)
    x = self.dropout(x)

    x = self.fc3(x)
    x = F.relu(x)
    x = self.dropout(x)

    x = self.fc4(x)  # Output finale
    return x


def get_tensor_dataset(x, y):
    x_tensor = torch.FloatTensor(x)
    y_tensor = torch.LongTensor(y)

    tensor_dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    return tensor_dataset

train_dataset = get_tensor_dataset(x_train, y_train)
val_dataset = get_tensor_dataset(x_val, y_val)
test_dataset = get_tensor_dataset(x_test, y_test)

# dataloader per caricare i dati in mini bach nella fase di training
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(x_val), shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(x_test), shuffle=False)

def train_model(model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_loss = float('inf')

    def test_prediction(test_input):
        model.eval()
        with torch.no_grad():
            output = model(test_input)
            probabilities = F.softmax(output, dim=1)
            return probabilities

    test_batch = next(iter(test_loader))
    test_inputs, test_labels = test_batch
    test_inputs = test_inputs.to(device)


    for epoch in range(1, num_epochs + 1):
        # Training Phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for train_inputs, train_labels in train_loader:
            train_inputs, train_labels = train_inputs.to(device), train_labels.to(device)
            optimizer.zero_grad()
            train_preds = model(train_inputs)
            loss = criterion(train_preds, train_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * train_inputs.size(0)
            train_correct += (train_preds.argmax(dim=1) == train_labels).sum().item()
            train_total += train_inputs.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                val_preds = model(val_inputs)
                loss = criterion(val_preds, val_labels)

                val_loss += loss.item() * val_inputs.size(0)
                val_correct += (val_preds.argmax(dim=1) == val_labels).sum().item()
                val_total += val_inputs.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        if epoch % log_every == 0:
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{now}] Epoch {epoch} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        #salvataggio
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(save_dir, 'model1.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, model_path)
            print(f"Nuovo miglior modello salvato Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        scheduler.step()

    #fase test
    model_reloaded = SimpleNN(num_features, num_labels, hidden_dim=hidden_dim)
    resume = torch.load(model_path, map_location=device)
    checkpoint = torch.load(model_path)
    model_reloaded.load_state_dict(checkpoint['model_state_dict'])  # ?
    model_reloaded.eval()

    for batch in test_loader:
        with torch.no_grad():
            test_inputs, test_labels = batch
            test_preds = model_reloaded(test_inputs)

            test_preds = torch.log_softmax(test_preds, dim=1)
            _, test_preds = torch.max(test_preds, dim=1)
            test_acc = torch.sum(test_preds == test_labels.data)

        print('Accuracy on test set: ' + str(test_acc.item() / len(test_labels)))


    return model