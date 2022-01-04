from torch import nn
import torch
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, input, output, hidden):
        super().__init__()
        self.fc1 = nn.Linear(input, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], hidden[2])
        self.fc4 = nn.Linear(hidden[2], output)
        
        # Dropout module with 0.2 drop probability
        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        # Now with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        
        # output so no dropout here
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x

def train(model, trainloader, testloader, criterion, optimizer, epochs):

    steps = 0
    train_losses, test_losses = [], []
    for e in range(epochs):
        running_loss = 0
        model.train()
        for images, labels in trainloader:
            
            optimizer.zero_grad()
            
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        else:
            with torch.no_grad():
                model.eval()
                accuracy = 0
                for images, labels in testloader:
                    output = model(images)
                    top_p, top_class = output.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy = torch.mean(equals.type(torch.FloatTensor))
                print(f'Loss: {running_loss}     Accuracy: {accuracy.item()*100}%')
