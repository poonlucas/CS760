import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# Custom NN
class NN:
    def __init__(self, w1, w2):
        self.w1 = w1
        self.w2 = w2

    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))

    def softmax(self, x):
        return torch.exp(x) / torch.sum(torch.exp(x), dim=0)

    def train(self, train_loader, e=1, lr=0.01):
        accuracies = []
        losses = []

        for epoch in range(e):
            # running loss
            running_loss = 0.0
            # correct
            correct = 0
            for i, (images, labels) in enumerate(train_loader):
                # forward pass
                image = torch.flatten(images, start_dim=1)
                z1 = torch.matmul(self.w1, image.t())
                a = self.sigmoid(z1)
                z2 = torch.matmul(self.w2, a)
                # outputs
                g = self.softmax(z2)

                # backward pass
                y = torch.nn.functional.one_hot(labels, num_classes=10).t()
                dz2 = g - y
                dW2 = torch.matmul(dz2, a.t())
                da = torch.matmul(self.w2.t(), dz2)
                dz1 = da * a * (1 - a)
                dW1 = torch.matmul(dz1, image)

                self.w2 = self.w2 - lr * dW2
                self.w1 = self.w1 - lr * dW1

                celoss = -torch.sum(y * torch.log(g))
                running_loss += celoss * len(images)

                predictions = torch.argmax(g, dim=0)
                for idx, pred in enumerate(predictions):
                    if pred == labels[idx]:
                        correct += 1

            accuracy = (100 * correct) / len(train_loader.sampler)
            accuracies.append(accuracy)
            loss = running_loss / len(train_loader.sampler)
            losses.append(loss)
            print(f"Train Epoch: {epoch} "
                  f"Accuracy: {correct}/{len(train_loader.sampler)}({accuracy:.3f}%) "
                  f"Loss: {loss}")
        return accuracies, losses

    def test(self, test_loader):
        error = 0
        for data in test_loader:
            image, label = data
            image = torch.flatten(image, start_dim=1)
            z1 = torch.matmul(self.w1, image.t())
            a = self.sigmoid(z1)
            z2 = torch.matmul(self.w2, a)
            # outputs
            g = self.softmax(z2)
            predictions = torch.argmax(g, dim=0)
            for idx, pred in enumerate(predictions):
                if pred != label[idx]:
                    error += 1
        print(f"Test Error: {error}")


# Pytorch NN
def build_model():
    model = nn.Sequential(
        # Flatten 2d to 1d array
        nn.Flatten(),
        # Linear with 300 nodes with Sigmoid activation
        nn.Linear(784, 300, bias=False),
        nn.Sigmoid(),
        # Linear with 10 nodes
        nn.Linear(300, 10, bias=False),
    )
    # model.apply(init_weights)
    return model


def train_model(model, train_loader, criterion, episodes):
    # Optimizer
    opt = optim.SGD(model.parameters(), lr=0.01)
    # Set model to train mode
    model.train()

    accuracies = []
    losses = []

    for epoch in range(episodes):

        # running loss
        running_loss = 0.0
        # correct
        correct = 0

        # train model
        for i, (images, labels) in enumerate(train_loader, 0):
            # zero the parameter gradients
            opt.zero_grad()

            # forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()

            # multiply by batch size as loss.item() gives average
            running_loss += loss.item() * len(images)  # len(images) gives the same as labels.size(0)

            # correct
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            # print statistics

        accuracy = (100 * correct) / len(train_loader.sampler)
        loss = running_loss / len(train_loader.sampler)
        accuracies.append(accuracy)
        losses.append(loss)
        print(f"Train Epoch: {epoch} "
              f"Accuracy: {correct}/{len(train_loader.sampler)}({accuracy:.3f}%) "
              f"Loss: {loss}")
    return accuracies, losses


def evaluate_model(model, test_loader, criterion):
    model.eval()
    # correct
    correct = 0
    # total
    total = 0
    # test loss
    test_loss = 0.0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            loss = criterion(outputs, labels)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # loss
            # multiply by batch size as loss.item() gives average
            test_loss += loss.item() * images.size(0)
    print(f"Average loss: {test_loss / len(test_loader.sampler)}")
    print(f"Accuracy: {correct}/{len(test_loader.sampler)}({(100 * correct) / total}%)")


def init_weights_zero(m):
    if type(m) == nn.Linear:
        m.weight.data.fill_(0)


def init_weights_uniform(m):
    if type(m) == nn.Linear:
        m.weight.data.uniform_(-1, 1)


if __name__ == '__main__':
    mnist_train = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
    mnist_test = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())

    batch_size = 64

    train_set = torch.utils.data.DataLoader(mnist_train, batch_size=32)
    test_set = torch.utils.data.DataLoader(mnist_test, batch_size=len(mnist_test))

    # # Q4.2
    # nn = NN((-2) * torch.rand(300, 784) + 1, (-2) * torch.rand(10, 300) + 1)
    # accuracy, loss = nn.train(train_set, 5)
    # nn.test(test_set)
    # episodes = [1, 2, 3, 4, 5]
    #
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig.suptitle(f"Learning Curve - Accuracy and Loss")
    #
    # ax1.title.set_text("Loss")
    # ax1.plot(episodes, loss)
    # ax1.set_xlabel("Episode")
    # ax1.set_ylabel("Loss")
    # ax1.grid()
    #
    # ax2.title.set_text("Accuracy")
    # ax2.plot(episodes, accuracy)
    # ax2.set_xlabel("Episode")
    # ax2.set_ylabel("Accuracy (%)")
    # ax2.grid()
    #
    # plt.show()

    # # Q4.3
    # model = build_model()
    # criterion = nn.CrossEntropyLoss()
    # accuracy, loss = train_model(model, train_set, criterion, 5)
    # evaluate_model(model, test_set, criterion)
    # episodes = [1, 2, 3, 4, 5]
    #
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # fig.suptitle(f"Learning Curve - Accuracy and Loss")
    #
    # ax1.title.set_text("Loss")
    # ax1.plot(episodes, loss)
    # ax1.set_xlabel("Episode")
    # ax1.set_ylabel("Loss")
    # ax1.grid()
    #
    # ax2.title.set_text("Accuracy")
    # ax2.plot(episodes, accuracy)
    # ax2.set_xlabel("Episode")
    # ax2.set_ylabel("Accuracy (%)")
    # ax2.grid()
    #
    # plt.show()

    # # Q4.4
    model1 = build_model()
    model1.apply(init_weights_zero)
    criterion = nn.CrossEntropyLoss()
    accuracy1, loss1 = train_model(model1, train_set, criterion, 5)
    evaluate_model(model1, test_set, criterion)

    model2 = build_model()
    model2.apply(init_weights_uniform)
    criterion = nn.CrossEntropyLoss()
    accuracy2, loss2 = train_model(model2, train_set, criterion, 5)
    evaluate_model(model2, test_set, criterion)

    episodes = [1, 2, 3, 4, 5]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(f"Learning Curve - Accuracy and Loss")

    ax1.title.set_text("Loss")
    ax1.plot(episodes, loss1)
    ax1.plot(episodes, loss2)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Loss")
    ax1.legend(["Zero initial weights", "Random from -1 to 1 weights"])
    ax1.grid()

    ax2.title.set_text("Accuracy")
    ax2.plot(episodes, accuracy1)
    ax2.plot(episodes, accuracy2)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend(["Zero initial weights", "Random from -1 to 1 weights"])
    ax2.grid()

    plt.show()
    # nn = NN(torch.zeros(300, 784), torch.zeros(10, 300))
    # batch_loss = nn.train(train_set, 5)
    # batch_counts = [(idx + 1) for idx, _ in enumerate(batch_loss)]
    # nn.test(test_set)
    #
    # plt.plot(batch_counts, batch_loss)
    #
    # nn2 = NN((-2) * torch.rand(300, 784) + 1, (-2) * torch.rand(10, 300) + 1)
    # batch_loss2 = nn2.train(train_set, 5)
    # batch_counts2 = [(idx2 + 1) for idx2, _ in enumerate(batch_loss2)]
    # nn2.test(test_set)
    #
    # plt.plot(batch_counts2, batch_loss2)
    # plt.title(f"Learning Curve - Loss per batch (batch size = {batch_size})")
    # plt.xlabel("Batch")
    # plt.ylabel("Loss")
    # plt.grid()
    # plt.legend(loc='upper right', labels=["Zero initial weights", "Random from -1 to 1 weights"])
    # plt.show()
