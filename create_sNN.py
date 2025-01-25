import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

#createing vars
datasetSize = 100
categorySize = 2
inputDimensions = 1
hiddenLayerSize = 20
outputClasses = 2
learningRate = 0.001
lossIndex = []
passes = 100
input = []
output = []

for i in range(datasetSize):
    number = random.random()
    input.append(number)
    if number < 0.5:
        output.append(0)
    else:
        output.append(1)

input = torch.tensor(input)
output = torch.tensor(output)

g = torch.Generator().manual_seed(420)
W1 = torch.randn((inputDimensions, hiddenLayerSize), generator=g) * (5.0/3.0) / (inputDimensions * hiddenLayerSize ** 0.5)
W1.requires_grad = True

W2 = torch.randn((hiddenLayerSize, outputClasses), generator=g, requires_grad=True)
print(W1.shape, W2.shape)

for i in range(passes):
    #forward pass
    indexed = random.randrange(datasetSize)
    currentInput = input[indexed]
    currentInput = currentInput.view(1,1)
    hiddenInput = currentInput @ W1
    hiddenOutput = torch.tanh(hiddenInput)
    logits = hiddenOutput @ W2
    currentOutput = output[indexed].view(1)
    
    #comput loss
    loss = F.cross_entropy(logits, currentOutput)
    W1.grad = None
    W2.grad = None
    loss.backward()
    
    #updating weights
    W1.data += -learningRate * W1.grad
    W2.data += -learningRate * W2.grad
    
    if i % 1000 == 0:
        print(loss.item())
    lossIndex.append(loss.item())

plt.plot(lossIndex)
plt.show()

for i in range(10):
    inferenceInput = random.randrange(datasetSize)
    currentInput = input[inferenceInput].view(1,1)
    hiddenInput = currentInput @ W1
    hiddenOutput = torch.tanh(hiddenInput)
    logits = hiddenOutput @ W2
    probs = F.softmax(logits, dim=1)
    currentOutput = output[inferenceInput].view(1)
    
    correctAnswer = False
    if(probs[:,0].item() > probs[:,1].item()):
        if currentOutput.item() == 0:
            correctAnswer = True
    else:
        if currentOutput.item() == 1:
            correctAnswer = True
    print('input: ', round(currentInput.item(), 2), '| predict 0: ', round(probs[:,0].item(), 2), '| predict 1: ', round(probs[:,1].item(), 2), '| target ouput: ', currentOutput.item(), '| correct: ', correctAnswer)
    
    
    