import torch
from src.models.model import MyAwesomeConvolutionalModel

def test_output_shape():
    input = 50
    input_tensor = torch.ones((input,1,28,28))
    model = MyAwesomeConvolutionalModel()
    output_shape = torch.ones(input,10).shape
    output = model(input_tensor)
    assert output_shape == output.shape



if __name__ == '__main__':
    print("Test 1")
    test_output_shape()
