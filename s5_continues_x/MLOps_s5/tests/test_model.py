import torch
import pytest
from src.models.model import MyAwesomeConvolutionalModel

@pytest.mark.parametrize("test_input,expected", [(3+5, 8), (2+4, 6), (6*7, 42)])
def test_output_shape(test_input,expected):
    input = test_input
    input_tensor = torch.ones((test_input,1,28,28))
    model = MyAwesomeConvolutionalModel()
    output_shape = torch.ones(expected,10).shape
    output = model(input_tensor)
    assert output_shape == output.shape

#def test_error_on_wrong_shape():
#   with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
#        model = MyAwesomeConvolutionalModel()
#        model(torch.randn(10,1,28))


#if __name__ == '__main__':
#    print("Test 1")
#    test_output_shape()
#
#    print("Test 2")
#    test_error_on_wrong_shape()
