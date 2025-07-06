from onnx import TensorProto
from onnx.checker import check_model
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info

# inputs

# 'X' is the name, TensorProto.FLOAT the type, [None, None] the shape
X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
A = make_tensor_value_info("A", TensorProto.FLOAT, [None, None])
B = make_tensor_value_info("B", TensorProto.FLOAT, [None, None])

# outputs, the shape is left undefined

Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])

# nodes

# It creates a node defined by the operator type MatMul,
# 'X', 'A' are the inputs of the node, 'XA' the output.
node1 = make_node("MatMul", ["X", "A"], ["XA"])
node2 = make_node("Add", ["XA", "B"], ["Y"])

# from nodes to graph
# the graph is built from the list of nodes, the list of inputs,
# the list of outputs and a name.

graph = make_graph([node1, node2], "lr", [X, A, B], [Y])  # nodes  # a name  # inputs  # outputs

# onnx graph
# there is no metadata in this case.

onnx_model = make_model(graph)

# Let's check the model is consistent,
# this function is described in section
# Checker and Shape Inference.
check_model(onnx_model)

# the work is done, let's display it...
print(onnx_model)
