import torch
import torch.nn
import onnx
model = torch.load('docs/last.pt')
model.eval()
input_names = ['input']
output_names = ['output']
x = torch.randn(1, 3, 640, 640, requires_grad=True)
torch.onnx.export(model, x, 'docs/best.onnx', input_names=input_names, output_names=output_names, verbose='True')