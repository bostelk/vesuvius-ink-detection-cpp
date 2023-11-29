# vesuvius-ink-detection-cpp
ONNX C/C++ Runtime to detect ink in Vesuvius scrolls. Is a work in progress / not yet correct.

To export ONNX model from PyTorch:
  1. Git clone https://github.com/younader/Vesuvius-First-Letters.
  2. Edit "inference.py" after model.eval statement.

```
torch_input = torch.randn(1, 1, 32, 32) // Todo(kbostelmann): Use dynamic input shape?
onnx_program = torch.onnx.dynamo_export(model, torch_input)
onnx_program.save("my_image_classifier.onnx")
```

   3. Install python packages and run script.

You must run on Linux because dynamo is not supported on Windows. You must also edit "i3dall.py" or an unsupported class list error is thrown:

```
    def forward(self, x):
       if self.forward_features:
            features=[]
            x = self._modules['Conv3d_1a_7x7'](x)
            x = self._modules['MaxPool3d_2a_3x3'](x)
            x = self._modules['Conv3d_2b_1x1'](x)
            x = self._modules['Conv3d_2c_3x3'](x)
            features.append(x)
            x = self._modules['MaxPool3d_3a_3x3'](x)
            x = self._modules['Mixed_3b'](x)
            x = self._modules['Mixed_3c'](x)
            features.append(x)
            x = self._modules['MaxPool3d_4a_3x3'](x)
            x = self._modules['Mixed_4b'](x)
            x = self._modules['Mixed_4c'](x)
            x = self._modules['Mixed_4d'](x)
            x = self._modules['Mixed_4e'](x)
            x = self._modules['Mixed_4f'](x)
            features.append(x)
            x = self._modules['MaxPool3d_5a_2x2'](x)
            x = self._modules['Mixed_5b'](x)
            x = self._modules['Mixed_5c'](x)
            features.append(x)
            return features
```
