import torch
import torchvision
from torch import nn


class Cat_Dog_CNN(nn.Module):
    def __init__(self,
                 in_channels : int,
                 out_channels : int ,
                 num_conv_layers : int,
                 activation_function : int,
                 normalization: bool,
                 num_classifications : int,
                 ) -> None:
        super(Cat_Dog_CNN, self).__init__()
        
        
        self.layers = []
        for i in range(num_conv_layers,0, -1):
            if i == num_conv_layers:
                self.layers.append(nn.Conv2d(in_channels=in_channels, out_channels=(2**(i-1))*out_channels, kernel_size=4, padding=1, stride=2))
            else:
                self.layers.append(nn.Conv2d(in_channels=(2**(i))*out_channels, out_channels=(2**(i-1))*out_channels, kernel_size=4, padding=1, stride=2))
            if normalization:
                self.layers.append(nn.BatchNorm2d((2**(i-1))*out_channels))
            self.layers.append(activation_function)
        
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(2**(11-num_conv_layers)*out_channels, num_classifications))
        self.layers.append(nn.Softmax(dim=1))
        self.main = nn.Sequential(*self.layers)
        
    def forward(self, input_img: torch.Tensor):
        return self.main(input_img)



if  __name__ == "__main__":
    activation_func = nn.ReLU()
    model = Cat_Dog_CNN(in_channels=3,out_channels=3,num_conv_layers=5,activation_function=activation_func, normalization=True, num_classifications=2)

    def print_shape_hook(module, input, output):
        print(f"{module.__class__.__name__} output shape: {output.shape}")
    print('layers: ', model.layers)
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    
    for layer in model.children():
        layer.register_forward_hook(print_shape_hook)

    sample_input = torch.randn(8,3,256,256)
    
    model(sample_input)
    