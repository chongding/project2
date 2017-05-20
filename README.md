# project2
CNN, a must read://cs231n.github.io/convolutional-networks/

pay attention to memory size (e.g. a vgg net 93MB/img in fwd, and ~200MB/img in bwd), smaller batch size helps
weight initilization made a big difference in the results:mean=0.01, stddev=0.1

~~~~~~~~~Conv Layer~~~~~~~~~ from cs231n~~~~~~~~
Accepts a volume of size W1×H1×D1
Requires four hyperparameters:
Number of filters K,
their spatial extent F,
the stride S,
the amount of zero padding P.
Produces a volume of size W2×H2×D2 where:
W2=(W1−F+2P)/S+1
H2=(H1−F+2P)/S+1 (i.e. width and height are computed equally by symmetry)
D2=K
With parameter sharing, it introduces F⋅F⋅D1 weights per filter, for a total of (F⋅F⋅D1)⋅K weights and K biases.
In the output volume, the dd-th depth slice (of size W2×H2) is the result of performing a valid convolution of the d-th filter over the input volume with a stride of S, and then offset by d-th bias.
~~~~~~~~~~Pooling Lyer  from cs231n~~~~~~~~
It is common to periodically insert a Pooling layer in-between successive Conv layers in a ConvNet architecture. Its function is to progressively reduce the spatial size of the representation to reduce the amount of parameters and computation in the network, and hence to also control overfitting
