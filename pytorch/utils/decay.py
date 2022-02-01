# CUSTOM NEURON

class DecayNeuron(nn.Module):
    def __init__(self):
        super(DecayNeuron, length, self).__init__()
        self.weights = nn.Parameter(torch.rand(length))
        self.bias = nn.Parameter(torch.rand(1))
        # 2 rest iterations
        self.current_active = torch.zeros(length, 3)
        self.init_positions = np.array([[i,x] for i,x in enumerate(np.random.randint(3, size=(10, )))])
        self.current_active[self.init_positions[:, 0], self.init_positions[:, 1]] = 1
        
    def forward(self, x):
        x = torch.matmul(x, current_active[:, 2] * self.weights) + self.bias
        # We will roll to make it inactive for the next two iterations
        self.current_active = torch.roll(self.current_active, 1, 1)
        return x
        
class DecayLinear(nn.Module):
    def __init__(self):
        super(DecayLinear, length, self).__init__()
        self.neurons = 
