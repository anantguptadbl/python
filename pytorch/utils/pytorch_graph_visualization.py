# VISUALIZE THE Pytorch Graph

# sudo apt-get install graphviz
# pip install graphviz

from torchviz import make_dot


# Model Object
# Prediction from the model
XOut = model(X)

make_dot(XOut, params=dict(list(model.named_parameters()))).render("model_graph_image", format="png")

