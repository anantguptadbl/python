# Adverserial FSGM
# Inspiration
#https://www.tensorflow.org/tutorials/generative/adversarial_fgsm

vgg16 = models.vgg16(pretrained=True)
labrador=plt.imread("gdrive/My Drive/Colab Notebooks/YellowLabradorLooking_new.jpg")
labrador=labrador/255.0
import cv2
labrador=Variable(torch.from_numpy(cv2.resize(labrador,(224,224)).reshape(1,3,224,224).astype(np.float32)),requires_grad=True)
predictedLabels=vgg16(labrador)
criterion=nn.CrossEntropyLoss()
loss=criterion(predictedLabels,Variable(torch.from_numpy(np.array([208])).type(torch.LongTensor)))
print("The loss is {0}".format(loss))
grads=torch.autograd.grad(loss,labrador)[0]
grads=torch.sign(grads[0])

# Image of the perturbations
plt.imshow((grads.detach().numpy().reshape(224,224,3)*255).astype(np.int))

for epsilon in [0.001,0.01,0.1,0.15]:
  adversarialImage = labrador + epsilon*grads
  #print(epsilon*grads)
  adversarialImage = torch.clamp(adversarialImage, 0.0, 1.0)
  plt.figure()
  predictedLabels=vgg16(adversarialImage)
  loss=criterion(predictedLabels,Variable(torch.from_numpy(np.array([208])).type(torch.LongTensor)))
  print("The loss is {0}".format(loss))
  plt.imshow((adversarialImage.detach().numpy().reshape(224,224,3)*255).astype(np.int),label="epsilon = {0}".format(epsilon))
  
print("Cell Execution Completed")
