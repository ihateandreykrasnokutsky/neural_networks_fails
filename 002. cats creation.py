from PIL import Image
from skimage.io import imread
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import warnings
import random
warnings.filterwarnings("ignore", category=UserWarning, module="skimage") #to remove warnings about "low contast image" (because early outputs are always low contrast)
np.set_printoptions(threshold=np.inf)

compr_size=64 #the size of the compressed image
bs=11 #batch size
hid_nrn=50 #the number of hidden neurons
epochs=50000 
lr=0.1 #learning rate
clip=10 #clipping of too high values
loss_threshold=1e-10 #loss thershold, after which learning stops (I doesn't really use it here)
i=random.randint(0,bs-1) #a random number to choose a random input and target

def sum_pool (matrix): #reduces the dimensionality from bs images to 1 (I don't use it here)
    matrix=np.sum(matrix, axis=0, keepdims=True)
    return matrix 

def image_to_matrix (image_path, size=(compr_size,compr_size)): #convert image to a flat matrix
    img=imread(image_path, as_gray=True)
    img=resize(img, size, anti_aliasing=True, preserve_range=True)
    return img.flatten()

def matrix_to_image (image_path, matrix, size): #convert matrix to image and save it
    matrix=matrix*255
    matrix=matrix.reshape(size)
    matrix=matrix.astype(np.uint8)
    imsave(image_path, matrix)
    return 0

def leaky_relu(x, alpha=0.05): #activation
    return np.where(x>0, x, x*alpha)
def leaky_relu_derivative(x, alpha=0.05): #derivative of activation
    return np.where(x>0, 1, alpha)
def sigmoid(x): #last layer's activation
    return 1/(1+np.exp(-x))
def sigmoid_derivative(x): #derivative
    return (sigmoid(x)*(1-sigmoid(x)))

cats=[image_to_matrix(f"/home/amk/Pictures/machine_learning/cats_data/cat_{i}.png") for i in range(bs)] #input data
cats_faces=[image_to_matrix(f"/home/amk/Pictures/machine_learning/cats_data/cat_face_{i}.png") for i in range(bs)] #target (label) data

w1=np.random.randn(compr_size**2, hid_nrn)*np.sqrt(2/(compr_size**2)) #weights and biases, the numbers are reduced to avoid stack overflow
b1=np.zeros((1, hid_nrn))
w2=np.random.randn(hid_nrn, hid_nrn)*np.sqrt(2/hid_nrn)
b2=np.zeros((1,hid_nrn))
w3=np.random.randn(hid_nrn, hid_nrn)*np.sqrt(2/hid_nrn)
b3=np.zeros((1,hid_nrn))
w4=np.random.randn(hid_nrn, hid_nrn)*np.sqrt(2/hid_nrn)
b4=np.zeros((1,hid_nrn))
w5=np.random.randn(hid_nrn, hid_nrn)*np.sqrt(2/hid_nrn)
b5=np.zeros((1,hid_nrn))
w6=np.random.randn(hid_nrn, hid_nrn)*np.sqrt(2/hid_nrn)
b6=np.zeros((1,hid_nrn))
w7=np.random.randn(hid_nrn, hid_nrn)*np.sqrt(2/hid_nrn)
b7=np.zeros((1,hid_nrn))
w8=np.random.randn(hid_nrn, hid_nrn)*np.sqrt(2/hid_nrn)
b8=np.zeros((1,hid_nrn))
w9=np.random.randn(hid_nrn, hid_nrn)*np.sqrt(2/hid_nrn)
b9=np.zeros((1,hid_nrn))
w10=np.random.randn(hid_nrn, compr_size**2)*np.sqrt(2/hid_nrn)
b10=np.zeros((1,compr_size**2))
for epoch in range(epochs):
    
    #forward pass
    x=np.array(cats[i]).reshape(1,-1) #choosing 1 picture from the input data
    y=np.array(cats_faces[i]).reshape(1,-1) #same from the label data
    z1=np.clip(x@w1+b1, -clip, clip)
    a1=leaky_relu(z1)
    z2=np.clip(a1@w2+b2, -clip, clip)
    a2=leaky_relu(z2)
    z3=np.clip(a2@w3+b3, -clip, clip)
    a3=leaky_relu(z3)
    z4=np.clip(a3@w4+b4, -clip, clip)
    a4=leaky_relu(z4)
    z5=np.clip(a4@w5+b5, -clip, clip)
    a5=leaky_relu(z5)
    z6=np.clip(a5@w6+b6, -clip, clip)
    a6=leaky_relu(z6)
    z7=np.clip(a6@w7+b7, -clip, clip)
    a7=leaky_relu(z7)
    z8=np.clip(a7@w8+b8, -clip, clip)
    a8=leaky_relu(z8)
    z9=np.clip(a8@w9+b9, -clip, clip)
    a9=leaky_relu(z9)
    z10=np.clip(a9@w10+b10, -clip, clip)
    a10=sigmoid(z10)
    
    #backpropagation
    l=np.mean(np.square(a10-y)) #loss function
    z10grad=(2*(a10-y))*sigmoid_derivative(z10)
    z9grad=z10grad@w10.T*leaky_relu_derivative(z9)
    z8grad=z9grad@w9.T*leaky_relu_derivative(z8)
    z7grad=z8grad@w8.T*leaky_relu_derivative(z7)
    z6grad=z7grad@w7.T*leaky_relu_derivative(z6)
    z5grad=z6grad@w6.T*leaky_relu_derivative(z5)
    z4grad=z5grad@w5.T*leaky_relu_derivative(z4)
    z3grad=z4grad@w4.T*leaky_relu_derivative(z3)
    z2grad=z3grad@w3.T*leaky_relu_derivative(z2)
    z1grad=z2grad@w2.T*leaky_relu_derivative(z1)
    w10grad=a9.T@z10grad
    w9grad=a8.T@z9grad
    w8grad=a7.T@z8grad
    w7grad=a6.T@z7grad
    w6grad=a5.T@z6grad
    w5grad=a4.T@z5grad
    w4grad=a3.T@z4grad
    w3grad=a2.T@z3grad
    w2grad=a1.T@z2grad
    w1grad=x.T@z1grad
    b10grad=np.sum(z10grad, axis=0, keepdims=True)
    b9grad=np.sum(z9grad, axis=0, keepdims=True)
    b8grad=np.sum(z8grad, axis=0, keepdims=True)
    b7grad=np.sum(z7grad, axis=0, keepdims=True)
    b6grad=np.sum(z6grad, axis=0, keepdims=True)
    b5grad=np.sum(z5grad, axis=0, keepdims=True)
    b4grad=np.sum(z4grad, axis=0, keepdims=True)
    b3grad=np.sum(z3grad, axis=0, keepdims=True)
    b2grad=np.sum(z2grad, axis=0, keepdims=True)
    b1grad=np.sum(z1grad, axis=0, keepdims=True)
    
    #updating w and b
    w1-=lr*np.clip(w1grad*lr, -clip, clip)
    w2-=lr*np.clip(w2grad*lr, -clip, clip)
    w3-=lr*np.clip(w3grad*lr, -clip, clip)
    w4-=lr*np.clip(w4grad*lr, -clip, clip)
    w5-=lr*np.clip(w5grad*lr, -clip, clip)
    w6-=lr*np.clip(w6grad*lr, -clip, clip)
    w7-=lr*np.clip(w7grad*lr, -clip, clip)
    w8-=lr*np.clip(w8grad*lr, -clip, clip)
    w9-=lr*np.clip(w9grad*lr, -clip, clip)
    w10-=lr*np.clip(w10grad*lr, -clip, clip)
    b1-=lr*np.clip(b1grad*lr, -clip, clip)
    b2-=lr*np.clip(b2grad*lr, -clip, clip)
    b3-=lr*np.clip(b3grad*lr, -clip, clip)
    b4-=lr*np.clip(b4grad*lr, -clip, clip)
    b5-=lr*np.clip(b5grad*lr, -clip, clip)
    b6-=lr*np.clip(b6grad*lr, -clip, clip)
    b7-=lr*np.clip(b7grad*lr, -clip, clip)
    b8-=lr*np.clip(b8grad*lr, -clip, clip)
    b9-=lr*np.clip(b9grad*lr, -clip, clip)
    b10-=lr*np.clip(b10grad*lr, -clip, clip)
    
    if epoch%1000==0: print (f"epoch: {epoch}, loss {l}") #print epochs and loss each 1000 epochs
    if epoch==epochs-1: print (f"epoch: {epoch}, loss {l}") #print epoch and loss at the end of the cycle
    if l<loss_threshold: #stop learning, if the loss is small enough
        print (f"epoch: {epoch}, loss {l}, BREAK")
        break
    if epoch%5000==0 or epoch==0 or epoch==epochs-1: #each 5000 epochs save the the learned picture
        alearning=a10
        matrix_to_image(f"/home/amk/Pictures/machine_learning/cats_data/learning {epoch} output, loss {l}.png", alearning, size=(compr_size,compr_size))
    if l<1e-4: i=random.randint(0,bs-1) #if the picture is learned well, switch to another one
    
#inference
i+=1 #switch to the next input picture
if i>bs-1: i=0 #if the picture is the last one, switch to the first one
tx=image_to_matrix(f"/home/amk/Pictures/machine_learning/cats_data/cat_{i}.png")
tz1=tx@w1+b1                                      
ta1=leaky_relu(tz1)
tz2=ta1@w2+b2
ta2=leaky_relu(tz2)
tz3=ta2@w3+b3
ta3=leaky_relu(tz3)
tz4=ta3@w4+b4
ta4=leaky_relu(tz4)
tz5=ta4@w5+b5
ta5=leaky_relu(tz5)
tz6=ta5@w6+b6
ta6=leaky_relu(tz6)
tz7=a6@w7+b7
ta7=leaky_relu(tz7)
tz8=ta7@w8+b8
ta8=leaky_relu(tz8)
tz9=ta8@w9+b9
ta9=leaky_relu(tz9)
tz10=np.clip((ta9@w10+b10), -10, 10)
ta10=sigmoid(tz10)

matrix_to_image(f"/home/amk/Pictures/machine_learning/cats_data/inference_input.png", tx, size=(compr_size,compr_size)) #inference input
matrix_to_image(f"/home/amk/Pictures/machine_learning/cats_data/inference_output.png", ta10, size=(compr_size,compr_size)) #inference output

#MY COMMENTS DURING WRITING OF THIS PROGRAM

#comment 1 (10 different images as input => 10 same images as output/label)
#THE PROGRAM CREATES AN AVERAGE PICTURE OF ALL PICTURES (GEOMETRICALLY), SO IT'S NOT VERY FUN ACTUALLY.
#ALSO I WASN'T ABLE TO MAKE IT FULLY GRAYSCALE, IT'S JUST BLACK AND WHITE

#comment 2 (1 image as input, 1 image as output/label)
#The program (if it has a 300x300 pixel image and about 10 neurons)
#creates the desired picture by erasing the previous one and painting the label.
#Very smart and boring.

#comment 3
#My idea for the next day is to transform 10 pictures into 1, then to add more pictures of cats.
#I doubt that anything good will appear from this (except blurry geometrical average picture),
# but it worth a try, especially with big number of neurons+high resolution+big number of inputs.
#The average of the many pictures may be suprisingly good.
#check the file "cats 9 dimensionality reduction (especially options 1 and 3)".

#comment 4 (adding pooling to reduce dimensionality)

#comment 5
#I added pooling, works great.
#Now I want to randomize the input each 10 epochs (shuffle the 10 images),
#plus randomize the target to make the neural network extract essential feautres, not just the exact picture

#comment 6
#Fixed input and random targets didn't work good.
#Next time try random input (for each epoch) + fixed target
#(important: the target shouldn't be in the input set).
#though there's a chance that the network will just erase input, painting the needed image instead.
#So, I think it's better to use 2-5 decent cat images as random targets + fixed input.

#comment 7
#I think I need the neural network to see the dependence between input and target, so I need the target
#to change from 0 to bs (batch size) regularly, not randomly. At least, it's my best bet for now.

#comment 8
#A better idea! Make 2 selections of images: 1st is cats (your usual batch), 2nd is same, but only faces!
#And the faces should be placed in the center, unlike in the original images! And maybe increased in size!

#comment 9
#the program doesn't work with a completly new inference input
#so I need to check if it will work with a known input (but still different from a last one)
#but still, it probably won't work:(

#comment 10   
#So it works well, if the inference input is from the learning batch (but not the last image from the learning cycle),
#so the reason it doesn't work on unseen images is the small amount of data.
#I need either more data or dropout.
#On the other hand, the inference output is not really connected with the inference input, so the program still can't recognize the
#spatial structure of images. It likely just relearns the weights and biases from 0 each time it has a new image. So I need a convolutional NN.

#The training data and the output images are in the archive "cats_data.zip"
