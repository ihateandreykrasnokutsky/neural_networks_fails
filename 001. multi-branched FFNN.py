#THE PROBLEM OF THIS MULTI-BRANCH NEURAL NETWORK IS IN THE FACT THAT ONLY THE INPUT LAYER (THAT SHOULD BE MADE, AND THAT SHOULD COMPRESS DATA) DOES SORTING, THE FURTHER LAYERS JUST MAKE THE SORTED DATA EQUAL TO THE GUN LABEL
#I CAN DO THE NETWORK MULTI-BRANCHED, WHERE EACH BRANCH WILL DO CLASSIFICATION, BUT IT WON'T BE PROFOUNDLY DIFFERENT FROM THE ONE-BRANCH NEURAL NETWORK
import numpy as np
from PIL import Image
np.set_printoptions(threshold=np.inf)

bs=20 #batch size
t_bs=3 #test batch size
cmp_size=100 #compressed image's size
lr=0.01 #learning rate
epochs=1000 #number of training epochs
hid_nrn=10 #number of hidden neurons
loss_threshold=1e-5 #threshold, after which training stops
clip_low=-1
clip_high=1
pistol_index=0.1 #a number for each gun type, that program will use instead of a word
rifle_index=0.1
shotgun_index=0.1
bow_index=0.1
no_id=0


def leaky_relu (x, alpha=0.05):
    return np.where (x>0, x, alpha*x)
def leaky_relu_derivative (x, alpha=0.05):
    return np.where (x>0, 1, alpha)
def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))
def scaled_sigmoid(x,c): #I invented the scaled sigmoid to do the classification between 0 and c (that can be 0,1,2,3,4, depending on the type of gun.)
    c+=1e-20 #to avoid calculation errors when gun type = 0
    return c/(c+np.exp(-x))
def scaled_sigmoid_derivative(x,c): #I just copied the derivative made by ChatGPT
    c+=1e-20 #to avoid calculation errors when gun type = 0
    return (c*np.exp(-x))/((c+(np.exp(-x))**2))
def binary_cross_entropy(y_true, y_pred, epsilon=1e-12):
    y_pred=np.clip(y_pred, epsilon, y_pred-epsilon)
    loss=-np.mean(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))
    loss=np.clip(loss,-1,1) #cut the loss to avoid stack overflow and nan
    return loss
def binary_cross_entropy_derivative(y_true, y_pred, epsilon=1e-12):
    y_pred=np.clip(y_pred, epsilon, y_pred-epsilon) 
    grad=-(y_true/y_pred)+(1-y_true)/(1-y_pred)
    grad/=y_true.shape[0]
    return grad
def dropout(x, dropout_rate=0.001):
    mask=(np.random.rand(*x.shape)>dropout_rate)/(1-dropout_rate)
    return x*mask, mask
def image_to_matrix(image_path, size=(cmp_size,cmp_size)):
    pic=Image.open(image_path).convert('L')
    pic=pic.resize(size)
    matrix=np.array(pic)
    return matrix.flatten()

pistol=[image_to_matrix(r"D:\Pictures\machine_learning\guns_data\pistol_"+str(i)+".png") for i in range(bs)]
rifle=[image_to_matrix(r"D:\Pictures\machine_learning\guns_data\rifle_"+str(i)+".png") for i in range(bs)]
shotgun=[image_to_matrix(r"D:\Pictures\machine_learning\guns_data\shotgun_"+str(i)+".png") for i in range(bs)]
bow=[image_to_matrix(r"D:\Pictures\machine_learning\guns_data\bow_"+str(i)+".png") for i in range(bs)]
empty=[image_to_matrix(r"D:\Pictures\machine_learning\guns_data\empty_"+str(i)+".png") for i in range(bs)]
x=np.vstack(pistol+rifle+shotgun+bow) #(bs, cmp_size**2)
y0=np.array([[pistol_index]]*bs+[[no_id]]*bs+[[no_id]]*bs+[[no_id]]*bs) #labels for each gun type, dims (bs, 4)
y1=np.array([[no_id]]*bs+[[rifle_index]]*bs+[[no_id]]*bs+[[no_id]]*bs)
y2=np.array([[no_id]]*bs+[[no_id]]*bs+[[shotgun_index]]*bs+[[no_id]]*bs)
y3=np.array([[no_id]]*bs+[[no_id]]*bs+[[no_id]]*bs+[[bow_index]]*bs)
w00=np.random.randn(cmp_size**2, hid_nrn)/cmp_size**2 #naming structure is w[layer][gun index]
w01=np.random.randn(cmp_size**2, hid_nrn)/cmp_size**2
w02=np.random.randn(cmp_size**2, hid_nrn)/cmp_size**2
w03=np.random.randn(cmp_size**2, hid_nrn)/cmp_size**2
w10=np.random.randn(hid_nrn,hid_nrn)/hid_nrn
w11=np.random.randn(hid_nrn,hid_nrn)/hid_nrn
w12=np.random.randn(hid_nrn,hid_nrn)/hid_nrn
w13=np.random.randn(hid_nrn,hid_nrn)/hid_nrn
w20=np.random.randn(hid_nrn,1)/hid_nrn
w21=np.random.randn(hid_nrn,1)/hid_nrn
w22=np.random.randn(hid_nrn,1)/hid_nrn
w23=np.random.randn(hid_nrn,1)
b00=np.zeros((1,hid_nrn)) #naming structure is b[layer][gun index]
b01=np.zeros((1,hid_nrn))
b02=np.zeros((1,hid_nrn))
b03=np.zeros((1,hid_nrn))
b10=np.zeros((1,hid_nrn))
b11=np.zeros((1,hid_nrn))
b12=np.zeros((1,hid_nrn))
b13=np.zeros((1,hid_nrn))
b20=np.zeros((1,1))
b21=np.zeros((1,1))
b22=np.zeros((1,1))
b23=np.zeros((1,1))

for epoch in range(epochs):
    #FORWARD PASS
    #layer z0
    z00=np.clip(x@w00+b00, clip_low,clip_high)
    z01=np.clip(x@w01+b01, clip_low,clip_high)
    z02=np.clip(x@w02+b02, clip_low,clip_high)
    z03=np.clip(x@w03+b03, clip_low,clip_high)
    #activation a0
    a00=leaky_relu(z00)
    a01=leaky_relu(z01)
    a02=leaky_relu(z02)
    a03=leaky_relu(z03)
    #layer z1
    z10=np.clip(a00@w10+b10, clip_low,clip_high)
    z11=np.clip(a01@w11+b11, clip_low,clip_high)
    z12=np.clip(a02@w12+b12, clip_low,clip_high)
    z13=np.clip(a03@w13+b13, clip_low,clip_high)
    #activation a1
    a10=leaky_relu(z10)
    a11=leaky_relu(z11)
    a12=leaky_relu(z12)
    a13=leaky_relu(z13)
    #layer z2
    z20=np.clip(a10@w20+b20, clip_low,clip_high)
    z21=np.clip(a11@w21+b21, clip_low,clip_high)
    z22=np.clip(a12@w22+b22, clip_low,clip_high)
    z23=np.clip(a13@w23+b23, clip_low,clip_high)
    #activation a2
    a20=leaky_relu(z20)
    a21=leaky_relu(z21)
    a22=leaky_relu(z22)
    a23=leaky_relu(z23)
    #BACKPROPAGATION
    #loss for each type of gun
    l0=np.mean(np.square(a20-y0))
    l1=np.mean(np.square(a21-y1))
    l2=np.mean(np.square(a22-y2))
    l3=np.mean(np.square(a23-y3))
    #layer a2 gradient for each gun type, dl/da2[gun index]
    a20grad=2*(a20-y0)
    a21grad=2*(a21-y1)
    a22grad=2*(a22-y2)
    a23grad=2*(a23-y3)
    #layer z2 gradient
    z20grad=a20grad*leaky_relu_derivative(z20)
    z21grad=a21grad*leaky_relu_derivative(z21)
    z22grad=a22grad*leaky_relu_derivative(z22)
    z23grad=a23grad*leaky_relu_derivative(z23)
    #layer z1 gradient
    z10grad=z20grad@w20.T*leaky_relu_derivative(z10)
    z11grad=z21grad@w21.T*leaky_relu_derivative(z11)
    z12grad=z22grad@w22.T*leaky_relu_derivative(z12)
    z13grad=z23grad@w23.T*leaky_relu_derivative(z13)
    #layer z0 gradient
    z00grad=z10grad@w10.T*leaky_relu_derivative(z00)
    z01grad=z10grad@w11.T*leaky_relu_derivative(z01)
    z02grad=z10grad@w12.T*leaky_relu_derivative(z02)
    z03grad=z10grad@w13.T*leaky_relu_derivative(z03)
    #layer z2 weights gradients
    w20grad=a10.T@z20grad
    w21grad=a11.T@z21grad
    w22grad=a12.T@z22grad
    w23grad=a13.T@z23grad
    #layer z1 weights gradients
    w10grad=a00.T@z10grad
    w11grad=a01.T@z11grad
    w12grad=a02.T@z12grad
    w13grad=a03.T@z13grad
    #layer z0 weights gradients
    w00grad=x.T@z00grad
    w01grad=x.T@z01grad
    w02grad=x.T@z02grad
    w03grad=x.T@z03grad
    #layer z2 bias gradients
    b20grad=np.sum(z20grad, axis=0, keepdims=True)
    b21grad=np.sum(z21grad, axis=0, keepdims=True)
    b22grad=np.sum(z22grad, axis=0, keepdims=True)
    b23grad=np.sum(z23grad, axis=0, keepdims=True)
    #layer z1 bias gradients
    b10grad=np.sum(z10grad, axis=0, keepdims=True)
    b11grad=np.sum(z11grad, axis=0, keepdims=True)
    b12grad=np.sum(z12grad, axis=0, keepdims=True)
    b13grad=np.sum(z13grad, axis=0, keepdims=True)
    #layer z0 bias gradients
    b00grad=np.sum(z00grad, axis=0, keepdims=True)
    b01grad=np.sum(z01grad, axis=0, keepdims=True)
    b02grad=np.sum(z02grad, axis=0, keepdims=True)
    b03grad=np.sum(z03grad, axis=0, keepdims=True)
    #layer z2 weights updates
    w20-=w20grad*lr
    w21-=w21grad*lr
    w22-=w22grad*lr
    w23-=w23grad*lr
    #layer z1 weights updates
    w10-=w10grad*lr
    w11-=w11grad*lr
    w12-=w12grad*lr
    w13-=w13grad*lr
    #layer z0 weights updates
    w00-=w00grad*lr
    w01-=w01grad*lr
    w02-=w02grad*lr
    w03-=w03grad*lr
    #layer z2 bias updates
    b20-=b20grad*lr
    b21-=b21grad*lr
    b22-=b22grad*lr
    b23-=b23grad*lr
    #layer z1 bias updates
    b10-=b10grad*lr
    b11-=b11grad*lr
    b12-=b12grad*lr
    b13-=b13grad*lr
    #layer z0 bias updates
    b00-=b00grad*lr
    b00-=b00grad*lr
    b00-=b00grad*lr
    b00-=b00grad*lr
    #printing of the loss during learning
    if l0<loss_threshold and l1<loss_threshold and l2<loss_threshold and l3<loss_threshold:
        print (f"epoch {epoch}, loss0 {l0:.4f}, loss1 {l1:.4f}, loss2 {l2:.4f}, loss3 {l3:.4f}\nBREAK")
        break
    if epoch%100==0 or epoch==epochs-1: print (f"epoch {epoch}, loss0 {l0:.4f}, loss1 {l1:.4f}, loss2 {l2:.4f}, loss3 {l3:.4f}") 
    
#inference
t_pistol=[image_to_matrix(r"D:\Pictures\machine_learning\guns_data\pistol_test_"+str(i)+".png") for i in range(t_bs)]
t_rifle=[image_to_matrix(r"D:\Pictures\machine_learning\guns_data\rifle_test_"+str(i)+".png") for i in range(t_bs)]
t_shotgun=[image_to_matrix(r"D:\Pictures\machine_learning\guns_data\shotgun_test_"+str(i)+".png") for i in range(t_bs)]  
t_bow=[image_to_matrix(r"D:\Pictures\machine_learning\guns_data\bow_test_"+str(i)+".png") for i in range(t_bs)]
t_empty=[image_to_matrix(r"D:\Pictures\machine_learning\guns_data\empty_test_"+str(i)+".png") for i in range(t_bs)]
t_x=np.vstack(t_pistol+t_rifle+t_shotgun+t_bow)

t_z00=t_x@w00+b00
t_z01=t_x@w01+b01
t_z02=t_x@w02+b02
t_z03=t_x@w03+b03
#activation a0
t_a00=leaky_relu(t_z00)
t_a01=leaky_relu(t_z01)
t_a02=leaky_relu(t_z02)
t_a03=leaky_relu(t_z03)
#layer z1
t_z10=t_a00@w10+b10
t_z11=t_a01@w11+b11
t_z12=t_a02@w12+b12
t_z13=t_a03@w13+b13
#activation a1
t_a10=leaky_relu(t_z10)
t_a11=leaky_relu(t_z11)
t_a12=leaky_relu(t_z12)
t_a13=leaky_relu(t_z13)
#layer z2
t_z20=t_a10@w20+b20
t_z21=t_a11@w21+b21
t_z22=t_a12@w22+b22
t_z23=t_a13@w23+b23
#activation a2
t_a20=leaky_relu(t_z20)
t_a21=leaky_relu(t_z21)
t_a22=leaky_relu(t_z22)
t_a23=leaky_relu(t_z23)

print (f"pistols output:\n {t_a20}")
print (f"rifle output:\n {t_a21}")
print (f"shotgun output:\n {t_a22}")
print (f"bow output:\n {t_a23}")
