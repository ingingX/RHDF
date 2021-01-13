# ---- Using packages
using Plots # Draw pictures
using Flux, Flux.Data.MNIST, Statistics # Flux Frame and MNIST DB
using Flux: onehotbatch, onecold, crossentropy, throttle # One-Hot Encoding and One-Cold Encoding function
using Base.Iterators: repeated # Repeat training process function


# ---- Prepare input data
# Load training data: total 60000 images with 28*28 grayscale
imgs = MNIST.images()

# Draw some pictures
imgplots = plot.(imgs[1:9])
plot(imgplots...)

# Reorder each image as 28*28 = 784 length column vector, and row the 60000 images.
imagestrip(image::Matrix{}) = Float32.(reshape(image, :))

# Generate the 784x60000 input matrix X to ANN model.
X = hcat(imagestrip.(imgs)...)

# Load labels
labels = MNIST.labels()

# Creat a batch of one-hot encoded labels and spesify what digit each image represents.
Y = onehotbatch(labels, 0:9)


# ---- Defining the neural network
# Use simple multi-layer perceptron
m = Chain( # link three layers
 Dense(28*28, 32, relu), # First layer: 28*28 = 784 input nodes; 32 output nodes; ReLU works as activation function
 Dense(32, 10), # Hidden-layer: 32 input nodes corresponding to 32 outputs from the First layer; 10 output nodes
 softmax) # Output Layer: normalize along each column using softmax function
 
# Define loss function using crossentropy function instead of MSE
loss(x, y) = crossentropy(m(x), y)

# Specify 200 Epochs
dataset = repeated((X, Y), 200)

# Define Optimizer
# ADAM(), instead of Descent(), is more recommended when dealing with a lot of data with a fair amount of noise
opt = ADAM()


# ---- Training
# Feedback error dropped upon each epoch
evalcb = () -> @show(loss(X, Y))

# Perform training on data
Flux.train!(loss, params(m), dataset, opt, cb = throttle(evalcb, 10))


# ---- Verify trained model
# Define accuracy funciton
accuracy(x, y) = mean(onecold((m(x))) .== onecold(y))

# Reorder test database 
tX = hcat(float.(reshape.(MNIST.images(:test), :))...)
tY = onehotbatch(MNIST.labels(:test), 0:9)

# Test our trained model and output the accuracy
@show(accuracy(tX, tY))