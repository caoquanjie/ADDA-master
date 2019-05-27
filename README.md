# ADDA
# adversarial domain discrimative adaptation

a tensorflow version implement of ADDA

## requirements
python 3.6
tensorflow 1.4.0
numpy
scipy

## SVHN to MNIST
## encoder network achitecture
lenet-5:
input (32,32,3)
conv1 filter (5,5,20)  output (28,28,20)
maxpool (14,14,20)
conv2 filter (5,5,50)  output (10,10,50)
maxpool (5,5,50)
flat1  (1250,120)
flat2 (120,84)
classifier (84,10)

more details can be seen in adda.py if you want to design another CNN network

## usage
this repositority only implement SVHN to MNIST, you can change another dataset such as USPS to MNIST or MNIST to USPS
if you are interested.

python main.py

step1: training the source network
step2: training the target and discriminator network.
step3: test target dataset.

target accuracy is 63%(only source) and 77%(after adda).