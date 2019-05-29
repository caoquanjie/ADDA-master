# ADDA
# adversarial domain discrimative adaptation

a tensorflow version implement of ADDA

## SVHN to MNIST
### requirements
python 3.6</br>
tensorflow 1.4.0</br>
numpy</br>
scipy</br>

### encoder network achitecture
**lenet-5**:</br>
input (32,32,3)</br>
conv1 filter (5,5,20)  output: (28,28,20)</br>
maxpool output:(14,14,20)</br>
conv2 filter (5,5,50)  output: (10,10,50)</br>
maxpool output:(5,5,50)</br>
flat1  output: (1250,120)</br>
flat2 output: (120,84)</br>
classifier output: (84,10)</br>

**More details can be seen in adda.py if you want to design another CNN network</br>**

### usage
**this repositority only implement SVHN to MNIST, you can change another dataset such as USPS to MNIST or MNIST to USPS
if you are interested.</br>**

python main.py</br>

step1: training the source network</br>
step2: training the target and discriminator network.</br>
step3: test target dataset.</br>

target accuracy is **63%** (only source) and **77%** (after adda).