# cifar10-multi-cnn-classifier
Multi-CNN image classification project using AlexNet, LeNet-5, VGG-16, and ResNet trained on CIFAR-10 with an interactive Streamlit web app.


🧠 CIFAR-10 Multi-Model Image Classification

A Deep Learning project that trains and compares multiple CNN architectures on the CIFAR-10 dataset and deploys them using a Streamlit web application.

📌 Models Implemented

This project implements and compares the following CNN architectures:

AlexNet

LeNet-5

VGG-16

ResNet

All models are trained on the CIFAR-10 dataset and saved for deployment.

📊 Dataset Used

CIFAR-10 Dataset

60,000 color images

Image size: 32x32x3

10 classes:

Airplane

Automobile

Bird

Cat

Deer

Dog

Frog

Horse

Ship

Truck

#### Dataset is loaded directly from TensorFlow:
```bash
tf.keras.datasets.cifar10.load_data()
```
#### 🏗 Project Structure
```bash
cnn_alexnet_lenet_vgg_resnet_model/
│
├── alexnet.py          # AlexNet architecture
├── leenet.py           # LeNet-5 architecture
├── vgg.py              # VGG-16 architecture
├── resnet.py           # ResNet architecture
├── train.py            # Training & evaluation script
├── app.py              # Streamlit web app
├── requirements.txt    # Required libraries
├── .gitignore
└── save_models/        # Saved trained models
```
## ⚙️ Training Models
##### Run the training script:
```bash
python train.py
```
This will:
- Train all 4 models
- Save best models in save_models/
- Evaluate performance on test dataset

##### Models are saved using:
```bash
ModelCheckpoint(save_best_only=True)
```
## 🌐 Running the Web Application
##### After training models, start the Streamlit app:
```bash
streamlit run app.py
```

Features of Web App

Select any trained model

Upload an image (jpg/png)

Automatic preprocessing

Top prediction with confidence

Top-3 predictions

Probability bar chart

🧠 Technologies Used

TensorFlow / Keras

Streamlit

NumPy

Pandas

Matplotlib

Pillow
