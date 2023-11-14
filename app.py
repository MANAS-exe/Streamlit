import streamlit as st
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from skorch import NeuralNetClassifier
from sklearn.metrics import accuracy_score
import torch
from torch import nn
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from PIL import Image
from streamlit_drawable_canvas import st_canvas
#import time
st.markdown(
    """
    <style>
        .header-text {
            font-family: 'Arial', sans-serif;
            font-size: 24px;
            font-weight: bold;
        }
        .title-text {
            font-family: 'Times New Roman', serif;
            font-size: 20px;
            font-weight: bold;
        }
        .normal-text {
            font-family: 'Courier New', monospace;
            font-size: 16px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

#st.header("IMAGE CLASSIFICATION - MNIST DATASET")
#st.latex(r''' e^{i\pi} + 1 = 0 ''')
st.markdown('<p class="header-text">IMAGE CLASSIFICATION - MNIST DATASET</p>', unsafe_allow_html=True)


def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test that is a number from 1 to 9')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)

def plot_example(X, y):
    """Plot the first 100 images in a 10x10 grid."""
    fig = plt.figure(figsize=(15, 15))  # Set figure size to be larger (you can adjust as needed)

    for i in range(10):  # For 10 rows
        for j in range(10):  # For 10 columns
            index = i * 10 + j
            ax = fig.add_subplot(10, 10, index + 1)
            #plt.subplot(10, 10, index + 1)  # 10 rows, 10 columns, current index
            plt.imshow(X[index].reshape(28, 28))  # Display the image
            ax.set_xticks([])  # Remove x-ticks
            ax.set_yticks([])  # Remove y-ticks
            ax.set_title(y[index], fontsize=8)   # Display the label as title with reduced font size

    fig.subplots_adjust(wspace=0.5, hspace=0.5)  # Adjust spacing (you can modify as needed)
    plt.tight_layout()  # Adjust the spacing between plots for better visualization
     # Display the entire grid
    return fig
def home_page():
    st.title("Home Page")
    st.write("This is an app based on the classification of digits based on MNIST data.")
    st.write('The MNIST dataset is a collection of 28 x 28 pixel grayscale images of handwritten digits (0 through 9). It is a widely used dataset in machine learning and computer vision for training and testing various image processing systems. The name "MNIST" stands for Modified National Institute of Standards and Technology.')
    image = Image.open('mnist.webp')
    st.image(image , caption='This is the dataset which you will train ')

    st.header('Now what is Machine Learning ?')
    st.write('Machine learning is a subset of artificial intelligence (AI) that focuses on the development of algorithms and models that enable computer systems to learn from data and improve their performance on a specific task without being explicitly programmed. In other words, machine learning involves the development of systems that can automatically learn and make predictions or decisions based on data.')

    image2=Image.open('public_images_1605842918803-AI+vs+ML+vs+DL.png')
    st.image(image2,caption = 'DL is what we will focus more')
    # Add content specific to the home page
    st.header("MACHINE LEARNING OR DEEP LEARNING PIPELINE:")
    image3=Image.open('manual-pipeline.png')
    st.image(image3,caption = "This is a general pipeline")
    st.image("The-Pattern-Recognition-Pipeline-From-the-original-image-to-the-classified-result.png",caption = "Image classification pipeline")
    st.text('but we will not focus this much in depth at the level of class 5-10')

    st.header("But what is a MODEL ?")
    st.write("A machine learning (ML) model is a mathematical representation or a computational system that learns patterns from data and makes predictions or decisions without being explicitly programmed for the task. In other words, a machine learning model is trained on historical data to recognize patterns and relationships, and it can then use that knowledge to make predictions or decisions on new, unseen data.")

    st.write("Our model typically consists of several key components designed to effectively process and classify images. Here's a theoretical overview of a simple CNN architecture for MNIST classification:")
    image4 = Image.open('cnn.webp')
    st.image(image4)

    st.write('Now we will proceed towards our ML model :muscle: 	:female-technologist:')
    st.text('SO SIT BACK,RELAX AND ENJOY IT !	 ')
    st.write('	:tropical_drink: :tropical_drink:')

    st.text('Various parameters with which you are going to play with has been explained in detail in the "Definitions" page.')
    st.text("SO KINDLY HAVE A LOOK ON THAT WHILE VARYING IT IN THE TEST SECTION. ")
    st.write("All the parameters have been set to default value for better training but you can change them and see the effects (that's how you will learn MACHINE LEARNING :wink::wink:) ")
def train():
    test_size_key = "test_size_slider"
    random_state_key = "random_state_slider"
    epoch_key = "epoch_slider"
    lr_key = "lr_slider"
    dropout_key = "dropout_slider"

    st.title("TRAINING THE DATA")

    st.markdown("<a id='test_size_section'></a>", unsafe_allow_html=True)

        
    test_size = st.slider(' ENTER TEST SIZE :', min_value=0.00, max_value=1.00, value=0.25, step=0.01, format='%0.2f',key=test_size_key)
    random_state = st.number_input("ENTER RANDOM STATE", min_value=1, step=1, value=42,key = random_state_key)

    mnist = fetch_openml('mnist_784', as_frame=False, cache=False)
    X = mnist.data.astype('float32')
    y = mnist.target.astype('int64')
    X = X/255.0
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    assert(X_train.shape[0] + X_test.shape[0] == mnist.data.shape[0])
    st.write('Training Samples -->',  X_train.shape[0])
    st.write('Testing Samples -->', X_test.shape[0])
    #st.write('y_train =', y_train.shape)

    if st.button('Show example images'):
        fig = plot_example(X_train, y_train)
    
    # Display the figure in Streamlit
        st.pyplot(fig)
    
    mnist_dim = X.shape[1]
    hidden_dim = int(mnist_dim/8)
    output_dim = len(np.unique(mnist.target))

    mnist_dim, hidden_dim, output_dim


    torch.manual_seed(0)
    epoch = st.number_input("ENTER EPOCH", min_value=1, step=1, value=20,key = epoch_key)
    lr = st.slider("ENTER Learning Rate", min_value=0.0, max_value=1.0, value=0.1, step=0.1, format='%0.1f',key=lr_key)
    
    dropout = st.slider("ENTER DROPOUT",min_value=0.0,max_value=1.0,value=0.2,step=0.1,format='%0.2f',key=dropout_key)

    accuracy_scores = []

    
    class ClassifierModule(nn.Module):
        def __init__(
                    self,
                    input_dim=mnist_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    dropout=dropout,
            ):
                super(ClassifierModule, self).__init__()
                self.dropout = nn.Dropout(dropout)

                self.hidden = nn.Linear(input_dim, hidden_dim)
                self.output = nn.Linear(hidden_dim, output_dim)

        def forward(self, X, **kwargs):
            X = F.relu(self.hidden(X))
            X = self.dropout(X)
            X = F.softmax(self.output(X), dim=-1)
            return X
        
    #accuracy_placeholder = st.empty()

# X_accumulated = np.empty_like(X_train[:0])
# y_accumulated = np.empty_like(y_train[:0])
    
    
    

    net = NeuralNetClassifier(
    ClassifierModule,
    max_epochs=epoch,
    lr=lr,
    device=device,
    )
    
    net.fit(X_train, y_train);

    y_pred = net.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
        ##accuracy_scores.append(acc)

        #accuracy_placeholder.text(f"Epoch {epoch + 1} - Accuracy: {acc:.3f}")
    col1 , col2=st.columns(([3,1]))
    col1.text("ACCURACY SCORE - " + str(accuracy_score(y_test, y_pred)))
    pr = col2.progress(0)

    for i in range(int(accuracy_score(y_test, y_pred))):
        time.sleep(0.1)
        pr.progress(i+1)

   

    
        # Add content specific to the about page
    return net


def test(net):
    st.title("TESTING THE DATA")

    if trained_model is None:
        st.warning("Please train the model before making predictions.")
    else:

        uploaded_file = st.file_uploader(label='Pick an image depicting a single digit from 0 to 9')
        if uploaded_file is not None:
            image_data = uploaded_file.getvalue()
            image = Image.open(uploaded_file).convert("L").resize((28, 28))
            image = np.array(image) / 255.0

            st.image(image, caption='Processed Image', use_column_width=True)
            
            image = image.flatten().reshape(1, -1)
            image_tensor = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0)

            # Reshape the tensor to remove unnecessary dimensions
            image_tensor = image_tensor.view(1, -1)

            # Make predictions using the provided model (net)
            prediction = net.predict(image_tensor)

            # Ensure prediction is a 1D array before converting to int
            predicted_digit = int(prediction[0] if prediction.ndim > 1 else prediction)
            st.write("Predicted Digit:   ", predicted_digit)

            ANS = st.radio('Is the Prediction Correct ?',options=['True','False'],index=None)
            if ANS=='True':
                if st.button("Want to celebrate :partying_face: :partying_face:"):
                    st.balloons()
            
                
                
                
        if st.button('Use canvas to predict') :
            drawing_mode = st.sidebar.selectbox(
            "Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
            )

            stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
            if drawing_mode == 'point':
                point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
            stroke_color = st.sidebar.color_picker("Stroke color hex: ")
            bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
            bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])

            #realtime_update = st.sidebar.checkbox("Update in realtime", True)



            # Create a canvas component
            canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image=Image.open(bg_image) if bg_image else None,
            #update_streamlit=realtime_update,
            height=150,
            drawing_mode=drawing_mode,
            point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
            key="canvas",
            )


    # Do something interesting with the image data and paths
            if canvas_result.image_data is not None:
                st.image(canvas_result.image_data)
        

trained_model = None
def main():
    global trained_model

    st.sidebar.title("MAIN MENU")
    page_options = ["HOME", "YOUR MODEL!", "DEFINITIONS"]
    selected_page = st.sidebar.radio("Select a page", page_options)

    if selected_page == "HOME":
        home_page()
    elif selected_page == "YOUR MODEL!":
        st.empty()
        trained_model = train()
        test(trained_model)
    #elif selected_page == "Test":
    #    test()
    
        #load_image()
               
        
        #st.sidebar.title("MAIN MENU")
    elif selected_page == "DEFINITIONS":
        st.empty()
        st.header(":rainbow[1.TEST SIZE :] ")
        st.write("In the context of data analysis, machine learning, or statistical modeling, It typically refers to the proportion of the dataset that is used for testing the performance of a model. When you split a dataset into training and testing sets, the test size parameter determines the fraction of the data that is reserved for testing the model's performance.")
        st.write("It ranges between 0.0 and 1.0")

        st.header(":rainbow[2.RANDOM STATE :]")
        st.write("In machinelearning, the random_state parameter is used to seed the random number generator. It's particularly relevant when there is a need for reproducibility. By setting the random_state to a specific value, you ensure that the random numbers generated during the training process are the same every time you run the code. This is crucial when you want to compare different models or rerun experiments to validate results")
        st.write("Imagine you have a deck of cards, and you want to shuffle it. In machine learning terms, this deck represents your dataset. The order of the cards in the deck is analogous to the order of your data points")
        st.write("If you shuffle the deck without specifying a random_state, it's like shuffling the deck without controlling the random process. Each time you shuffle, you get a different order, just as if you run your machine learning code without specifying random_state")   
        st.write("If you set a specific random_state, it's like using a specific seed for the random number generator. It ensures that every time you shuffle with that seed, you get the same order. This is analogous to setting the random_state in machine learning to get reproducible results")

        st.header(":rainbow[3.EPOCH :]")
        st.write("In the context of machine learning, an epoch refers to one complete cycle through the entire training dataset. During an epoch, the learning algorithm processes the entire dataset, calculates the error, and updates the model's weights. The goal of training a machine learning model is to reduce the error, and this process is typically repeated over multiple epochs.")
        c1,c2 = st.columns([2,2])
        c1.subheader("Components of epoch")
        c1.write("1.Forward Pass")
        c1.write("2.Backward Pass")
        c1.write("3.Update Parameters")
        
        c2.image(Image.open('epoch.webp'))

        c3,c4=st.columns([2,2])
        st.header(":rainbow[4.LEARNING RATE :]")
        st.write('The learning rate is a critical hyperparameter in machine learning models')
        st.subheader("HYPERPARAMETER ?")
        st.write("In machine learning, hyperparameters are external configuration settings that are not learned from the data but are set prior to the training process. These parameters are essential for controlling the learning process and the overall behavior of a machine learning model. ")

       # c1.write("A higher learning rate generally leads to faster convergence during training. The model may reach a solution more quickly, especially in the early epochs.")
        st.image(Image.open('alpha.png'))

        st.header(":rainbow[5.DROPOUT :]")

        st.write('Dropout is a regularization technique commonly used in neural networks during training to prevent overfitting. Overfitting occurs when a model learns not only the underlying patterns in the training data but also noise and details that are specific to that data, leading to poor performance on new, unseen data. Dropout is a simple yet effective method to improve the generalization ability of neural networks.')
        st.image('dropout.png')
if __name__ == '__main__':
        main()
