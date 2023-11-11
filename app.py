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
    # Add content specific to the home page

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
    st.write('X_train =',  X_train.shape)
    st.write('y_train =', y_train.shape)

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
        

   
   # X_accumulated = np.empty_like(X_train[:0])
   # y_accumulated = np.empty_like(y_train[:0])
    if st.button('SHOW ITERATIONS'):
        for i in range(epoch):

            net = NeuralNetClassifier(
            ClassifierModule,
            max_epochs=i,
            lr=lr,
            device=device,
            )
            
            # st.write(network)

            #network = net.fit(X_train, y_train);
            #network = net.fit(X_train, y_train)
            net.fit(X_train, y_train);

            y_pred = net.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.write(f"In the iteration {i+1} the accuracy is {acc : 0.3f} ")
        
    else :
        net = NeuralNetClassifier(
        ClassifierModule,
        max_epochs=epoch,
        lr=lr,
        device=device,
        )
        net.fit(X_train, y_train);

        y_pred = net.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
    
    st.write("ACCURACY SCORE - ", accuracy_score(y_test, y_pred))
   
    # Add content specific to the about page
    return net


def test(net):
    st.title("TESTING THE DATA")

    uploaded_file = st.file_uploader(label='Pick an image to test that is a number from 1 to 9')
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
        st.write("Predicted Digit:", predicted_digit)
    
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


def main():
   
    st.sidebar.title("MAIN MENU")
    page_options = ["Home", "Train", "Definitions"]
    selected_page = st.sidebar.radio("Select a page", page_options)

    if selected_page == "Home":
        home_page()
    elif selected_page == "Train":
        net=train()
        test(net)
    #elif selected_page == "Test":
    #    test()
    
        #load_image()
               
        
        #st.sidebar.title("MAIN MENU")
    elif selected_page == "Definitions":
        st.header("1.TEST SIZE :")
        st.text("In the context of data analysis, machine learning, or statistical modeling, It typically refers to the proportion of the dataset that is used for testing the performance of a model. When you split a dataset into training and testing sets, the test size parameter determines the fraction of the data that is reserved for testing the model's performance.")
        

            

if __name__ == '__main__':
        main()
