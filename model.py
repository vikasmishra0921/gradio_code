import gradio as gr
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


#Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data,columns=iris.feature_names)
y = iris.target
model = RandomForestClassifier()
model.fit(X,y)

#Define prediction function

def predict(sepal_length,sepal_width,petal_length,petal_width):
    input_data = [[sepal_length,sepal_width,petal_length,petal_width]]
    prediction = model.predict(input_data)[0]
    return iris.target_names[prediction]

#create Gradio interface

interface = gr.Interface(
        fn=predict,
        inputs=[
            gr.slider(4.0,8.0,label="sepal_length (cm)"),
            gr.Slider(2.0,4.5,label = "sepal Width (cm)"),
            gr.Textbox(1.0,7.0, label ="Petal Length (cm)"),
            gr.Textbox(0.1,2.5, label = "Petal Width (cm)")
        ],
        outputs = "text",
        title = "Iris Flower classifier",
        description = "Enter sepal and petal dimensions to predict the species."

)

#Launch App
interface.launch()