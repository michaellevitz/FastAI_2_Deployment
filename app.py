from fastai.vision.all import *
import gradio as gr
def is_cat(x): return x[0].isupper()

learn = load_learner('model.pkl')

categories = ('Dog', 'Cat')

def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))

demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),  # Image uploader
    #outputs=gr.Textbox(label="Classification Result"),  # Textbox for the output
    outputs=gr.Label(label="Classification Result"),  # Bar chart and label for the output
    examples=['dog.jpg', 'cat.jpg', 'dunno.jpg']  # Example images
)
demo.launch(inline=False,share=True)