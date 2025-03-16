# 1
import gradio as gr
import os
import torch
from model import create_effnetb2
from timeit import default_timer as timer
from typing import Tuple, Dict
from PIL import Image
class_names=['pizza', 'steak', 'sushi']
# 2
effnetb2,effnetb2_transforms=create_effnetb2(num_classes=3)
# load save weights
effnetb2.load_state_dict(
    torch.load(
        f="effnetb2 model.pth",
        map_location=torch.device("cpu") #load model to cpu

    )
)
# 3
def predict(img):
  start_time=timer()
  img=effnetb2_transforms(img).unsqueeze(0) #add batch dim
  effnetb2.eval()
  with torch.inference_mode():
    pred_logit=effnetb2(img)
    pred_probs=torch.softmax(pred_logit,dim=1)
    pred_labels_and_probs={class_names[i]:float(pred_probs[0][i])for i in range(len(class_names)) }
    end_time=timer()
    pred_time=round(end_time-start_time,4)
    return pred_labels_and_probs,pred_time

# 4

import gradio as gr
title="FoodVision Mini 🍕🥩🍣"
description="Classify images of food into pizza, steak, or sushi using an EfficientNet-B2 feature extractor. Quick, accurate, and perfect for showcasing computer vision in action!  "
# example list
# getting list of list

example_list=[["examples/"+example]for example in os.listdir("examples")]
# craete the gradient demo
demo=gr.Interface(fn=predict, #maps input to output
                  inputs=gr.Image(type="pil"),
                  outputs=[gr.Label(num_top_classes=3,label="Predictions"),
                           gr.Number(label="Prediction time(s)")],
                           examples=example_list,
                           title=title,
                           description=description

                  )
# launch it
demo.launch(
    debug=False,
    share=True
    # preints error locally? like in googlr collab
    # generate link publicaly like share with public
)


