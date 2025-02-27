import langchain
import cv2
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import base64
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
import matplotlib.pyplot as plt
import numpy as np
import sys
from PIL import Image

def filesize_gt_20mb(image_path):
    filesize = os.stat(image_path).st_size/1024**2
    if filesize >= 20: 
        return True
    
def compress_image_opencv(input_path, quality=85):
    output_path = input_path.replace('.jpg','_resized.jpg')
    if filesize_gt_20mb(input_path):
        img = cv2.imread(input_path)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        cv2.imwrite(output_path, img, encode_param)
        print(f'{input_path} resized')
        return output_path
    
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def get_file_names(image_files_dir):
    return [file for file in os.listdir('images') if (file.endswith('.jpg') or file.endswith('.JPG')) and not file.endswith('_resized.jpg')]

def genrate_caption(chain,system_prompt,image_file_name):
    image_path = f'images/{image_file_name}'
    if filesize_gt_20mb(image_path=image_path):
        base64_image = encode_image(image_path=compress_image_opencv(input_path=image_path,quality=85))
        response = chain.invoke({'system_prompt':system_prompt,'base64_image':base64_image})
        return {image_file_name:response}
    else:
        base64_image = encode_image(image_path=image_path)
        response = chain.invoke({'system_prompt':system_prompt,'base64_image':base64_image})
        return {image_file_name:response}
    
def display_results(final_results_dict):
    for k,v in final_results_dict.items():
        im = Image.open(f'images/{k}')
        plt.imshow(im)
        plt.show()
        print(v)

def regenrate_caption(chain,system_prompt,image_file_name,initial_output):
    image_path = f'images/{image_file_name}'
    if filesize_gt_20mb(image_path=image_path):
        base64_image = encode_image(image_path=compress_image_opencv(input_path=image_path,quality=85))
        response = chain.invoke({'system_prompt':system_prompt,'initial_output':initial_output,'base64_image':base64_image})
        return {image_file_name:response}
    else:
        base64_image = encode_image(image_path=image_path)
        response = chain.invoke({'system_prompt':system_prompt,'initial_output':initial_output,'base64_image':base64_image})
        return {image_file_name:response}