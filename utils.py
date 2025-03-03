import langchain
import cv2
import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import base64
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
import matplotlib.pyplot as plt
import numpy as np
import sys
from PIL import Image
import json

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

def save_output(final_result_dict):
    with open('final_output.json', 'w') as fp:
        json.dump(final_result_dict, fp,indent=4)
    print('Output Saved')

def display_results(results_dict):
    for k,v in results_dict.items():
        im = Image.open(f'images/{k}')
        plt.imshow(im)
        plt.show()
        print(v)

def gen_image_encoding(image_file_name):
    image_path = f'images/{image_file_name}'
    if filesize_gt_20mb(image_path=image_path):
        base64_image = encode_image(image_path=compress_image_opencv(input_path=image_path,quality=85))
    else:
        base64_image = encode_image(image_path=image_path)
    return base64_image

def generate_sytem_prompt(sys_prompt_file):
    with open(sys_prompt_file,'r') as prompt_file:
        base_system_prompt = prompt_file.read()
        output_schema_prompt = """\nYou should return the output wrapped in JSON tags using the schema given below
                            {image file name (provided with the image input): 
                            your generated caption as per system instructions }"""
        system_prompt = base_system_prompt + output_schema_prompt
    print(f"System Prompt is: \n {base_system_prompt} \n")
    return SystemMessage(content=system_prompt)

def generate_user_prompt(image_list):
    user_prompt_list = []
    for i in image_list:
        base64_image = gen_image_encoding(image_file_name=i)
        user_prompt_list.extend([{
            "type":"text",
            "text":f"Image file name:{i}"
        },
        {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        }])
    return HumanMessage(content=user_prompt_list)

def generate_final_prompt(system_prompt_file,image_list):
    system_message = generate_sytem_prompt(sys_prompt_file=system_prompt_file)
    user_message = generate_user_prompt(image_list=image_list)
    print(f'Num of Images: {len(image_list)}')
    return (system_message + user_message)

def genrate_caption(llm,system_prompt_file,image_list):
    prompt = generate_final_prompt(system_prompt_file=system_prompt_file,image_list=image_list)
    parser = JsonOutputParser()
    chain = prompt | llm | parser
    res_dict = chain.invoke({})
    display_results(res_dict)
    return res_dict
    
def regenrate_caption(chain,system_prompt,image_file_name,initial_output):
    image_path = f'images/{image_file_name}'
    if filesize_gt_20mb(image_path=image_path):
        base64_image = encode_image(image_path=compress_image_opencv(input_path=image_path,quality=85))
        return(chain.invoke({'system_prompt':system_prompt,'image_file_name':image_file_name,'initial_output':initial_output,'base64_image':base64_image}))
    else:
        base64_image = encode_image(image_path=image_path)
        return(chain.invoke({'system_prompt':system_prompt,'image_file_name':image_file_name,'initial_output':initial_output,'base64_image':base64_image}))