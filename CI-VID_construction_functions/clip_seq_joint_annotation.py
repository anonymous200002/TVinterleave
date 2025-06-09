from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import re
import torch
import os
import logging
import random
from tqdm import tqdm
import json
from PIL import Image
import math, argparse
import io
import tempfile
import shutil
import cv2


def add_img(imgs, content, text):
    for img in imgs[:-1]:
        content.append({
                        "type": "image",
                        "image": img,
                    })
        content.append({
                        "type": "text",
                        "text": text})
    content.append({
                        "type": "image",
                        "image": imgs[-1],
                    })
    return content

def save_image_temp(img: Image.Image, temp_path: str):
    path_dir = os.path.dirname(temp_path)
    os.makedirs(path_dir, exist_ok=True)
    img.save(temp_path)
    # 确保写入完成
    with open(temp_path, "rb") as f:
        _ = f.read()  # 强制 flush 检查完整性

def cal_qwen(messages, model, processor):
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=512)

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]


def video2img_diff(clip_paths):
    all_frames = []
    for clip_path in clip_paths:
        cap = cv2.VideoCapture(clip_path)

        # 获取视频总帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 获取视频帧率
        fps = cap.get(cv2.CAP_PROP_FPS)
        # 计算视频时长（秒）
        video_length = total_frames / fps

        num_frames = 3

        interval = video_length / (num_frames + 1)

        start_time = interval
        end_time = video_length

        current_time = start_time

        frames = []
        while current_time < (end_time - interval/2):
            # 设置到当前时间点
            cap.set(cv2.CAP_PROP_POS_MSEC, current_time * 1000)
            ret, frame = cap.read()
            frames.append(frame)
            current_time += interval

        assert len(frames) == num_frames
        all_frames.extend(frames)
        
    merged_imgs = get_merged_img_diff(all_frames)
    return  merged_imgs



def get_merged_img_diff(frames):
    l = len(frames)

    # Convert frames to images
    images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]

    img_width, img_height = images[0].size
    batch_images = images

    img_grid_w = 3
    img_grid_h = math.ceil(len(batch_images) / img_grid_w)

    # Create a blank white row to separate the images
    blank_row_height = 40  # Adjust the height of the blank space
    blank_row = Image.new('RGB', (img_grid_w * img_width, blank_row_height), (255, 255, 255))

    # Create the grid image
    grid_image = Image.new('RGB', (img_grid_w * img_width, (img_grid_h * img_height) + (img_grid_h - 1) * blank_row_height))

    # Paste images and insert blank rows between them
    current_y = 0
    for row in range(img_grid_h):
        for col in range(img_grid_w):
            index = row * img_grid_w + col
            if index < len(batch_images):  # Ensure the index is within the number of images
                image = batch_images[index]
                grid_image.paste(image, (col * img_width, current_y))
        # After each row, add a blank row
        current_y += img_height + blank_row_height

    # Resize if necessary
    target_height = grid_image.height
    target_width = grid_image.width

    # Scale the image if the height exceeds a certain limit
    if target_height > 5000:
        scale_factor = 5000 / target_height
        target_width = int(target_width * scale_factor)
        target_height = int(target_height * scale_factor)

    grid_image = grid_image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    return grid_image

def cal_response_joint(joint_paths, model, processor):
    response_texts = []
    for joint_path in joint_paths:
        content = []
        prompt_all = {"type": "text",
                    "text": f"""
                            Here are two consecutive video clips. Each row corresponds to one clip. Please describe the changes in the second clip (bottom row) compared to the first clip (above row):
                            1) Describe the change in the camera angle (the perspective from which the camera captures the main object).
                            2) Describe the change in the camera movement.
                            3) Describe the change in the background, including environmental factors, lighting, space, etc.
                            4) Provide a detailed description of the first clip content, including actions, plot, characters, objects, colors, and any other relevant visual or thematic elements (300 words).
                            5) Provide a detailed description of the second clip content, including actions, plot, characters, objects, colors, and any other relevant visual or thematic elements (300 words).
                            6) Provide a detailed description of the change in video content.
                            Don't analyze, subjective interpretations, aesthetic rhetoric, etc., focus solely on objective descriptions.
                            Directly return in the json format like this:
                            {{"camera_angle": "...", "camera_movement": "...", "content": "...", "background": "..."}}. 
                            """
                }
        
        content.append(prompt_all)
        content = add_img([joint_path], content, '')
        messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]
        response_text = cal_qwen(messages, model, processor)
        response_texts.append(response_text)
    return response_texts


def generate_joint_clip_captions(video_clip_paths, temp_dir = '', model_path = "pretrained_models/Qwen-VL"):
    """
    输入 clip 路径列表，生成相邻 pair joint caption。
    返回值: List[caption string]
    """
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto", low_cpu_mem_usage=True).eval()

    min_pixels = 256*28*28
    max_pixels = 1280*28*28
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)

    joint_imgs = [video2img_diff(video_clip_paths[i:i+2]) for i in range(len(video_clip_paths)-1)]

    # 存储为临时路径
    temp_paths = []
    os.makedirs(temp_dir, exist_ok=True)
    for i, img in enumerate(joint_imgs):
        temp_path = f"{temp_dir}/joint_{i}.jpg"
        save_image_temp(img, temp_path)
        temp_paths.append(temp_path)

    captions = cal_response_joint(temp_paths, model, processor)
    return captions


video_clip_paths = [
    '',
    '',
    ''
    ]

temp_dir = ''
captions = generate_joint_clip_captions(video_clip_paths, temp_dir)
print(captions)
