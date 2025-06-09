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

       
def get_merged_img(frames, all_flag=False):
    l = len(frames)

    images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in frames]

    img_width, img_height = images[0].size
    batch_images = images


    batch_images_l = len(batch_images)

    if all_flag:
        img_grid_w = 3
        img_grid_h = math.ceil(len(batch_images) / img_grid_w)
    else:
        if batch_images_l == 3:
            img_grid_w, img_grid_h = 3,1
        elif batch_images_l ==4:
            img_grid_w, img_grid_h = 2, 2
        elif batch_images_l ==6:
            img_grid_w, img_grid_h = 3, 2
        else:
            print(f'error!!!!!!!!!!!!!!!!!!!!!! batch_images_l: {batch_images_l}')
        
    # 创建一个空白画布
    grid_image = Image.new('RGB', (img_grid_w * img_width, img_grid_h * img_height))

    # 将图像粘贴到网格中
    for index, image in enumerate(batch_images):
        x = (index % img_grid_w) * img_width
        y = (index // img_grid_w) * img_height
        grid_image.paste(image, (x, y))

    # 计算目标尺寸
    target_height = img_grid_h * img_height
    target_width = img_grid_w * img_width

    # 判断是否需要缩放
    if target_height > 5000:
        scale_factor = 5000 / target_height
        target_width = int(target_width * scale_factor)
        target_height = int(target_height * scale_factor)

    grid_image = grid_image.resize((target_width, target_height), Image.Resampling.LANCZOS)

    return grid_image 


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


def video2img(clip_path):
    cap = cv2.VideoCapture(clip_path)

    # 获取视频总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 获取视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 计算视频时长（秒）
    video_length = total_frames / fps

    num_frames = max(int(video_length), 4)
    num_frames = min(num_frames, 6)

    if num_frames == 5:
        num_frames = 4
        
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

    merged_imgs = get_merged_img(frames)
    return  merged_imgs

def ordinal(i):
    if 10 <= i % 100 <= 20:  # 特殊处理 11-13
        suffix = 'th'
    else:
        suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(i % 10, 'th')
    return f"{i}{suffix}"


def cal_response_single(single_paths, response_all, model, processor):
    l = len(single_paths)
    single_response_text = []
    for i, path in enumerate(single_paths):
        content = []
        suffix = ordinal(i+1)
        
        prompt_single = {"type": "text",
                    "text": f"""
                            Here is a video containing {l} clips, and here are the videp title and description: {response_all}.
                            Here is the {suffix} clip of the video. Please describe the clip by:
                            1) the camera angle (the angle from which the camera shoots the main object, 30 words)
                            2) the camera movement (15 words)
                            3) the clip content in detail, including what is happening, the plot, characters' actions, environment, lighting, objects, colors, and other visual or thematic elements (300 words).
                            Note: Don't analyze, subjective interpretations, aesthetic rhetoric, etc., just objective statements.
                            Do not describe the frames individually but the whole video clip.
                            Directly return in the json format like this:
                            {{"camera_angle": "...", "camera_movement": "...", "content": "..."}}. 
                            """
                        }
    
        content.append(prompt_single)
        content = add_img([path], content, '')
        messages = [
                {
                    "role": "user",
                    "content": content
                }
            ]
        response_text = cal_qwen(messages, model, processor)
        single_response_text.append(response_text)
    return single_response_text


def generate_single_clip_captions(video_clip_paths, response_all = '', temp_dir = '', model_path = "Qwen-VL"):
    """
    从视频路径列表生成单个 clip 的描述。
    返回值: List[caption string]
    """
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto", low_cpu_mem_usage=True).eval()

    min_pixels = 256*28*28
    max_pixels = 1280*28*28
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)

    merged_imgs = [video2img(path) for path in video_clip_paths]

    # 存储为临时路径
    temp_paths = []
    os.makedirs(temp_dir, exist_ok=True)
    for i, img in enumerate(merged_imgs):
        temp_path = f"{temp_dir}/clip_{i}.jpg"
        save_image_temp(img, temp_path)
        temp_paths.append(temp_path)

    captions = cal_response_single(temp_paths, response_all, model, processor)
    return captions

video_clip_paths = [
    '',
    '',
    ''
    ]

response_all = ""
temp_dir = ''
captions = generate_single_clip_captions(video_clip_paths, response_all, temp_dir)
for i, caption in enumerate(captions):
    print(f"Clip {i}:", caption)
