import os
import torch
import numpy as np
from loguru import logger
from tqdm import tqdm
import av
from PIL import Image
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from sklearn.metrics.pairwise import cosine_similarity


def cosine_similarity_between_embeddings(emb1, emb2):
    emb1 = emb1.cpu().numpy().reshape(1, -1)
    emb2 = emb2.cpu().numpy().reshape(1, -1)
    return cosine_similarity(emb1, emb2)[0][0]


def calculate_similarity(embeddings):
    similarities = []
    for i in range(len(embeddings) - 1):
        similarity = cosine_similarity_between_embeddings(embeddings[i], embeddings[i + 1])
        similarities.append(float(similarity))
    return similarities


def extract_frames_uniformly(video_path, num_frames=3):
    """
    将视频划分为 num_frames 段，从每段中间抽取一帧，返回 PIL.Image 列表。
    不保存帧到磁盘，仅返回图像对象。
    """
    container = av.open(video_path)
    stream = container.streams.video[0]
    total_frames = stream.frames

    if total_frames <= 0:
        print(f"Warning: Cannot get frame count for {video_path}, skipping.")
        return []

    # 每段取中间位置的帧索引
    split_size = total_frames / (num_frames + 1)
    indices = [int(split_size * (i + 1)) for i in range(num_frames)]

    frames = []
    for i, frame in enumerate(container.decode(stream)):
        if i in indices:
            img = frame.to_image()
            frames.append(img)
        if i > max(indices):
            break

    return frames

def save_horizontal_concat(imgs_list, save_dir, pad=10):
    """
    横向拼接每组图像，图像间隔 pad 像素，并保存为图片。

    Args:
        imgs_list: List[List[PIL.Image]]
        save_dir: 保存目录
        pad: 每张图之间的 padding，单位像素

    Returns:
        frame_paths: List[str] 保存的图片路径列表
    """
    os.makedirs(save_dir, exist_ok=True)
    frame_paths = []

    for i, imgs in enumerate(imgs_list):
        widths, heights = zip(*(img.size for img in imgs))
        total_width = sum(widths) + pad * (len(imgs) - 1)
        max_height = max(heights)

        new_img = Image.new('RGB', (total_width, max_height), color=(255, 255, 255))
        x_offset = 0
        for img in imgs:
            new_img.paste(img, (x_offset, 0))
            x_offset += img.size[0] + pad

        save_path = os.path.join(save_dir, f'video_frames_{i}.jpg')
        new_img.save(save_path)
        frame_paths.append(save_path)

    return frame_paths
    
def find_subsequences_with_conditions(lst, max_gap=5, min_length=2):
    subsequences = []  # 用来存储符合条件的子串的列表

    if len(lst) == 0:
        return []
    
    # 变量初始化
    current_subseq = [lst[0]]  # 初始子串包含第一个元素

    for i in range(1, len(lst)):
        # 判断当前数字与前一个数字的差是否在允许的最大间隔内
        current_value = int(lst[i].split('/')[-1].split('_')[0])
        prev_value = int(lst[i - 1].split('/')[-1].split('_')[0])
        
        # 判断当前数字与前一个数字的差是否在最大间隔范围内
        if current_value - prev_value <= max_gap:
            # 如果当前子串的长度小于 10，添加当前元素
            if len(current_subseq) < 10:
                current_subseq.append(lst[i])  # 如果符合条件，则加入当前子串
            else:
                # 如果当前子串已满，添加到结果列表，并开始新的子串
                subsequences.append(current_subseq)
                current_subseq = [lst[i]]  # 重新开始新的子串
        else:
            # 如果当前子串的长度大于最小长度，则添加到结果
            if len(current_subseq) >= min_length:
                subsequences.append(current_subseq)
            current_subseq = [lst[i]]  # 重新开始新的子串

    # 最后检查一次当前子串，确保它被添加到结果中
    if len(current_subseq) >= min_length:
        subsequences.append(current_subseq)

    return subsequences




def process_video_paths(video_paths, temp_dir):
    """
    输入：视频路径列表
    输出：sample 拼图路径列表，如 [{'video': ..., 'sample': ...}, ...]
    """
    os.makedirs(temp_dir, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = imagebind_model.imagebind_huge(pretrained=True).eval().to(device)

    samples = []

    frames = [extract_frames_uniformly(video_path, num_frames=3) for video_path in video_paths]

    frame_paths = save_horizontal_concat(frames, temp_dir)

    inputs = {
        ModalityType.VISION: data.load_and_transform_vision_data(frame_paths, device ),
    }
    with torch.no_grad():
        embeddings = model(inputs)[ModalityType.VISION]
        
    similarities = calculate_similarity(embeddings)
    similarities.insert(0, 0)
    print(similarities)
    temp = []
    for similarity, video_path in zip(similarities, video_paths):
        if similarity > 0.8:
            continue
        elif similarity > 0.6:
            temp.append(video_path)
        else:
            if temp != []:
                samples.append(temp)
            temp =[video_path]

    if temp != []:
        samples.append(temp)

    return samples


video_list = [
    '', '', '', '',
]

result = process_video_paths(video_list, temp_dir="")
for r in result:
    print(r)