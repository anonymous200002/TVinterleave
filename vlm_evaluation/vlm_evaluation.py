import json, re
import os
import random
from tqdm import tqdm
import math
import shutil
import openai
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed

def cal_llm(img, txt_prompt, example_img, example_prompt):

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": img,
                },
                {"type": "text", "text": txt_prompt},
                {
                    "type": "image_url",
                    "image_url": example_img,
                },
                {"type": "text", "text": example_prompt},
            ],
        }
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=3000
    )

    response_txt = response.choices[0].message.content
    print(response_txt)
    return response_txt


def code_img(img):
    # img = Image.open(img).convert('RGB')
    with open(img, "rb") as image_file:
        img = base64.b64encode(image_file.read()).decode('utf-8')
    img = {
                "url": f"data:image/jpeg;base64,{img}"
            }
    return img

def decode_response(response):
    match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
    
    json_str = match.group(1)
    
    data = json.loads(json_str)
    score_list = []
    keys = ["Stylistic Consistency", "Entity Consistency", "Background Consistency", "Perspective Transition Coherence", "Text Prompt Alignment", "Visual Plausibility"]
    for key in keys:
        score = data[key]["score"]
        assert score in ['1', '2', '3', '4', '5', 1, 2, 3, 4, 5]
        score = int(score)
        score_list.append(score)
    return score_list


def process_whole(line):
    frame_path, text_prompts = line
    narratives = str(text_prompts['narratives'])
    img = code_img(frame_path)
    example_img = code_img(example_path)

    # 
    txt_prompt = f"""
            There are 6 key frames extracted from the video sequence generated based on a sequence of 6 text prompts.

            ### text prompts:
            {narratives}
            ###

            Each frame corresponds to one sence in the text prompts, and the video as a whole is meant to form a coherent narrative.

            Your task is to assess the quality of the video sequence based on the following six dimensions:
            Be critical in your evaluation. Only assign a score of 5 if the dimension is nearly perfect. Use lower scores when you notice inconsistencies or room for improvement.


            1. **Stylistic Consistency** –  the visual styles of the six clips need to be consistent across all clips (e.g., color tone, lighting, rendering technique, texture details) 
            2. **Entity Consistency** – the key characters and objects need to be consistent across all clips. (e.g., retain the same attributes and identity)
            3. **Background Consistency** – the backgrounds and environments need to be  consistent across all clips? 
            4. **Perspective Transition Coherence** – the transitions between camera angles and scenes need to be  smooth and logically aligned.
            5. **Text Prompt Alignment** – the video sequence need to be accurately reflect the content and intent of the original text prompts.
            6. **Visual Plausibility** – Is the overall visual quality realistic? Are there any noticeable artifacts, glitches, or implausible elements?
            ---
            Score Guide:
            - 5 = Very Excellent – Perfect: all 6 clips are not only flawless but demonstrate outstanding consistency and execution across all dimensions.
            - 4 = Excellent – Flawless: all 6 clips are fully correct with no noticeable issues.
            - 3 = Good –  Nearly flawless: all clips are correct with only minor, negligible imperfections. 
            - 2 = Fair – Minor flaws observed in one clip.
            - 1 = Poor – Major or multiple flaws in more than one clip.
        """

    example_prompt = f"""
                For example, in the beach-themed video shown before, there are clear inconsistencies across clips in terms of visual style, character appearance, and background composition. Therefore, Stylistic Consistency, Entity Consistency, and Background Consistency should each be rated as 1 point.
                In addition, the Perspective Transition is abrupt and lacks coherent flow between clips, which justifies a rating of 2 points.
                Moreover, there are obvious visual errors — for instance, the person is merged with the skateboard, and an extra skateboard appears unnaturally in the scene. These flaws reduce Visual Plausibility, which should be rated as 1 point.     
                
                Important: Please examine each clip carefully and make an effort to identify any possible flaws or inconsistencies. Only assign high scores when the criteria are fully satisfied. If you observe any issue, even if it seems minor, consider giving a lower score accordingly.


                Return your answer in the following structured JSON format:

                ```json
                {{
                "Stylistic Consistency": {{
                    "score": [1–5],
                }},
                "Entity Consistency": {{
                    "score": [1–5],
                }},
                "Background Consistency": {{
                    "score": [1–5],
                }},
                "Perspective Transition Coherence": {{
                    "score": [1–5],
                }},
                "Text Prompt Alignment": {{
                    "score": [1–5],
                }},
                "Visual Plausibility": {{
                    "score": [1–5],
                }},
                }}
                ```           
                """
                    
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        score_list = [0, 0, 0, 0, 0, 0]
        try:
            response = cal_llm(img, txt_prompt, example_img, example_prompt)
            score_list = decode_response(response)
            break  # success break
        except Exception as e:
            print(f"[Attempt {attempt}] ❌ Failed to decode response: {e}")
            if attempt == max_attempts:
                raise  # error
            
    return score_list


def process_pair(line):
    merged_frames, text_prompts = line
    narratives = text_prompts['narratives']

    all_score_list = []
    example_img = code_img(example_path)
    for i in range(5):
        merged_frame = merged_frames[i]
        merged_frame = code_img(merged_frame)

        pre_sence, tal_sence = narratives[i], narratives[i+1]

        txt_prompt = f"""
                There are two video clips generated based on two text prompts.
                The top row shows frames from the first clip, and the bottom row shows frames from the second clip.

                ### text prompts:
                fist prompt:
                {pre_sence}
                second prompt:
                {tal_sence}
                ###

                The two clips as a whole are meant to form a coherent narrative.
                Your task is to assess the quality of the clips based on the following six dimensions:

                1. **Stylistic Consistency** –  the visual styles of the six clips need to be consistent across clips (e.g., color tone, lighting, rendering technique, texture details) 
                2. **Entity Consistency** – the key characters and objects need to be consistent across clips. (e.g., retain the same attributes and identity)
                3. **Background Consistency** – the backgrounds and environments need to be  consistent across clips? 
                4. **Perspective Transition Coherence** – the transitions between camera angles and scenes need to be  smooth and logically aligned.
                5. **Text Prompt Alignment** – the video sequence need to be accurately reflect the content and intent of the original text prompts.
                6. **Visual Plausibility** – Is the overall visual quality realistic? Are there any noticeable artifacts, glitches, or implausible elements?
                ---
                Score Guide:
                - 5 = Very Excellent – Perfect: all 6 clips are not only flawless but demonstrate outstanding consistency and execution across all dimensions.
                - 4 = Excellent – Flawless: all 6 clips are fully correct with no noticeable issues.
                - 3 = Good –  Nearly flawless: all clips are correct with only minor, negligible imperfections. 
                - 2 = Fair – Minor flaws observed in one clip.
                - 1 = Poor – Major or multiple flaws in more than one clip.
            """

        example_prompt = f"""
                    For example, in the beach-themed video shown before, there are clear inconsistencies across clips in terms of visual style, character appearance, and background composition. Therefore, Stylistic Consistency, Entity Consistency, and Background Consistency should each be rated as 1 point.
                    In addition, the Perspective Transition is abrupt and lacks coherent flow between clips, which justifies a rating of 2 points.
                    Moreover, there are obvious visual errors — for instance, the person is merged with the skateboard, and an extra skateboard appears unnaturally in the scene. These flaws reduce Visual Plausibility, which should be rated as 1 point.     
                    
                    Important: Please examine each clip carefully and make an effort to identify any possible flaws or inconsistencies. Only assign high scores when the criteria are fully satisfied. If you observe any issue, even if it seems minor, consider giving a lower score accordingly.
                    Return your answer in the following structured JSON format:

                    ```json
                    {{
                    "Stylistic Consistency": {{
                        "score": [1–5],
                    }},
                    "Entity Consistency": {{
                        "score": [1–5],
                    }},
                    "Background Consistency": {{
                        "score": [1–5],
                    }},
                    "Perspective Transition Coherence": {{
                        "score": [1–5],
                    }},
                    "Text Prompt Alignment": {{
                        "score": [1–5],
                    }},
                    "Visual Plausibility": {{
                        "score": [1–5],
                    }},
                    }}
                    ```           
                    """
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            score_list = [0, 0, 0, 0, 0, 0]
            try:
                response = cal_llm(merged_frame, txt_prompt, example_img, example_prompt)
                score_list = decode_response(response)
                break  # 
            except Exception as e:
                print(f"[Attempt {attempt}] ❌ Failed to decode response: {e}")
                if attempt == max_attempts:
                    raise  # 
        all_score_list.append(score_list)
                
    return all_score_list




TIMEOUT = 1200  # Timeout for OpenAI API requests (in seconds)

openai.api_key = ""
openai.api_base = ""

# Path to the prompt file provided in the benchmark dataset (in JSONL format)
evaluation_data_path = 'vlm_evaluation/vlm_evaluation_data.jsonl'

# Path to an example image (used for documentation or UI reference)
example_path = 'vlm_evaluation/0000.jpg'

# Path to pairwise frame composites: a 2×3 grid image for each sample, where 3 frames are extracted from each of 2 clips
pair_frame_path = ""

# Path to whole-sequence frame composites: a 1×6 image for each sample, where 6 frames are uniformly extracted across all clips
whole_frame_path = ""


evalution_datas = []
with open(evaluation_data_path, "r") as f:
    idx = 0
    buffer = ""
    for line in f:
        buffer += line
        try:
            item = json.loads(buffer)
            evalution_datas.append(item)
            buffer = ""  
            idx += 1
        except json.JSONDecodeError:
            continue 

with open(whole_frame_path, "r") as f:
    whole_frames = json.load(f)

with open(pair_frame_path, "r") as f:
    pair_frames = json.load(f)

all_results = []
with ThreadPoolExecutor(max_workers=128) as executor:
    future_to_line = {
        executor.submit(process_whole, line): line for line in zip(whole_frames[:], evalution_datas)
    }

    for future in as_completed(future_to_line):
        try:
            result = future.result()  
            all_results.extend(result)
        except Exception as e:
            print('Error during processing:', e)

with ThreadPoolExecutor(max_workers=128) as executor:
    future_to_line = {
        executor.submit(process_pair, line): line for line in zip(pair_frames[:], evalution_datas)
    }

    for future in as_completed(future_to_line):
        try:
            result = future.result()  
            all_results.extend(result)
        except Exception as e:
            print('Error during processing:', e)

num_rows = len(all_results)
num_cols = 6

mean_values = [
    sum(row[i] for row in all_results) / num_rows
    for i in range(num_cols)
]

print("avg per dimension:", mean_values)

