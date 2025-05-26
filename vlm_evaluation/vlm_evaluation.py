import json, re
import os
import random
from tqdm import tqdm
import math
import shutil
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

def cal_llm(img, txt_prompt, example_prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": img,
                },
                {"type": "text", "text": txt_prompt},
                {"type": "text", "text": example_prompt},
            ],
        }
    ]

    chat_response = client.chat.completions.create(
        model="Qwen2-VL-72B-Instruct",
        messages=messages) 

    response_txt =  json.loads(chat_response.json())["choices"][0]["message"]["content"]
    print(response_txt)
    return response_txt


def code_img(img):
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
    keys = ["Stylistic Consistency", "Entity Consistency", "Background Consistency", "Perspective Transition Coherence"]
    for key in keys:
        score = data[key]["score"]
        assert score in ['0', '1', '2', '3', '4', '5',0, 1, 2, 3, 4, 5]
        score = int(score)
        score_list.append(score)
    return score_list



def decode_response_nointerleave(response):
    match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
    
    json_str = match.group(1)
    
    data = json.loads(json_str)
    score_list = []
    keys = ["Text Prompt Alignment", "Visual Plausibility"]
    for key in keys:
        score = data[key]["score"]
        assert score in ['0', '1', '2', '3', '4', '5',0, 1, 2, 3, 4, 5]
        score = int(score)
        score_list.append(score)
    return score_list


def process_input(evaluation_data_path, whole_frame_path, pair_frame_path, n=50):
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

    whole_frames = sorted(
        [os.path.join(whole_frame_path, fname) for fname in os.listdir(whole_frame_path) if fname.endswith(".jpg")]
    )

    subdirs = sorted([d for d in os.listdir(pair_frame_path) if os.path.isdir(os.path.join(pair_frame_path, d))])

    pair_frames = []

    for subdir in subdirs:
        subdir_path = os.path.join(pair_frame_path, subdir)
        images = sorted([
            os.path.join(subdir_path, fname)
            for fname in os.listdir(subdir_path)
            if fname.endswith(".jpg")
        ])
        pair_frames.append(images)
        
    return evalution_datas, whole_frames[:n], pair_frames[:n]


def process_whole(line):
    whole_frame, merged_frames, text_prompts = line
    narratives = text_prompts['narratives']


    all_score_list = []
    merged_frame = code_img(whole_frame)

    txt_prompt = f"""
            The top row shows exmaple frames and the bottom row shows target frames, both contain six frames.
            Your task is to assess the quality of the bottom row based on the following four dimensions:

            1. **Stylistic Consistency** –  the visual styles of frames need to be consistent (e.g., color tone, lighting, rendering technique, texture details) 
            2. **Entity Consistency** – the key characters and objects need to be consistent across frames. (e.g., retain the same attributes and identity)
            3. **Background Consistency** – the backgrounds and environments need to be consistent across frames? 
            4. **Perspective Transition Coherence** – the transitions between camera angles and scenes need to be smooth and logically aligned.
            ---
            Score Guide:
            - 5 = Very Excellent – Perfect: not only flawless but demonstrate outstanding consistency and execution.
            - 4 = Excellent – Flawless: no noticeable issues.
            - 3 = Good –  Nearly flawless, with only minor, negligible imperfections. 
            - 2 = Fair – Minor flaws observed in one clip.
            - 1 = Poor – Major or multiple flaws.
            - 0 = Very Poor – Multiple (> 1) major flaws.
        """

    example_prompt = f"""
                For example, in the top row, there are none clear inconsistencies across frames in terms of visual style. (3 points). However the appearance of surfboard and character cloth changed clearly (0 points). Thought the beach and sea looks consistent, the cloud shape in background changed a little (2 ponits). The Perspective Transition is natural between frames (3 points).
                Important: Please examine each clip carefully and make an effort to identify any possible flaws or inconsistencies. Only assign high scores when the criteria are fully satisfied. If you observe any issue, even if it seems minor, consider giving a lower score accordingly.

                ```json
                {{
                "Stylistic Consistency": {{
                    "score": int,
                }},
                "Entity Consistency": {{
                    "score": int,
                }},
                "Background Consistency": {{
                    "score": int,
                }},
                "Perspective Transition Coherence": {{
                    "score": int,
                }},
                }}
                ```           
                """

    txt_prompt_nointer = f"""
            The top row shows exmaple frames and the bottom row shows target frames, both contain six frames.

            Bottom frames generated based on the text prompts:
            {narratives};

            Your task is to assess the quality of the bottom row based on the following two dimensions:

            1. **Text Prompt Alignment** – the frames need to be accurately reflect the content and intent of the original text prompts.
            2. **Visual Plausibility** – Is the overall visual quality realistic? Are there any noticeable artifacts, glitches, or implausible elements?
            ---
            Score Guide:
            - 5 = Very Excellent – Perfect: not only flawless but demonstrate outstanding consistency and execution.
            - 4 = Excellent – Flawless: no noticeable issues.
            - 3 = Good –  Nearly flawless, with only minor, negligible imperfections. 
            - 2 = Fair – Minor flaws observed in one clip.
            - 1 = Poor – Major or multiple flaws.
            - 0 = Very Poor – Multiple (> 1) major flaws.
        """

    example_prompt_nointer = f"""
                For example, in the top row, the overall content is generally accurate. However, in the second frame, the position of the surfboard is incorrect. In the fourth frame, the gestures associated with discussion are not clearly depicted (1 point). In the third frame, there is a visual error in the depiction of the boy, and in the fifth frame, the positioning of the skateboarding boy’s feet is visibly incorrect (0 points)
                Important: Please examine each clip carefully and make an effort to identify any possible flaws or inconsistencies. Only assign high scores when the criteria are fully satisfied. If you observe any issue, even if it seems minor, consider giving a lower score accordingly.

                ```json
                "Text Prompt Alignment": {{
                    "score": int,
                }},
                "Visual Plausibility": {{
                    "score": int,
                }},
                ```  
        """


    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        score_list_inter = [0, 0, 0, 0,]
        try:
            response = cal_llm(merged_frame, txt_prompt, example_prompt)
            score_list_inter = decode_response(response)
            break  
        except Exception as e:
            print(f"[Attempt {attempt}] ❌ Failed to decode response: {e}")
            if attempt == max_attempts:
                raise  

    for attempt in range(1, max_attempts + 1):
        score_list_nointer = [0, 0]
        try:
            response = cal_llm(merged_frame, txt_prompt_nointer, example_prompt_nointer)
            score_list_nointer = decode_response_nointerleave(response)
            break  
        except Exception as e:
            print(f"[Attempt {attempt}] ❌ Failed to decode response: {e}")
            if attempt == max_attempts:
                raise  

    score_list = score_list_inter + score_list_nointer
                
    return [score_list]

def process(line):
    whole_frame, merged_frames, text_prompts = line
    narratives = text_prompts['narratives']


    all_score_list = []

    for i in range(5):
        merged_frame = merged_frames[i]
        merged_frame = code_img(merged_frame)

        pre_sence, tal_sence = narratives[f'sence_{i+1}'], narratives[f'sence_{i+2}']


        txt_prompt = f"""
                The top row shows exmaple frames and the bottom row shows target frames, both contain two frames.
                Your task is to assess the quality of the target frames (bottom row) based on the following dimensions:

                1. **Stylistic Consistency** –  the visual styles of frames need to be consistent (e.g., color tone, lighting, rendering technique, texture details) 
                2. **Entity Consistency** – the key characters and objects need to be consistent across frames. (e.g., retain the same attributes and identity)
                3. **Background Consistency** – the backgrounds and environments need to be consistent across frames? 
                4. **Perspective Transition Coherence** – the transitions between camera angles and scenes need to be smooth and logically aligned.
                ---
                Score Guide:
                - 5 = Very Excellent – Perfect: not only flawless but demonstrate outstanding consistency and execution.
                - 4 = Excellent – Flawless: no noticeable issues.
                - 3 = Good –  Nearly flawless, with only minor, negligible imperfections. 
                - 2 = Fair – Minor flaws observed in one clip.
                - 1 = Poor – Major or multiple flaws.
                - 0 = Very Poor – Multiple (> 1) major flaws.
            """

        example_prompt = f"""
                    For example, in the top row, there are none clear inconsistencies across frames in terms of visual style. (3 points). However the cloth of character changed (1 points). Thought the beach and sea looks consistent, the cloud shape in background changed (2 ponits). The Perspective Transition is natural (3 points).
                    Important: Please examine each clip carefully and make an effort to identify any possible flaws or inconsistencies. Only assign high scores when the criteria are fully satisfied. If you observe any issue, even if it seems minor, consider giving a lower score accordingly.

                    ```json
                    {{
                    "Stylistic Consistency": {{
                        "score": int,
                    }},
                    "Entity Consistency": {{
                        "score": int,
                    }},
                    "Background Consistency": {{
                        "score": int,
                    }},
                    "Perspective Transition Coherence": {{
                        "score": int,
                    }},
                    }}
                    ```           
                    """

        txt_prompt_nointer = f"""
                The top row shows exmaple frames and the bottom row shows target frames, both contain two frames.

                Bottom frames generated based on two text prompts:
                {pre_sence};
                {tal_sence};

                Your task is to assess the quality of the target frames (bottom row) based on the following dimensions:

                1. **Text Prompt Alignment** – the frames need to be accurately reflect the content and intent of the original text prompts.
                2. **Visual Plausibility** – Is the overall visual quality realistic? Are there any noticeable artifacts, glitches, or implausible elements?
                ---
                Score Guide:
                - 5 = Very Excellent – Perfect: not only flawless but demonstrate outstanding consistency and execution.
                - 4 = Excellent – Flawless: no noticeable issues.
                - 3 = Good –  Nearly flawless, with only minor, negligible imperfections. 
                - 2 = Fair – Minor flaws observed in one clip.
                - 1 = Poor – Major or multiple flaws.
                - 0 = Very Poor – Multiple (> 1) major flaws.
            """

        example_prompt_nointer = f"""
                    For example, in the top row, The content is generally correct, but the gestures, discussing are not well presented (2 points). The position of the skateboarding boy's feet is obviously wrong (1 points).
                    Important: Please examine each clip carefully and make an effort to identify any possible flaws or inconsistencies. Only assign high scores when the criteria are fully satisfied. If you observe any issue, even if it seems minor, consider giving a lower score accordingly.

                    ```json
                    "Text Prompt Alignment": {{
                        "score": int,
                    }},
                    "Visual Plausibility": {{
                        "score": int,
                    }},
                    ```  
            """


        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            score_list_inter = [0, 0, 0, 0,]
            try:
                response = cal_llm(merged_frame, txt_prompt, example_prompt)
                score_list_inter = decode_response(response)
                break  
            except Exception as e:
                print(f"[Attempt {attempt}] ❌ Failed to decode response: {e}")
                if attempt == max_attempts:
                    raise  

        score_list_nointer = [0, 0]
 
        for attempt in range(1, max_attempts + 1):
            score_list_nointer = [0, 0]
            try:
                response = cal_llm(merged_frame, txt_prompt_nointer, example_prompt_nointer)
                score_list_nointer = decode_response_nointerleave(response)
                break  
            except Exception as e:
                print(f"[Attempt {attempt}] ❌ Failed to decode response: {e}")
                if attempt == max_attempts:
                    raise  

        score_list = score_list_inter + score_list_nointer
        all_score_list.append(score_list)
                
    return all_score_list


openai_api_key = "EMPTY"
openai_api_base = f"http://localhost:{8002}/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
TIMEOUT = 2000

root = ''


evaluation_data_path = f'{root}/llm_evaluation_data.jsonl'
whole_frame_path = f"{root}/result"
pair_frame_path = f"{root}/result"


evalution_datas, whole_frames, pair_frames = process_input(evaluation_data_path, whole_frame_path, pair_frame_path, 1000)

all_results = []

with ThreadPoolExecutor(max_workers=32) as executor:
    future_to_line = {
        executor.submit(process_whole, line): line for line in zip(whole_frames[:], pair_frames[:], evalution_datas)
    }

    for future in as_completed(future_to_line):
        try:
            result = future.result()  
            all_results.extend(result)
        except Exception as e:
            print('Error during processing:', e)

with ThreadPoolExecutor(max_workers=32) as executor:
    future_to_line = {
        executor.submit(process, line): line for line in zip(whole_frames[:], pair_frames[:], evalution_datas)
    }

    for future in as_completed(future_to_line):
        try:
            result = future.result()  
            all_results.extend(result)
        except Exception as e:
            print('Error during processing:', e)


num_rows = len(all_results)
num_cols = len(all_results[0])

keys = ["Stylistic Consistency", "Entity Consistency", "Background Consistency", "Perspective Transition Coherence", "Text Prompt Alignment", "Visual Plausibility"]

mean_values = {keys[i]:
    sum(row[i] for row in all_results) / num_rows
    for i in range(num_cols)
}

print("avg per dimension:", mean_values)

