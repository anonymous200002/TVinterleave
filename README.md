# 🧩 CI-VID Construction  
* 📁 **Provided Files**
  * <small>`CI-VID_construction_functions/`</small>

This part of the repository contains the core code for constructing the CI-VID dataset. Five key functions have been modularized to facilitate experimentation and integration into new workflows.

- **Functions**:
   - *clip_similarity_detection_and_segment*
   - *clip_seq_main_object_detection*
   - *clip_seq_single_annotation*
   - *clip_seq_joint_annotation*


# 🗂️ Samples of CI-VID
* 📁 **Provided Files**
  * <small>`CI-VID_samples_for_visualization/`
    
This part of the repository contains samples extracted from CI-VID to better illustrate the dataset’s structure and characteristics.


# 📊 Quantitative Evaluation for CI-VID

This part of the repository contains the quantitative evaluation resources for the CI-VID dataset, including evaluation code, prompts, visualizations, and sample annotations. We provide three complementary evaluation approaches:

## 🔍 Overview

We propose three evaluation protocols:

1. **Human Evaluation**  
2. **VLM-based Evaluation**  
3. **Similarity-based Evaluation**

---

## 👥 1. Human Evaluation

* 📁 **Provided Files**
  * <small>`human_evaluation/prompts.jsonl` → *Prompts used for evaluation.*</small>  
  * <small>`human_evaluation/visual_contrast/` → *Visualizations for human evaluation (1,000 prompts). 💡download via: wget https://flagchat.ks3-cn-beijing.ksyuncs.com/TVinterleve/visual_contrast.zip </small>  

---


Human evaluation is based on 1,000 automatically generated prompts (Seeded with keywords from VBench), with each describing 6 scenes composing a coherent multi-scene narratives.


- **Models Compared**: Baseline (trained on Emu3) vs Fine-tuned (further finetuned on CI-VID).

- **Examples**:
<img src="https://flagchat.ks3-cn-beijing.ksyuncs.com/TVinterleve/130.jpg" border=0 width=70%>
<img src="https://flagchat.ks3-cn-beijing.ksyuncs.com/TVinterleve/358.jpg" border=0 width=70%>
<img src="https://flagchat.ks3-cn-beijing.ksyuncs.com/TVinterleve/093.jpg" border=0 width=70%>
<img src="https://flagchat.ks3-cn-beijing.ksyuncs.com/TVinterleve/137.jpg" border=0 width=70%>
<img src="https://flagchat.ks3-cn-beijing.ksyuncs.com/TVinterleve/181.jpg" border=0 width=70%>
<img src="https://flagchat.ks3-cn-beijing.ksyuncs.com/TVinterleve/012.jpg" border=0 width=70%>



- **Procedure**: 3 professional annotators compare model outputs across:
<small>
  -  *Consistency*
  -  *Narrativity*
  -  *Factual correctness*
</small>

- **Judgment Format**: Side-by-side comparison, anonymized model identity, random top-bottom order.

---

- **Result Summary**:

| Metric        | Win    | Tie   | Loss  |
|---------------|--------|-------|--------|
| Consistency   | **90.0%** | 6.5% | 3.6% |
| Narrativity   | **80.9%** | 15.0% | 4.1% |
| Correctness   | **78.3%** | 9.8% | 11.9% |


---

## 🤖 2. VLM-based Evaluation
* 📁 **Provided Files**
  * <small>`vlm_evaluation/vlm_evaluation_data.jsonl` → *Prompts used for evaluation.*</small>  
  * <small>`vlm_evaluation/vlm_evaluation.py` → Code for VLM-based evaluation </small>  
---


We use the same prompts as human evaluation perform a VLM-based evaluation.

- **Procedure**:  Qwen2-VL-72B-Instruct is prompted to assess model outputs along the following six dimensions:
<small>

- 1. **Stylistic Consistency**  
  the visual styles of frames need to be consistent (e.g., color tone, lighting, rendering technique, texture details)

- 2. **Entity Consistency**  
  the key characters and objects need to be consistent across frames. (e.g., retain the same attributes and identity)

- 3. **Background Consistency**  
  the backgrounds and environments need to be consistent across frames? 

- 4. **Perspective Transition Coherence**  
  the transitions between camera angles and scenes need to be smooth and logically aligned.

- 5. **Text Prompt Alignment**  
  the frames need to be accurately reflect the content and intent of the original text prompts.prompt.

- 6. **Visual Plausibility**  
  is the overall visual quality realistic? Are there any noticeable artifacts, glitches, or implausible elements?

</small>


- **Score Guide**:

<small>

  - **5 - Very Excellent:**  
    Perfect: not only flawless but demonstrate outstanding consistency and execution.
    
  - **4 – Excellent:**  
    Flawless: no noticeable issues.

  - **3 – Good:**  
    Nearly flawless, with only minor, negligible imperfections. 

  - **2 – Fair:**  
    Minor flaws observed in one clip.

  - **1 – Poor:**  
    Major or multiple flaws.

  - **0 – Very Poor:**  
    Multiple (> 1) major flaws.
</small>

> 🧪 Averaged over 6 evaluations per sample (1 full + 5 pairwise), with VLM calibration via reference examples.

---
- **Result Summary**:

| Model       | Style | Entity | Background | Perspective | Prompt Align | Visual |
|-------------|-------|--------|------------|-------------|--------------|--------|
| Baseline    | 3.07   | 2.84   | 2.80       | 3.02       | 3.99       | 3.25   |
| **+CI-VID** | **3.83** | **3.73** | **3.75** | **3.81** | **4.07** | 3.62  |

---


## 🎯 3. Similarity-based Evaluation


* 📁 **Provided Files**
  * <small>`similarity_evaluation/object_similarity_data.jsonl` → *Captions and first clips for similarity-based evaluation.*</small>  
  * <small>`similarity_evaluation/object_similarity_evaluation.py` → Code for computing similarity evaluation.
  * <small>`middle_frames.zip` → *Ground-truth middle frames for similarity evaluation. 💡download via: wget https://flagchat.ks3-cn-beijing.ksyuncs.com/TVinterleve/middle_frames_for_sim_eval.zip </small>  
  * <small>`rectangles.zip` → *Manually filtered object bounding boxes on ground-truth video frames for similarity evaluation. 💡download via: wget https://flagchat.ks3-cn-beijing.ksyuncs.com/TVinterleve/rectangles.zip </small>  
  * <small>`CI-VID_results.zip` → *Visualizations of results from the CI-VID fine-tuned model. 💡download via: wget https://flagchat.ks3-cn-beijing.ksyuncs.com/TVinterleve/CoI-VID_sim_results.zip </small>  
  * <small>`observation_for_object_similarity_data.zip` → *Visual observation files for similarity-based evaluation data. 💡download via: wget https://flagchat.ks3-cn-beijing.ksyuncs.com/TVinterleve/observation_for_object_similarity_data.zip </small>  

---


We construct a similarity-based evaluation dataset based on CI-VID data. To avoid data leakage, all test data and data from the same source videos are excluded from the CI-VID training set. This evaluation compares the similarity between the generated and ground-truth videos at both the **global** and **object** levels.

---

### ⚙️ Evaluation Setup

1. **Object Detection**:  
   YOLO is applied to each video clip. For every clip, 3 frames are uniformly sampled and processed.

2. **Manual Filtering**:  
   Non-essential objects are removed manually. A maximum of two narrative-relevant object boxes are kept per frame.

3. **Evaluation Protocol**:  
   - Each sample includes the **first clip** and the **full caption** as input.  
   - The evaluated model generates the remaining video clips.
   - We compute similarity between:
     - Generated and ground-truth **middle frames** → for **whole-sequence similarity**
     - Object boxes in generated and reference frames → for **object-level similarity**
   - For object similarity, we match each generated object to ground-truch object across 3 frames per clip, and use the best score as the clip score, then average all clip scores as sample score. The final results are the average of all samples.

- **Ground-truth Examples**:
<img src="https://flagchat.ks3-cn-beijing.ksyuncs.com/TVinterleve/200.jpg" border=0 width=70%>
<img src="https://flagchat.ks3-cn-beijing.ksyuncs.com/TVinterleve/201.jpg" border=0 width=70%>
<img src="https://flagchat.ks3-cn-beijing.ksyuncs.com/TVinterleve/388.jpg" border=0 width=70%>

---


- **Result Summary**:

| Model       | CLIP ↑ | 1-LPIPS ↑ | SSIM ↑ | CLIP (Obj) ↑ | 1-LPIPS (Obj) ↑ | SSIM (Obj) ↑ |
|-------------|--------|-----------|--------|--------------|------------------|--------------|
| Baseline    | 0.512  | 0.309     | 0.199  | 0.601        | 0.360           | 0.278        |
| **+CI-VID** | **0.670**  | **0.381**     | **0.272**  | **0.702**        | **0.412**           | **0.391**        |


---


