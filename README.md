# 📊 Quantitative Evaluation for CoI-VID

This repository contains the quantitative evaluation resources for the CoI-VID dataset, including evaluation code, prompts, visualizations, and sample annotations. We provide three complementary evaluation approaches:

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


- **Models Compared**: Baseline (trained on Emu3) vs Fine-tuned (further finetuned on CoI-VID).

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

- **Procedure**:  GPT-4o is prompted to assess model outputs along the following six dimensions:
<small>

- 1. **Stylistic Consistency**  
  The visual styles of the six clips should remain consistent — including aspects like color tone, lighting, rendering technique, and texture details.

- 2. **Entity Consistency**  
  Key characters and objects should retain consistent identity and attributes across all clips.

- 3. **Background Consistency**  
  Backgrounds and environments should remain visually and semantically consistent across clips.

- 4. **Perspective Transition Coherence**  
  Transitions between scenes and camera angles should be smooth and logically aligned.

- 5. **Text Prompt Alignment**  
  The generated video sequence should accurately reflect the meaning and intention of the original text prompt.

- 6. **Visual Plausibility**  
  Overall visual quality should be realistic, with no obvious artifacts, glitches, or implausible elements.

</small>


- **Score Guide**:

<small>

  - **5 - Very Excellent:**  
    Perfect. All 6 clips are flawless and demonstrate outstanding consistency and execution across all dimensions.

  - **4 – Excellent:**  
    Flawless. All clips are fully correct with no noticeable issues.

  - **3 – Good:**  
    Nearly Flawless. All clips are correct with only minor, negligible imperfections.

  - **2 – Fair:**  
    Some Flaws. Minor flaws observed in one clip.

  - **1 – Poor:**  
    Significant Flaws. Major or multiple flaws present in more than one clip.
</small>

> 🧪 Averaged over 6 evaluations per sample (1 full + 5 pairwise), with VLM calibration via reference examples.

---

- **Result Summary**:

| Model       | Style | Entity | Background | Perspective | Prompt Align | Visual |
|-------------|-------|--------|------------|-------------|--------------|--------|
| Baseline    | 1.06  | 1.28   | 0.77       | 1.78        | 3.13         | 2.87   |
| +InterVID   | 0.92  | 1.07   | 0.72       | 1.45        | 2.61         | 2.68   |
| +OpenVID    | 1.01  | 1.44   | 0.80       | 1.66        | 2.98         | **3.05** |
| **+CoI-VID** | **2.91** | **3.35** | **2.65** | **2.81** | **3.41** | 2.94 |


---

## 🎯 3. Similarity-based Evaluation


* 📁 **Provided Files**
  * <small>`similarity_evaluation/object_similarity_data.jsonl` → *Captions and first clips for similarity-based evaluation.*</small>  
  * <small>`similarity_evaluation/object_similarity_evaluation.py` → Code for computing similarity evaluation.
  * <small>`middle_frames.zip` → *Ground-truth middle frames for similarity evaluation. 💡download via: wget https://flagchat.ks3-cn-beijing.ksyuncs.com/TVinterleve/middle_frames_for_sim_eval.zip </small>  
  * <small>`rectangles.zip` → *Manually filtered object bounding boxes on ground-truth video frames for similarity evaluation. 💡download via: wget https://flagchat.ks3-cn-beijing.ksyuncs.com/TVinterleve/rectangles.zip </small>  
  * <small>`CoI-VID_results.zip` → *Visualizations of results from the CoI-VID fine-tuned model. 💡download via: wget https://flagchat.ks3-cn-beijing.ksyuncs.com/TVinterleve/CoI-VID_sim_results.zip </small>  
  * <small>`observation_for_object_similarity_data.zip` → *Visual observation files for similarity-based evaluation data. 💡download via: wget https://flagchat.ks3-cn-beijing.ksyuncs.com/TVinterleve/observation_for_object_similarity_data.zip </small>  

---


We construct a similarity-based evaluation dataset based on CoI-VID data. To avoid data leakage, all test data and data from the same source videos are excluded from the CoI-VID training set. This evaluation compares the similarity between the generated and ground-truth videos at both the **global** and **object** levels.

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
| +InterVID   | 0.497  | 0.296     | 0.187  | 0.597        | 0.354           | 0.268        |
| +OpenVID    | 0.506  | 0.316     | 0.201  | 0.602        | 0.351           | 0.248        |
| **+CoI-VID** | **0.670**  | **0.381**     | **0.272**  | **0.702**        | **0.412**           | **0.391**        |


---


