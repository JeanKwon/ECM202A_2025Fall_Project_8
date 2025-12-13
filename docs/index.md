
# **Context-Aware Hybrid Object Detection for Autonomous Vehicle Perception**

![Project Banner](./assets/img/banner-placeholder.png)

---

## 👥 **Team**

- Hee Jean Kwon
---

## 📝 **Abstract**

This project presents a context-aware hybrid perception framework for autonomous vehicles that dynamically selects between local and cloud-based object detection to balance latency, accuracy, and resource usage. The system leverages multi-modal contextual inputs—including scene complexity, vehicle state, and network conditions to make real-time offloading decisions using a data-driven model selector. Lightweight local models provide low-latency inference under constrained conditions, while more accurate cloud models are selectively used for complex scenes when network latency permits. Experimental results on the Waymo Perception Dataset demonstrate that the hybrid policy achieves approximately 75% of the accuracy benefits of full cloud offloading while using cloud resources only about one-third of the time. These results demonstrate the feasbility of edge–cloud collaboration for safe and efficient autonomous vehicle perception under constraints.

---

## 📑 **Slides**

- [Midterm Checkpoint Slides](https://github.com/JeanKwon/ECM202A_2025Fall_Project_8/blob/main/docs/assets/Midterm%20Presentation%20.pptx%20(2).pdf)  
- [Final Presentation Slides](https://github.com/JeanKwon/ECM202A_2025Fall_Project_8/blob/main/docs/assets/Final%20Presentation%20.pptx.pdf)

---

# **1. Introduction**

### **1.1 Motivation & Objective**  
It is crucial for autonomous vehciels to detect surrdouning objects quicky and accrautely. Running advanced object detection models entirely on the vehicle reduces dependence on network connectivity but is limited by onboard compute and power budgets. On the other hand, cloud-based processing offers better accuracy but introduces variable network latency and reliability concerns. The objective of this project is to design a perception system that decides, in real time, whether to perform object detection locally or offload it to the cloud based on current driving context and system conditions. By doing so, the system aims to maximize perception accuracy while respecting strict latency and resource constraints.

### **1.2 State of the Art & Its Limitations**  
How is this problem addressed today?  
What gaps or limitations exist?  
Cite prior work using a consistent style like [Smith21].

### **1.3 Novelty & Rationale**  
The key novelty of this project is in its context-aware, data-driven model selection policy for hybrid perception. Instead of using predefined rules, the system employs learned decision models that incorporate multi-modal context, including scene complexity, vehicle state, and system/network conditions. A lightweight CNN-based gating model enables rapid scene assessment with minimal overhead, allowing the system to select the most appropriate detection model under a fixed latency budget. This system uses more computing power only when the scene is complex, while handling simpler situations locally, which improves efficiency without reducing safety.

### **1.4 Potential Impact**  
This project could demonstrate a scalable framework for adaptive edge–cloud collaboration under strict real-time constraints. It could reduce operational cloud costs, improve system robustness under variable network conditions, and contribute to safer autonomous driving. The framework is also extensible to other perception tasks and robotic systems beyond autonomous vehicles.

### **1.5 Challenges**  
The project could have several challenges such as designing a decision model that is both accurate and computationally lightweight, handling variability in network conditions in real time, and ensuring that fallback mechanisms preserve safety when issues emerge in cloud system.


### **1.6 Metrics of Success**  
The successs of project is evaludated using the following metrics. 
Perception Accuracy: Reduction in object count error compared to local-only inference.
Cloud Utilization Rate: Percentage of frames offloaded to the cloud relative to total frames processed.
Robustness: System behavior under varying network availability, including correct fallback to local processing.

---

# **2. Related Work**

Summarize prior works relevant to your project.  
For each: what did the authors do, how is it related, and what gap remains?

Reference all citations in **Section 6**.

---

# **3. Technical Approach**

Describe your system, methodology, algorithms, and design choices.  
Use figures generously:

- System architecture diagram  
- Data pipeline  
- Algorithm/model block diagram  
- Hardware setup photos  

💡 Tip: Add images, diagrams, and code snippets. Make your system reproducible.

Recommended subsections:

### **3.1 System Architecture**
![Project Banner](./assets/img/system_architecture.png)

### **3.2 Data Pipeline**
Explain how data is collected, processed, and used.

### **3.3 Algorithm / Model Details**
![Project Banner](./assets/img/systemflow.jpg)

### **3.4 Hardware / Software Implementation**
Explain equipment, libraries, or frameworks.

### **3.5 Key Design Decisions & Rationale**
Describe the main design decisions you made.

---

# **4. Evaluation & Results**

Present experimental results with clarity and professionalism.

Include:

- Plots (accuracy, latency, energy, error curves)  
- Tables (comparisons with baselines)  
- Qualitative visualizations (spectrograms, heatmaps, bounding boxes, screenshots)  
- Ablation studies  
- Error analysis / failure cases

Each figure should have a caption and a short interpretation.

---

# **5. Discussion & Conclusions**

Synthesize the main insights from your work.

- What worked well and why?  
- What didn’t work and why?  
- What limitations remain?  
- What would you explore next if you had more time?  

This should synthesize—not merely repeat—your results.

---

# **6. References**

Provide full citations for all sources (academic papers, websites, etc.) referenced and all software and datasets uses.

---

# **7. Supplementary Material**

## **7.a. Datasets**

Waymo Perception Dataset V2.0.1 
* [Source and URL](https://waymo.com/open/download/)
* Data format: The dataset is stored as Apache Parquet (.parquet) tables for each sensor/annotation modality. Records are synchronized across modalities using the composite frame identifiers segment_context_name (drive segment ID) and frame_timestamp_micros (frame timestamp), which together uniquely specify a single frame.
* Preprocessing steps: Because the dataset is modularized across separate folders/files by modality, I reconstructed per-frame samples by joining the required sensor and annotation tables using segment_context_name and frame_timestamp_micros. For each frame, I extracted the relevant fields and exported consolidated per-frame features to CSV for training and evaluation. To support intuitive inspection and debugging, front-facing RGB camera frames are converted to JPEG images. 
* Labeling/annotation efforts:The Waymo dataset is already annotated, so no additional manual is required. Relevant annotation-derived features are extracted and stored in a consolidated CSV indexed by the frame identifiers segment_context_name and frame_timestamp_micros to maintain alignment across modalities. An additional scene-level “complexity” label is needed for gating decisions model selector training. These labels are generated using unsupervised clustering. K-Means is applied to group frames based on traffic density (number of vehicles and pedestrians), ego-vehicle speed, and brightness. Cluster-level feature averages are then analyzed, and the cluster with higher object density and more dynamic driving conditions is labeled “complex,” while the other cluster is labeled “simple.”
* [Processed Data](https://github.com/JeanKwon/ECM202A_2025Fall_Project_8/blob/main/data/master_data.csv)

## **7.b. Software**

List:
* Python 3.9, NumPy 1.24.2, OpenCV 4.6.0, Pillow 9.2.0, ONNX Runtime 1.19.2, YOLO 8.3.230
* Links to repos 

---

> [!NOTE] 
> Read and then delete the material from this line onwards.

# 🧭 **Guidelines for a Strong Project Website**

- Include multiple clear, labeled figures in every major section.  
- Keep the writing accessible; explain acronyms and algorithms.  
- Use structured subsections for clarity.  
- Link to code or datasets whenever possible.  
- Ensure reproducibility by describing parameters, versions, and preprocessing.  
- Maintain visual consistency across the site.

---

# 📊 **Minimum vs. Excellent Rubric**

| **Component**        | **Minimum (B/C-level)**                                         | **Excellent (A-level)**                                                                 |
|----------------------|---------------------------------------------------------------|------------------------------------------------------------------------------------------|
| **Introduction**     | Vague motivation; little structure                             | Clear motivation; structured subsections; strong narrative                                |
| **Related Work**     | 1–2 citations; shallow summary                                 | 5–12 citations; synthesized comparison; clear gap identification                          |
| **Technical Approach** | Text-only; unclear pipeline                                  | Architecture diagram, visuals, pseudocode, design rationale                               |
| **Evaluation**       | Small or unclear results; few figures                          | Multiple well-labeled plots, baselines, ablations, and analysis                           |
| **Discussion**       | Repeats results; little insight                                | Insightful synthesis; limitations; future directions                                      |
| **Figures**          | Few or low-quality visuals                                     | High-quality diagrams, plots, qualitative examples, consistent style                      |
| **Website Presentation** | Minimal formatting; rough writing                           | Clean layout, good formatting, polished writing, hyperlinks, readable organization        |
| **Reproducibility**  | Missing dataset/software details                               | Clear dataset description, preprocessing, parameters, software environment, instructions   |
