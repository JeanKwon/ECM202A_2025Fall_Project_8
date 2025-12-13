
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

### **3.1 System Architecture**
![Project Banner](./assets/img/system_architecture.png)
Figure 1 System overview  
A diagram above illustrates the end-to-end flow. Incoming camera frames are processed in Jetson. Based on the scene compleixty and current network constraints, the system selects either a local YOLO 11n running on Jetson or a more accurate cloud-based YOLO 11x. Then Jetson requests cloud for YOLO 11x execution, and cloud sends the perception result back. 

### **3.2 Data Pipeline**
It is assumed that the RGB camera will provide an image frame that is in JPEG format. The image will be an input to lightweight CNN to decide wheter the scene is complex or not. After considering network condition and scene complexity, if the frame needs to be processed in cloud, Jetson sends over an image frame to cloud as shown in Figure 1. With the received image, the cloud runs a perception and sends back the perception result to the Jetson. 

### **3.3 Algorithm / Model Details**
![Project Banner](./assets/img/systemflow.jpg)
Figure 2 System Flow 
If it is available, then it checks if the cloud YOLO 11x execution time plus current network latency between Jetson and A6000 is smaller than the max perception latency allowed. Then, using the trained CNN, it checks if the scene is complex. If the frame meets the requirements, then the cloud YOLO 11x is used. If it fails to meet any network condition or the frame is classified to be simple, then local YOLo 11n is used. 

### **3.4 Hardware / Software Implementation**
Hardware:
* Jetson Orin
* A6000
Software Libraries:
* Python 3.8, NumPy 1.24.2, OpenCV 4.6.0, Pillow 9.2.0, ONNX Runtime 1.19.2, YOLO 8.3.230

### **3.5 Key Design Decisions & Rationale**
* Hybrid Edge–Cloud Architecture to balance latency guarantees and cost while maintaining higher accuracy than fully local YOLO 11n scenario.
* Latency-Aware Fallback Logic: If the cloud perception result does not arrive by the max perception latency allowed, the local perception result is used.
* Lightweight CNN Gating:Utilize a lightweight simple CNN to minimize decision time in time sensitive system. 
---

# **4. Evaluation & Results**
### **4.1 CNN Performance**
CNN takes a JEPG image as an input and gives a binary output for simple and complex. It consists of stacked convolutional blocks for spatial feature extraction, followed by flattening and two fully connected layers that map the learned features to a binary scene-selection output. It was trained with ~24000 frames and achieved 98.98% validation accuracy. The average execution time of this CNN model on Jetson Orin is 1.267ms. 
![Project Banner](./assets/img/cnn_map.png)
Figure 3 CNN Saliency map visualizing the pixel regions that contribute most to the CNN’s binary classification decision for scene selection.
Table 1 CNN and Feature Correlations 
![Project Banner](./assets/img/cnn_correlation.png)
Figure 3 shows which region does CNN look the most to make a decision. As shown in Table 1, scene complexity is strongly tied to object density, espcially the number of vehicles. Complexity scene score increases when there are more objects (traffic participants) in the frame while brightness has a mere contribution. 

### **4.2 Hybrid System Performance**
Experiment Assumption: 
Camera Frames are coming in 10 FPS (Waymo dataset rate) which means each frame needs to be processed within 100ms. For the experiment, max perception latency allowed is set to be 90ms. 
It is compared with the latest cloud frame YOLO execution time + current network latency 
Network availability is expressed in binary for simplipication. If the network is available, the cloud is available at its full capacity. Jetson Yolo 11n result is used if the cloud YOLO result doesn’t arrive back in 90ms. Total of 7967 frames are used for the experiment.  

Experiment Results:
68% of the entire frame was processed locally with YOLO 11n and 32% was processed in the cloud with YOLO 11x. Table 2 compares detection accuracy across the three perception policies using mean absolute error (MAE) for vehicle and pedestrian counts. The hybrid approach improves accuracy over local-only inference while using cloud resources for less than one-third of frames. It retains approximately 75% of the accuracy benefit of cloud-only perception at a lower cloud utilization rate. 

Table 2  Local vs Cloud vs Hybrid Perception
![Project Banner](./assets/img/hybridperception.png)

Table 3 shows the latency distribution for local inference, cloud inference, and network RTT. Cloud inference is computationally faster than local inference; but, network latency is a huge facotr. The hybrid policy ensures cloud inference is only used when total latency remains within the allowed budget.

Table 3 Latency of Hybrid Perception
![Project Banner](./assets/img/latency.png)

---

# **5. Discussion & Conclusions**

The hybrid perception framework successfully aligned computational effort with scene complexity. The lightweight CNN-based gating model proved effective at identifying high-density, complex scenes while introducing negligible overhead, enabling real-time operation. 

Limitations: 

Future Direction: 
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
* Python 3.8, NumPy 1.24.2, OpenCV 4.6.0, Pillow 9.2.0, ONNX Runtime 1.19.2, YOLO 8.3.230
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
