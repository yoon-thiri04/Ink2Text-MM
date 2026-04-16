# Ink2Text-MM: Burmese Handwriting OCR

**Ink2Text-MM** is a deep learning-based Optical Character Recognition (OCR) system specifically designed to bridge the gap between freehand Burmese scribbles and digital text. Designed for tablet and iPad users, it transforms fluid pen strokes on digital whiteboards into copyable, standard Burmese Unicode text.

---
## Demo
The system allows users to draw Burmese text on a sketch canvas and instantly converts it into digital Unicode text.

### Correct Recognition Examples
<img width="500" height="500" alt="Demo 1" src="https://github.com/user-attachments/assets/3e2b2d49-c5ff-47b4-b908-46bc7b7c3222" />

<img width="500" height="500" alt="Demo 2" src="https://github.com/user-attachments/assets/cdd0d5f2-c747-4ee1-91bf-a22ef689cbd7" />

---
###  Key Purposes
* **Digital Stationery:** Seamless note-taking on PDFs and digital sketches for iPad/Tablet users.
* **Educational Tool:** An interactive way to teach children Burmese calligraphy by providing instant digital feedback on their handwritten letters.
* **Accessibility:** Making digital content creation more intuitive for those who prefer handwriting over typing.
* **Digitization:** Rapidly converting handwritten archives or personal journals into searchable digital formats.

---

##  Model Architecture

The system utilizes a **CRNN (Convolutional Recurrent Neural Network) + CTC (Connectionist Temporal Classification)** pipeline, optimized for sequence-based character recognition.

### 1. Feature Extraction (CNN)
Use a **VGG-style CNN** architecture to process grayscale images (32×256). 

### 2. Sequence Modeling (BiLSTM)
A **Two-layer Bidirectional LSTM** processes the visual features extracted by the CNN.
* **Temporal Processing:** Handles the variable horizontal positioning of handwritten text.

### 3. Classification & Decoding (CTC)
* **Linear Layer:** Maps the BiLSTM hidden states to 68 distinct classes, including the Burmese alphabet, digits, and special markers.
* **Alignment-Free Decoding:** Uses **CTC Loss** and **Greedy Decoding** to predict the final text sequence without requiring pixel-perfect alignment between the image and the labels.

---

##  Technical Stack
* **Framework:** PyTorch
* **Loss Function:** CTCLoss
* **Optimization:** Trained for 50 epochs using AdamW optimizer for better weight decay.
* **Target:** 68 Classes (Consonants, Vowels, Medials, Digits, and Punctuation)

---
## Performance & Evaluation
The model was evaluated on a dedicated test set of handwritten Burmese scribbles. Despite the inherent variability in human handwriting, the CRNN pipeline achieved high precision in character recognition.

* **Character Error Rate (CER)**: 1.69%

* **Word Error Rate (WER)**: 12.14%

* **Exact-Match Accuracy**: 87.86%
---
## Confusion Analysis (Test Data Insights)

To better understand model behavior, confusion patterns were analyzed from the test dataset.
These highlight where the model struggles with visually similar Burmese characters.
### Most Frequent Confusion Pairs
* မ → ပ, ဖ
* ဗ → ပ, မ, ဖ, ဝ
* ဖ → ပ, မ
* ခ → မ
* ၇ → ရ
* ဓ → မ, စ
* စ → ဈ, ၈
* ဇ → စ
* ဂ → ဝ
* ဒ → အ
*  ု → ူ

So in the following test case,
- **Input (Handwritten):**  `ရန်ကုန်မြို့`

- **Model Output:** `ရန်ကုန်ပြို့`

- The model confuses visually similar characters such as: **မ, ပ, ဖ**
<img width="500" height="500" alt="5" src="https://github.com/user-attachments/assets/2fb8e685-d74c-474d-a955-1d0bf466935f" />

---
## 🗂️ Dataset

This project is trained using the **myOCR dataset**, a synthetic Myanmar OCR dataset containing over 25,000 text images across multiple font styles.

---

##  Citation

If you use this dataset, please cite the original work:

**Thura Aung, Ye Kyaw Thu and Myat Noe Oo**,
*"myOCR: Optical Character Recognition for Myanmar language with Post-OCR Error Correction"*,
Proceedings of the 19th International Joint Symposium on Artificial Intelligence and Natural Language Processing (iSAI-NLP 2024),
Nov 11–15, 2024, Pattaya, Thailand.

🔗 https://ieeexplore.ieee.org/document/10799448

---
##  Future Work

* **Stroke-Level Feedback:** Provide fine-grained feedback on handwriting quality to support learning applications for children.

* **Post-OCR Correction with Language Models:** Improve recognition accuracy by integrating:
  * N-gram based statistical correction
  * Neural language models (e.g., Transformer-based models)
  * LLM-based contextual correction for Burmese text

* **Multi-line & Document OCR:** Extend from single-line recognition to full-page document understanding.
---

## 👥 Contributors
* **Yoon Thiri Aung** – *Lead Developer & Model Architect*

---
*Developed with a passion for preserving the beauty of the Burmese script in the digital age.*

