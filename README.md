# Leveraging Deep Learning for Early Detection, Diagnosis, and Sub-Classification of Retinopathy

This repository contains the source code for my research project titled **"Leveraging Deep Learning for Early Detection, Diagnosis, and Sub-Classification of Retinopathy"**.  
The project proposes a lightweight two-stage classification framework to enhance the detection and diagnosis of diabetic retinopathy using fundus images.

---

## 🔍 Project Summary

This project addresses the classification of diabetic retinopathy (DR) based on fundus images from the APTOS 2019 Kaggle dataset.

### 🧠 Key Contributions:
- **Stage 1:** A lightweight CNN model (based on EfficientNet) is used for **five-level DR severity classification**.
- **Stage 2:** Outputs from the first model are combined with **unsupervised sub-category labels** and fed into a **Extreme Gradient Boosting (XGBoost)** for fine-grained **sub-classification**.
- Achieved **78.06% accuracy** overall while maintaining low model complexity.

### 📁 Folder Structure
```plaintext
- Data Pre-processing
    ├── DataSpliting_1.py
    ├── DataAugmentation.py
    └── DataPreparation_1.py

- First Stage Classification
    ├── EfficientNet.py
    ├── Train.py
    └── Test.py

- Sub-Classes Annotated Labels
    ├── FeatureExtraction_Labeling.py
    └── SubClass_CSV.py

- Second Stage Classification
    ├── SubClass_Model.py
    └── Final_SubClass.py
```
