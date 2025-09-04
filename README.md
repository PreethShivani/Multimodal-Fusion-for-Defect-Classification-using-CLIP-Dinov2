# Multimodal-Fusion-for-Defect-Classification-using-CLIP-Dinov2 + Logistic Regression

##  Overview
This project tackles **industrial defect detection** using the **cable subset** of the MVTec-AD dataset.  
It leverages **CLIP (ViT-B/32)** for semantic text–image alignment and **DINOv2 (ViT-S/14)** for high-quality visual features.  
Both embeddings are fused and fed into a **Logistic Regression classifier** to distinguish 9 cable defect classes.

---

##  Classes
- `good`  
- `bent_wire`  
- `cable_swap`  
- `combined`  
- `cut_inner_insulation`  
- `cut_outer_insulation`  
- `missing_cable`  
- `missing_wire`  
- `poke_insulation`

---

##  Method
1. **Embedding extraction**: CLIP + DINOv2 (frozen pretrained models).  
2. **Feature fusion**: concatenate image embeddings into a single representation.  
3. **Classification**: StandardScaler + multinomial Logistic Regression.  
4. **Evaluation**: Stratified 3-fold cross-validation, classification report, and confusion matrix.

---

## 📊 Results (3-fold CV)

| Class                 | Precision | Recall | F1-score |
|------------------------|-----------|--------|----------|
| bent_wire              | 0.80      | 0.80   | 0.80     |
| cable_swap             | 0.78      | 0.70   | 0.74     |
| combined               | 0.00      | 0.00   | 0.00     |
| cut_inner_insulation   | 0.73      | 0.80   | 0.76     |
| cut_outer_insulation   | 0.64      | 0.90   | 0.75     |
| good                   | 0.00      | 0.00   | 0.00     |
| missing_cable          | 0.89      | 0.80   | 0.84     |
| missing_wire           | 0.90      | 0.90   | 0.90     |
| poke_insulation        | 0.73      | 0.80   | 0.76     |

**Accuracy**: 76%  
**Macro F1**: 0.62  
**Weighted F1**: 0.74  

---

## 🔑 Key Points
- **CLIP prompts** refined with descriptive sentences improved class separation.  
- **DINOv2 embeddings** added fine-grained visual detail.  
- **Feature fusion + Logistic Regression** produced strong results on most classes.  
- Rare classes (`good`, `combined`) remain weak due to very limited data.  
