"""
Dataset Management & Kaggle Rice / Plant Leaf Image Pipeline
Author: Arjuna Fransesco
"""

import os
import pandas as pd
import numpy as np

DISEASE_CLASSES = [
    "Bacterial_Blight",
    "Brown_Spot",
    "Leaf_Blast",
    "Healthy_Leaf",
    "Tungro_Virus"
]

DISEASE_INFO = {
    "Bacterial_Blight": {
        "pathogen": "Xanthomonas oryzae",
        "symptoms": "Water-soaked lesions on leaf blades turning yellow-white with wavy margins.",
        "treatment": "Apply copper oxychloride (2.5 g/L) and avoid excess nitrogen fertilizer."
    },
    "Brown_Spot": {
        "pathogen": "Bipolaris oryzae (Cochliobolus miyabeanus)",
        "symptoms": "Circular to oval brown lesions with yellow halos across the leaf surface.",
        "treatment": "Apply Mancozeb or Tricyclazole spray; improve soil potassium and silicon nutrition."
    },
    "Leaf_Blast": {
        "pathogen": "Magnaporthe oryzae",
        "symptoms": "Spindle-shaped elliptical spots with greyish-white centers and dark brown borders.",
        "treatment": "Deploy systemic fungicides (Isoprothiolane / Tricyclazole 75% WP); adjust planting spacing."
    },
    "Healthy_Leaf": {
        "pathogen": "None (Healthy Tissue)",
        "symptoms": "Uniform vibrant green leaf pigmentation without necrosis or chlorosis.",
        "treatment": "Maintain balanced NPK fertilization and consistent drip/irrigation schedule."
    },
    "Tungro_Virus": {
        "pathogen": "Rice Tungro Bacilliform & Spherical Virus (vectored by Green Leafhopper)",
        "symptoms": "Yellow-orange discoloration starting from leaf tips, stunted plant growth.",
        "treatment": "Vector control using Imidacloprid to suppress Nephotettix virescens hoppers."
    }
}


def generate_kaggle_metadata(n_samples=2500, random_state=42):
    """
    Generates structured metadata reflecting the Kaggle Plant/Rice Leaf Disease Dataset.
    """
    np.random.seed(random_state)
    records = []
    
    weights = [0.22, 0.23, 0.20, 0.20, 0.15]
    
    for i in range(1, n_samples + 1):
        chosen_class = np.random.choice(DISEASE_CLASSES, p=weights)
        image_id = f"LEAF_{chosen_class[:4].upper()}_{10000 + i}.jpg"
        
        # Color & texture feature extraction simulation (HSV / GLCM)
        if chosen_class == "Healthy_Leaf":
            greenness = np.random.normal(0.78, 0.05)
            chlorosis = np.random.normal(0.04, 0.02)
            lesion_area_pct = 0.0
            contrast = np.random.normal(12.5, 2.0)
        elif chosen_class == "Bacterial_Blight":
            greenness = np.random.normal(0.48, 0.08)
            chlorosis = np.random.normal(0.38, 0.06)
            lesion_area_pct = np.clip(np.random.normal(0.32, 0.08), 0.1, 0.75)
            contrast = np.random.normal(38.0, 5.0)
        elif chosen_class == "Brown_Spot":
            greenness = np.random.normal(0.52, 0.07)
            chlorosis = np.random.normal(0.25, 0.05)
            lesion_area_pct = np.clip(np.random.normal(0.22, 0.06), 0.05, 0.55)
            contrast = np.random.normal(45.0, 6.0)
        elif chosen_class == "Leaf_Blast":
            greenness = np.random.normal(0.45, 0.08)
            chlorosis = np.random.normal(0.30, 0.07)
            lesion_area_pct = np.clip(np.random.normal(0.35, 0.09), 0.1, 0.8)
            contrast = np.random.normal(52.0, 7.0)
        else: # Tungro_Virus
            greenness = np.random.normal(0.32, 0.06)
            chlorosis = np.random.normal(0.58, 0.08)
            lesion_area_pct = np.clip(np.random.normal(0.45, 0.10), 0.15, 0.85)
            contrast = np.random.normal(28.0, 4.0)

        records.append({
            "image_id": image_id,
            "label": chosen_class,
            "label_idx": DISEASE_CLASSES.index(chosen_class),
            "mean_greenness_index": round(float(greenness), 4),
            "chlorosis_index": round(float(chlorosis), 4),
            "lesion_area_ratio": round(float(lesion_area_pct), 4),
            "glcm_contrast": round(float(contrast), 2),
            "split": np.random.choice(["train", "val", "test"], p=[0.7, 0.15, 0.15])
        })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    df = generate_kaggle_metadata(n_samples=2500)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/dataset_metadata.csv", index=False)
    print(f"[+] Successfully generated metadata: {df.shape}")
    print(df["label"].value_counts())
