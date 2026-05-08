import json
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from recognize import Recognizer, load_name_list

# Set matplotlib font to support Chinese
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

ROOT = Path(__file__).resolve().parents[1]
TEST_DATASET_DIR = ROOT / "testdataset"
OUTPUT_DIR = ROOT / "outputs"
POLITICAL_PERSONS_PATH = ROOT / "data/political_persons.txt"
POLITICAL_PERSONS = load_name_list(POLITICAL_PERSONS_PATH)

EXPECTED_CLASSES = {
    "暴恐": ["gun", "knife", "blood", "police", "riot_shield"],
    "违禁毒品": ["syringe", "powder"],
    "纳粹符号": ["swastika", "nazi_symbol"],
    "赌博": ["poker_card", "dice", "chip", "roulette", "slot_machine"],
    "不雅手势": ["middle_finger"],
    "游行集会": ["crowd", "flag", "banner", "signboard", "face"],
    "涉政": ["political_person"], # Special case for face recognition
    "正常": []
}

def evaluate():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    recognizer = Recognizer(enable_face=True)
    
    results = []
    
    # Iterate through each folder in testdataset
    for folder in TEST_DATASET_DIR.iterdir():
        if not folder.is_dir():
            continue
            
        true_business_label = folder.name
        print(f"Processing folder: {true_business_label}")
        
        images = [p for p in folder.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
        
        for img_path in tqdm(images):
            try:
                pred = recognizer.predict_image(img_path)
                
                pred_business_label = pred["business_category"]
                detected_classes = pred["categories"]
                
                # Special handling for face recognition in "涉政"
                if true_business_label == "涉政" and pred["face_recognition"]:
                    for identity in pred["face_recognition"].get("identities", []):
                        name = str(identity.get("name", "")).strip().casefold()
                        if name in POLITICAL_PERSONS:
                            detected_classes.append("political_person")
                
                results.append({
                    "image": img_path.name,
                    "true_business_label": true_business_label,
                    "pred_business_label": pred_business_label,
                    "detected_classes": detected_classes,
                    "is_business_correct": true_business_label == pred_business_label
                })
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "evaluation_results.csv", index=False, encoding="utf-8-sig")
    
    # 1. Business Label Accuracy
    business_acc = df.groupby("true_business_label")["is_business_correct"].mean().reset_index()
    business_acc.columns = ["业务标签", "准确率"]
    business_acc["准确率"] = business_acc["准确率"].apply(lambda x: f"{x:.2%}")
    
    print("\n--- 业务标签检测类测试准确率 ---")
    print(business_acc.to_markdown(index=False))
    business_acc.to_csv(OUTPUT_DIR / "business_label_accuracy.csv", index=False, encoding="utf-8-sig")
    
    # Plot Business Label Accuracy
    plt.figure(figsize=(10, 6))
    acc_values = df.groupby("true_business_label")["is_business_correct"].mean()
    sns.barplot(x=acc_values.index, y=acc_values.values, palette="viridis")
    plt.title("业务标签检测类测试准确率")
    plt.ylabel("准确率")
    plt.xlabel("业务标签")
    plt.ylim(0, 1.05)
    for i, v in enumerate(acc_values.values):
        plt.text(i, v + 0.01, f"{v:.2%}", ha='center')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "business_label_accuracy.png", dpi=300)
    plt.close()
    
    # 2. Original Detection Class Accuracy
    # Since the images are organized as testdataset/{business_label}/{original_class}/image.jpg,
    # we can calculate the image-level accuracy for each original class.
    
    original_results = []
    
    for folder in TEST_DATASET_DIR.iterdir():
        if not folder.is_dir():
            continue
            
        true_business_label = folder.name
        
        for subfolder in folder.iterdir():
            if not subfolder.is_dir():
                continue
                
            true_original_class = subfolder.name
            
            # Special case: "正常" folder doesn't have original class subfolders
            if true_business_label == "正常":
                continue
                
            images = [p for p in subfolder.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
            
            for img_path in images:
                # Find the prediction for this image
                pred_row = df[df["image"] == img_path.name]
                if pred_row.empty:
                    continue
                    
                detected_classes = pred_row.iloc[0]["detected_classes"]
                
                # Check if the true original class was detected
                # Handle special mapping for 涉政 -> 政治人物集
                if true_business_label == "涉政" and true_original_class == "政治人物集":
                    is_detected = "political_person" in detected_classes
                else:
                    is_detected = true_original_class in detected_classes
                
                original_results.append({
                    "业务场景": true_business_label,
                    "原始检测类": true_original_class,
                    "is_detected": is_detected
                })
                
    orig_df = pd.DataFrame(original_results)
    
    if not orig_df.empty:
        # Calculate accuracy per original class
        orig_acc = orig_df.groupby(["业务场景", "原始检测类"])["is_detected"].agg(['sum', 'count', 'mean']).reset_index()
        orig_acc.columns = ["业务场景", "原始检测类", "正确检测数", "样本总数", "准确率"]
        orig_acc["准确率_str"] = orig_acc["准确率"].apply(lambda x: f"{x:.2%}")
        
        print("\n--- 原始检测类测试准确率 (图像级别) ---")
        print(orig_acc[["业务场景", "原始检测类", "正确检测数", "样本总数", "准确率_str"]].to_markdown(index=False))
        orig_acc.to_csv(OUTPUT_DIR / "original_class_accuracy.csv", index=False, encoding="utf-8-sig")
        
        # Plot Original Class Accuracy
        plt.figure(figsize=(14, 8))
        sns.barplot(data=orig_acc, x="原始检测类", y="准确率", hue="业务场景", dodge=False, palette="Set2")
        plt.title("原始检测类测试准确率 (图像级别)")
        plt.ylabel("准确率")
        plt.xlabel("原始检测类")
        plt.xticks(rotation=45)
        plt.ylim(0, 1.05)
        for i, row in orig_acc.iterrows():
            plt.text(i, row["准确率"] + 0.01, f"{row['准确率']:.2%}", ha='center', fontsize=9)
        plt.legend(title="业务场景", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "original_class_accuracy.png", dpi=300)
        plt.close()
    
if __name__ == "__main__":
    evaluate()
