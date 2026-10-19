"""
Dataset Generator & Anti-Leakage Validator.

Generates high-resolution images, produces YOLO and COCO annotations,
applies GroupKFold by Plant ID to guarantee biological independence,
and audits for pseudoreplication.
"""

from __future__ import annotations
import sys
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.synthetic_generator import SyntheticVinePestGenerator
from src.data.yolo_parser import YOLOParser
from src.data.coco_parser import COCOParser
from src.data.validation_split import PlantIDGroupSplitter


def main():
    print("=" * 70)
    print(" GERADOR DE DATASET FITOSSANITÁRIO COM DIVISÃO ANTI-LEAKAGE")
    print("=" * 70)

    dataset_dir = PROJECT_ROOT / "data" / "vine_pest_dataset"
    train_img_dir = dataset_dir / "images" / "train"
    train_lbl_dir = dataset_dir / "labels" / "train"
    val_img_dir = dataset_dir / "images" / "val"
    val_lbl_dir = dataset_dir / "labels" / "val"

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    generator = SyntheticVinePestGenerator(seed=2026)
    samples = []
    images = []

    # 8 distinct grapevine plants, 4 leaves per plant = 32 images
    plants = [f"PLANTA_{i:02d}" for i in range(1, 9)]
    treatments = ["T0", "T1", "T2", "T3"]

    print("\n[1/3] Sintetizando 32 imagens de alta resolução (2560x1440)...")
    for p_idx, pid in enumerate(plants):
        treatment = treatments[p_idx % len(treatments)]
        for leaf_num in range(1, 5):
            sid = f"{pid}_folha_{leaf_num:02d}_{treatment}"
            img, sample = generator.generate_sample(
                sample_id=sid,
                plant_id=pid,
                treatment=treatment,
                width=2560,
                height=1440
            )
            samples.append(sample)
            images.append(img)

    print(f" -> Geradas {len(samples)} amostras de {len(plants)} plantas distintas.")

    # 2. Split dataset strictly by Plant ID
    print("\n[2/3] Aplicando divisão estrita por Plant ID (GroupKFold)...")
    splitter = PlantIDGroupSplitter(n_splits=4)
    folds = splitter.split(samples, n_splits=4)
    train_samples, val_samples = folds[0]

    # Audit leakage
    audit = PlantIDGroupSplitter.audit_data_leakage(train_samples, val_samples)
    print(f" -> Amostras de Treino: {audit['train_sample_count']} (Plantas: {audit['train_plant_count']})")
    print(f" -> Amostras de Validação: {audit['val_sample_count']} (Plantas: {audit['val_plant_count']})")
    print(f" -> Plantas Vazadas (Data Leakage): {audit['leaked_plant_count']}")
    print(f" -> Integridade Científica: {'✅ APROVADA (ZERO VAZAMENTO)' if audit['is_scientifically_sound'] else '❌ FALHA'}")

    # 3. Export images and annotations
    print("\n[3/3] Exportando arquivos no formato YOLO (.txt) e COCO (.json)...")
    sample_to_img = {s.sample_id: img for s, img in zip(samples, images)}

    # Save Train partition
    for s in train_samples:
        img = sample_to_img[s.sample_id]
        img_path = train_img_dir / f"{s.sample_id}.jpg"
        lbl_path = train_lbl_dir / f"{s.sample_id}.txt"
        cv2.imwrite(str(img_path), img)
        s.image_path = str(img_path)
        h, w = img.shape[:2]
        YOLOParser.save_annotation(s.ground_truth, lbl_path, img_width=w, img_height=h)

    # Save Val partition
    for s in val_samples:
        img = sample_to_img[s.sample_id]
        img_path = val_img_dir / f"{s.sample_id}.jpg"
        lbl_path = val_lbl_dir / f"{s.sample_id}.txt"
        cv2.imwrite(str(img_path), img)
        s.image_path = str(img_path)
        h, w = img.shape[:2]
        YOLOParser.save_annotation(s.ground_truth, lbl_path, img_width=w, img_height=h)

    # Save YOLO data.yaml
    yaml_path = dataset_dir / "data.yaml"
    YOLOParser.create_dataset_yaml(yaml_path, dataset_root=dataset_dir)
    print(f" -> Arquivo de configuração YOLO salvo em: {yaml_path}")

    # Save COCO JSON
    COCOParser.export_samples_to_coco(train_samples, dataset_dir / "annotations_train_coco.json")
    COCOParser.export_samples_to_coco(val_samples, dataset_dir / "annotations_val_coco.json")
    print(f" -> Anotações COCO salvas em: {dataset_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
