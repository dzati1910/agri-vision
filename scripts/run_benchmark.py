"""
End-to-End Agronomic Benchmark: Standard Resize vs SAHI Tiling.

Empirically demonstrates the resolution collapse of millimeter-scale pests
under standard resizing, proving why SAHI is essential for high-resolution cameras.
"""

from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
import cv2

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.synthetic_generator import SyntheticVinePestGenerator
from src.slicing.sahi_slicer import SAHISlicer
from src.models.reference_detector import ReferenceMorphologyDetector
from src.pipeline.hyper_inference import HyperInferencePipeline, StandardResizePipeline
from src.pipeline.agronomic_analyzer import AgronomicPestAnalyzer
from src.domain.metrics import evaluate_detections
from src.utils.visualizer import PestVisualizer


def run_comprehensive_benchmark():
    print("=" * 75)
    print(" AGRIVISION: BENCHMARK EXPERIMENTAL DE VISÃO COMPUTACIONAL (IFSERTÃOPE)")
    print(" Detecção de Planococcus citri em Ultra-Resolução (Galaxy S25) + SPAD-502")
    print("=" * 75)

    output_dir = PROJECT_ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate Synthetic Evaluation Dataset (12 leaves across 4 biological plants)
    generator = SyntheticVinePestGenerator(seed=42)
    dataset_samples = []
    dataset_images = []

    print("\n[1/4] Gerando dataset sintético de alta resolução (3840x2160)...")
    plant_ids = ["PLANTA_01", "PLANTA_02", "PLANTA_03", "PLANTA_04"]
    treatments = ["T0", "T1", "T2", "T3"]

    for i in range(12):
        pid = plant_ids[i % len(plant_ids)]
        trt = treatments[i % len(treatments)]
        sid = f"amostra_{i+1:02d}_{pid}_{trt}"
        img, sample = generator.generate_sample(
            sample_id=sid,
            plant_id=pid,
            treatment=trt,
            width=3840,
            height=2160
        )
        dataset_images.append(img)
        dataset_samples.append(sample)

    total_gt = sum(len(s.ground_truth) for s in dataset_samples)
    print(f" -> Total de amostras: {len(dataset_samples)}")
    print(f" -> Total de pragas (Ground Truth): {total_gt}")

    # 2. Setup Detectors and Pipelines
    detector = ReferenceMorphologyDetector()
    slicer = SAHISlicer(slice_height=640, slice_width=640, overlap_height_ratio=0.20, overlap_width_ratio=0.20)

    sahi_pipeline = HyperInferencePipeline(detector=detector, slicer=slicer, global_nms_iou=0.45, use_box_fusion=True)
    resize_pipeline = StandardResizePipeline(detector=detector)

    # 3. Evaluate Standard Resize Pipeline
    print("\n[2/4] Executando Método Convencional (Redimensionamento Direto para 640x640)...")
    resize_preds = []
    t0 = time.perf_counter()
    for img in dataset_images:
        res = resize_pipeline.run(img, conf_threshold=0.25)
        resize_preds.append(res.detections)
    time_resize = time.perf_counter() - t0

    gts = [s.ground_truth for s in dataset_samples]
    resize_report = evaluate_detections(resize_preds, gts)

    # 4. Evaluate SAHI Tiling Pipeline
    print("\n[3/4] Executando Método Proposto (SAHI: Fatiamento em Tiles com Resolução Óptica 100%)...")
    sahi_preds = []
    t0 = time.perf_counter()
    for img in dataset_images:
        res = sahi_pipeline.run(img, conf_threshold=0.25, evaluate_full_image=True)
        sahi_preds.append(res.detections)
    time_sahi = time.perf_counter() - t0

    sahi_report = evaluate_detections(sahi_preds, gts)

    # 5. Save visual comparison of the most infested sample
    print("\n[4/4] Gerando artefatos visuais e gráficos fitossanitários...")
    worst_idx = np.argmax([len(s.ground_truth) for s in dataset_samples])
    comp_banner = PestVisualizer.create_side_by_side_comparison(
        dataset_images[worst_idx],
        resize_preds[worst_idx],
        sahi_preds[worst_idx],
        target_display_width=2560
    )
    comp_path = output_dir / "comparativo_sahi_vs_resize.jpg"
    cv2.imwrite(str(comp_path), comp_banner)
    print(f" -> Imagem comparativa salva em: {comp_path}")

    # Fitosanitary SPAD correlation plot
    agro_report = AgronomicPestAnalyzer.analyze_dataset_correlation(dataset_samples, sahi_preds)
    plot_path = output_dir / "correlacao_spad_pragas.png"
    PestVisualizer.plot_agronomic_correlation(dataset_samples, agro_report, plot_path)
    print(f" -> Gráfico de correlação SPAD salvo em: {plot_path}")

    # 6. Print Benchmark Summary Table
    print("\n" + "=" * 75)
    print(" RESULTADOS CONSOLIDADOS DO BENCHMARK")
    print("=" * 75)
    print(f"{'Métrica':<35} | {'Resize Direto':<16} | {'SAHI (Proposto)':<16}")
    print("-" * 75)
    print(f"{'Total Ground Truth':<35} | {total_gt:<16} | {total_gt:<16}")
    print(f"{'Pragas Detectadas (TP+FP)':<35} | {resize_report.total_predictions:<16} | {sahi_report.total_predictions:<16}")
    print(f"{'Precisão (Precision)':<35} | {resize_report.overall_precision * 100:<15.1f}% | {sahi_report.overall_precision * 100:<15.1f}%")
    print(f"{'Revocação (Recall)':<35} | {resize_report.overall_recall * 100:<15.1f}% | {sahi_report.overall_recall * 100:<15.1f}%")
    print(f"{'F1-Score':<35} | {resize_report.overall_f1 * 100:<15.1f}% | {sahi_report.overall_f1 * 100:<15.1f}%")
    print(f"{'mAP @ 0.50':<35} | {resize_report.mean_ap_50 * 100:<15.1f}% | {sahi_report.mean_ap_50 * 100:<15.1f}%")
    print(f"{'mAP @ COCO (0.50:0.95)':<35} | {resize_report.mean_ap_coco * 100:<15.1f}% | {sahi_report.mean_ap_coco * 100:<15.1f}%")
    print(f"{'Tempo Total de Inferência':<35} | {time_resize:<14.2f} s | {time_sahi:<14.2f} s")
    print("-" * 75)
    print(f"Correlação de Pearson com Clorofila SPAD-502: r = {agro_report.pearson_r:.3f} (R² = {agro_report.r_squared:.3f})")
    print("=" * 75)
    print(" Conclusão Técnica: O fatiamento SAHI evita o desaparecimento óptico das ninfas,")
    print(" multiplicando o Recall de forma decisiva para a fenotipagem agrícola.")
    print("=" * 75)


if __name__ == "__main__":
    run_comprehensive_benchmark()
