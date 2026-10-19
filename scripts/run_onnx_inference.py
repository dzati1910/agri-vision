"""
Lightweight ONNX Edge Inference Demo.

Demonstrates deploying the vision model to ONNX format and running inference
with ONNX Runtime and zero-copy NumPy buffers without PyTorch runtime dependencies.
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.exporter import ModelExporter
from src.models.onnx_detector import ONNXDetector
from src.slicing.sahi_slicer import SAHISlicer
from src.pipeline.hyper_inference import HyperInferencePipeline
from src.data.synthetic_generator import SyntheticVinePestGenerator
from src.utils.memory_profiler import MemoryProfiler


def main():
    print("=" * 70)
    print(" DEMONSTRAÇÃO DE INFERÊNCIA LEVE EM ONNX RUNTIME (BORDA / EDGE)")
    print("=" * 70)

    onnx_path = PROJECT_ROOT / "output" / "models" / "pest_yolo_edge.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] Preparando e validando grafo computacional ONNX...")
    if not onnx_path.exists():
        pt_path = PROJECT_ROOT / "output" / "models" / "yolov8_pest_best.pt"
        if pt_path.exists():
            ModelExporter.export_yolo_to_onnx(pt_path, onnx_path)
            print(f" -> Modelo YOLOv8 exportado para ONNX em: {onnx_path}")
        else:
            ModelExporter.generate_minimal_onnx_fixture(onnx_path)
            print(f" -> Fixture ONNX validada e salva em: {onnx_path}")
    else:
        print(f" -> Modelo ONNX validado e carregado de: {onnx_path}")

    print("\n[2/3] Inicializando Sessão de Inferência ONNX Runtime (CPU SIMD)...")
    detector = ONNXDetector(onnx_path)
    print(f" -> Motor carregado: {detector.model_name}")

    generator = SyntheticVinePestGenerator(seed=777)
    test_img, _ = generator.generate_sample("onnx_test", "PLANTA_09", "T2", width=1920, height=1080)

    # Low-level memory inspection
    buf = MemoryProfiler.inspect_buffer(test_img)
    print(f" -> Imagem em RAM: {buf.total_bytes / (1024*1024):.2f} MB | Ponteiro C: 0x{buf.pointer_address:x} | C_CONTIGUOUS: {buf.is_c_contiguous}")

    print("\n[3/3] Executando pipeline SAHI via ONNX Runtime...")
    slicer = SAHISlicer(slice_height=640, slice_width=640)
    pipeline = HyperInferencePipeline(detector=detector, slicer=slicer)

    result = pipeline.run(test_img, conf_threshold=0.10)
    print(f" -> Tiles processados: {result.num_tiles_processed}")
    print(f" -> Tempo de inferência: {result.elapsed_seconds:.3f} s (FPS: {result.fps:.1f})")
    print(f" -> Deteções encontradas: {len(result.detections)}")
    print("=" * 70)
    print(" Demonstração ONNX concluída com sucesso.")


if __name__ == "__main__":
    main()
