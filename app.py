"""
Interactive Agricultural Computer Vision Dashboard.
IFSertãoPE - High-Resolution Planococcus citri Detection & SPAD-502 Sensor Fusion.

Backend: PyTorch + Ultralytics YOLOv8 / SAHI Tiling Engine.
Optimized for Responsive Desktop and Mobile (Samsung Galaxy S25) viewing.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
import cv2
import streamlit as st
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.domain.entities import BoundingBox, Detection
from src.slicing.sahi_slicer import SAHISlicer
from src.pipeline.hyper_inference import HyperInferencePipeline, StandardResizePipeline
from src.pipeline.agronomic_analyzer import AgronomicPestAnalyzer
from src.utils.visualizer import PestVisualizer
from src.data.synthetic_generator import SyntheticVinePestGenerator
from src.models.reference_detector import ReferenceMorphologyDetector
from src.models.yolo_detector import YOLODetector
from src.utils.memory_profiler import MemoryProfiler


# ---------------------------------------------------------
# Streamlit UI Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AgriVision | Detecção de Pragas em Alta Resolução",
    page_icon="🍇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, responsive mobile/desktop UI
st.markdown("""
<style>
    /* Responsive container adjustments */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 95%;
    }
    
    /* Metrics cards styling */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        color: #f8fafc;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .badge-healthy { background-color: #065f46; color: #34d399; }
    .badge-light { background-color: #854d0e; color: #fde047; }
    .badge-moderate { background-color: #9a3412; color: #fdba74; }
    .badge-critical { background-color: #991b1b; color: #fca5a5; }

    /* Mobile media query */
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }
        .metric-value {
            font-size: 1.4rem;
        }
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_detector_backend(use_pytorch_yolo: bool = True, model_name: Optional[str] = None):
    """
    Initializes detection backend.
    Supports either PyTorch YOLOv8 (trained on Planococcus citri) or Reference Morphological Detector.
    """
    if use_pytorch_yolo:
        target_model = model_name
        if target_model is None:
            custom_weights = PROJECT_ROOT / "output" / "models" / "yolov8_pest_best.pt"
            target_model = str(custom_weights) if custom_weights.exists() else "yolov8n.pt"
        try:
            return YOLODetector(target_model, device="cpu")
        except Exception as e:
            st.sidebar.warning(f"Fallback para detector de referência ({e})")
            return ReferenceMorphologyDetector()
    return ReferenceMorphologyDetector()


@st.cache_data
def get_preset_samples():
    """Generates benchmark samples for instant evaluation."""
    generator = SyntheticVinePestGenerator(seed=101)
    samples = {}
    for treatment, label in [
        ("T0", "Folha Sadia (T0 - Controle)"),
        ("T1", "Infestação Inicial (T1 - Poucas Ninfas)"),
        ("T2", "Infestação Moderada (T2 - Colônias)"),
        ("T3", "Surto Severo (T3 - Infestação Crítica)")
    ]:
        img, sample_meta = generator.generate_sample(
            sample_id=f"sample_{treatment}",
            plant_id=f"PLANTA_{treatment}",
            treatment=treatment,
            width=2560,
            height=1440
        )
        samples[treatment] = (img, sample_meta, label)
    return samples


def main():
    st.markdown("## 🍇 AgriVision: Detecção de Pragas Milimétricas em Ultra-Resolução")
    st.caption("Pesquisa Aplicada IFSertãoPE | Fatiamento SAHI vs Redimensionamento Convencional | Fusão Sensorial SPAD-502")

    # Sidebar Controls
    st.sidebar.header("⚙️ Painel de Controle")

    model_type = st.sidebar.radio(
        "Motor de Inferência (Backend):",
        ["PyTorch YOLOv8 (Ultralytics)", "Detector Morfológico Referência (Zero-Download)"],
        index=0
    )
    use_yolo = "PyTorch" in model_type
    detector = load_detector_backend(use_pytorch_yolo=use_yolo)

    st.sidebar.subheader("Parâmetros do Algoritmo SAHI")
    slice_size = st.sidebar.select_slider("Tamanho do Tile (Pixels):", options=[320, 512, 640], value=640)
    overlap_ratio = st.sidebar.slider("Sobreposição entre Tiles (%):", min_value=10, max_value=40, value=20, step=5) / 100.0
    conf_thresh = st.sidebar.slider("Limiar de Confiança (Confidence):", min_value=0.10, max_value=0.80, value=0.25, step=0.05)
    nms_iou = st.sidebar.slider("Limiar de Supressão (IoU NMS):", min_value=0.20, max_value=0.70, value=0.45, step=0.05)
    evaluate_macro = st.sidebar.checkbox("Incluir Passada Macro de Imagem Inteira", value=False)

    # Input Image Selection
    input_source = st.sidebar.radio("Fonte da Imagem:", ["Galeria de Amostras de Campo (IFSertãoPE)", "Fazer Upload (Galaxy S25 / Foto)"])

    presets = get_preset_samples()

    if input_source == "Fazer Upload (Galaxy S25 / Foto)":
        uploaded_file = st.sidebar.file_uploader("Selecione imagem JPG/PNG:", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            plant_id = "UPLOAD_USER"
            spad_val = st.sidebar.number_input("Leitura do Sensor SPAD-502 (Opcional):", min_value=10.0, max_value=60.0, value=42.0, step=0.5)
        else:
            st.info("Aguardando upload de imagem. Usando amostra padrão.")
            img_bgr, meta, _ = presets["T2"]
            plant_id = meta.plant_id
            spad_val = meta.spad_reading
    else:
        sample_key = st.sidebar.selectbox(
            "Selecione o Caso de Estudo:",
            list(presets.keys()),
            format_func=lambda k: presets[k][2]
        )
        img_bgr, meta, _ = presets[sample_key]
        plant_id = meta.plant_id
        spad_val = meta.spad_reading

    h, w = img_bgr.shape[:2]

    # Execution Button
    run_btn = st.sidebar.button("🚀 Processar Imagem com SAHI & YOLO", type="primary", use_container_width=True)

    # Pipeline Setup
    slicer = SAHISlicer(slice_height=slice_size, slice_width=slice_size, overlap_height_ratio=overlap_ratio, overlap_width_ratio=overlap_ratio)
    sahi_pipeline = HyperInferencePipeline(detector=detector, slicer=slicer, global_nms_iou=nms_iou, use_box_fusion=True)
    resize_pipeline = StandardResizePipeline(detector=detector)

    # Run inference
    with st.spinner("Executando inferência e fusão espacial..."):
        sahi_result = sahi_pipeline.run(img_bgr, conf_threshold=conf_thresh, evaluate_full_image=evaluate_macro)
        resize_result = resize_pipeline.run(img_bgr, conf_threshold=conf_thresh)

    # Fitosanitary Diagnosis
    diagnosis = AgronomicPestAnalyzer.evaluate_sample_severity(len(sahi_result.detections), spad_val)

    # ---------------------------------------------------------
    # Top KPI Cards
    # ---------------------------------------------------------
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Pragas Detectadas (SAHI)</div>
            <div class="metric-value" style="color: #38bdf8;">{len(sahi_result.detections)}</div>
            <small style="color: #94a3b8;">vs {len(resize_result.detections)} no método comum</small>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col2:
        gain = len(sahi_result.detections) - len(resize_result.detections)
        gain_str = f"+{gain} pragas" if gain >= 0 else f"{gain}"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Ganho de Resolução Óptica</div>
            <div class="metric-value" style="color: #4ade80;">{gain_str}</div>
            <small style="color: #94a3b8;">Ninfas salvas do borramento</small>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Índice Clorofila SPAD-502</div>
            <div class="metric-value" style="color: #facc15;">{spad_val:.1f}</div>
            <small style="color: #94a3b8;">Perda estimada: {diagnosis.chlorophyll_loss_percent}%</small>
        </div>
        """, unsafe_allow_html=True)

    with kpi_col4:
        badge_class = "badge-healthy"
        if "MODERADO" in diagnosis.level:
            badge_class = "badge-moderate"
        elif "CRÍTICO" in diagnosis.level:
            badge_class = "badge-critical"
        elif "LEVE" in diagnosis.level:
            badge_class = "badge-light"

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Diagnóstico Fitossanitário</div>
            <div style="margin-top: 8px;">
                <span class="status-badge {badge_class}">{diagnosis.level}</span>
            </div>
            <small style="color: #94a3b8; display:block; margin-top:6px;">Ação: {'Sim' if diagnosis.action_required else 'Não'}</small>
        </div>
        """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # Visual Comparison Section (Mobile & Desktop Friendly)
    # ---------------------------------------------------------
    st.markdown("### 🔍 Comparativo Visual: Detecção Direta vs SAHI Tiling")
    
    view_mode = st.radio(
        "Modo de Visualização:",
        ["Lado a Lado (Comparativo Direto)", "Apenas SAHI (Alta Resolução)", "Grid de Fatiamento (Tiles 640x640)"],
        horizontal=True
    )

    if view_mode == "Lado a Lado (Comparativo Direto)":
        comp_banner = PestVisualizer.create_side_by_side_comparison(
            img_bgr, resize_result.detections, sahi_result.detections, target_display_width=1600
        )
        st.image(cv2.cvtColor(comp_banner, cv2.COLOR_BGR2RGB), use_container_width=True)
    elif view_mode == "Apenas SAHI (Alta Resolução)":
        annotated = PestVisualizer.draw_detections(img_bgr, sahi_result.detections)
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True, caption=f"Total: {len(sahi_result.detections)} pragas detectadas com coordenadas globais.")
    else:
        windows = slicer.generate_windows(img_bgr.shape)
        grid_vis = PestVisualizer.draw_tiling_grid(img_bgr, windows)
        grid_annotated = PestVisualizer.draw_detections(grid_vis, sahi_result.detections)
        st.image(cv2.cvtColor(grid_annotated, cv2.COLOR_BGR2RGB), use_container_width=True, caption=f"Canvas fatiado em {len(windows)} janelas com sobreposição de {int(overlap_ratio*100)}%.")

    # ---------------------------------------------------------
    # Technical & Agronomic Breakdown
    # ---------------------------------------------------------
    tab_agro, tab_tech, tab_memory = st.tabs(["🌾 Recomendação Fitossanitária", "⚡ Estatísticas Técnicas", "🧠 Arquitetura de Memória (C/Python)"])

    with tab_agro:
        st.markdown(f"**Recomendação de Manejo Integrado (MIP):**")
        st.info(diagnosis.recommended_control)

        # Pest breakdown by stages
        nymphs = sum(1 for d in sahi_result.detections if d.class_id == 0)
        adults = sum(1 for d in sahi_result.detections if d.class_id == 1)
        ovisacs = sum(1 for d in sahi_result.detections if d.class_id == 2)

        c1, c2, c3 = st.columns(3)
        c1.metric("Ninfas (1-2 mm)", nymphs)
        c2.metric("Adultos (3-4 mm)", adults)
        c3.metric("Massas de Ovos / Colônias", ovisacs)

    with tab_tech:
        st.write({
            "Dimensões da Imagem Original": f"{w} x {h} pixels",
            "Janelas Processadas": sahi_result.num_tiles_processed,
            "Tempo de Processamento": f"{sahi_result.elapsed_seconds:.3f} s",
            "FPS Equivalente": f"{sahi_result.fps:.1f}",
            "Candidatos Pré-NMS": sahi_result.raw_tile_detection_count,
            "Detecções Consolidadas Pós-NMS": len(sahi_result.detections),
            "Filtro de Supressão": "Weighted Boxes Fusion (WBF) com média ponderada"
        })

    with tab_memory:
        mem_stats = MemoryProfiler.compute_memory_savings(
            image_shape=img_bgr.shape,
            num_tiles=sahi_result.num_tiles_processed,
            tile_shape=(slice_size, slice_size, 3)
        )
        st.markdown("""
        **Eficiência de Memória de Baixo Nível (Zero-Copy Slicing):**
        - O fatiamento em NumPy utiliza views contíguas (`char*` offset com strides), evitando alocações de heap repetitivas (`malloc`/`free`).
        """)
        st.write(mem_stats)


if __name__ == "__main__":
    main()
