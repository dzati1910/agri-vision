"""
Computer Vision Visualizer and Agronomic Plotter.

Generates side-by-side detection comparisons, tiling grid overlays,
and publication-ready matplotlib correlation charts.
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

from src.domain.entities import Detection, TileWindow, LeafSample
from src.pipeline.agronomic_analyzer import AgronomicAnalysisReport


CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (255, 140, 0),    # Nymph: Deep Sky Blue / Orange in BGR (0, 140, 255)
    1: (0, 0, 255),      # Adult: Bright Red in BGR
    2: (180, 0, 255),    # Ovisac: Magenta / Purple in BGR
}


class PestVisualizer:
    """Renders detection graphics, tiling grid slices, and comparative panels."""

    @staticmethod
    def draw_detections(
        image: np.ndarray,
        detections: List[Detection],
        line_thickness: Optional[int] = None,
        draw_labels: bool = True
    ) -> np.ndarray:
        """
        Draws bounding boxes and confidence tags onto an image copy.
        """
        canvas = image.copy()
        h, w = canvas.shape[:2]

        thickness = line_thickness or max(2, int(min(w, h) / 600))
        font_scale = max(0.4, thickness * 0.3)

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox.to_xyxy()]
            color = CLASS_COLORS.get(det.class_id, (0, 255, 0))

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

            if draw_labels:
                short_name = det.class_name.replace("planococcus_citri_", "")
                label = f"{short_name} {det.confidence:.2f}"
                (txt_w, txt_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

                # Background text label rectangle
                cv2.rectangle(canvas, (x1, max(0, y1 - txt_h - 4)), (x1 + txt_w + 4, y1), color, -1)
                cv2.putText(
                    canvas, label, (x1 + 2, max(txt_h + 2, y1 - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA
                )

        return canvas

    @staticmethod
    def draw_tiling_grid(
        image: np.ndarray,
        windows: List[TileWindow],
        grid_color: Tuple[int, int, int] = (0, 255, 255),  # Yellow
        alpha: float = 0.3
    ) -> np.ndarray:
        """
        Visualizes the overlapping tile windows across the high-resolution canvas.
        """
        overlay = image.copy()
        for win in windows:
            cv2.rectangle(overlay, (win.xmin, win.ymin), (win.xmax, win.ymax), grid_color, 2)

        return cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)

    @staticmethod
    def create_side_by_side_comparison(
        image: np.ndarray,
        standard_detections: List[Detection],
        sahi_detections: List[Detection],
        target_display_width: int = 1920
    ) -> np.ndarray:
        """
        Generates a comparative side-by-side banner:
        [Standard Resize: Vanishing Pests] vs [SAHI Tiling: Complete Detection]
        """
        left = PestVisualizer.draw_detections(image, standard_detections)
        right = PestVisualizer.draw_detections(image, sahi_detections)

        # Draw title banners on top of each panel
        banner_h = int(image.shape[0] * 0.06)
        font_scale = max(0.8, banner_h / 40.0)

        for panel, title, count, color in [
            (left, "YOLO RESIZE DIRETO (PERDA DE RESOLUÇÃO)", len(standard_detections), (0, 0, 180)),
            (right, "YOLO COM SAHI TILING (RESOLUÇÃO ÓPTICA 100%)", len(sahi_detections), (0, 150, 0))
        ]:
            cv2.rectangle(panel, (0, 0), (panel.shape[1], banner_h), color, -1)
            text = f"{title} | Pragas Detectadas: {count}"
            cv2.putText(panel, text, (30, int(banner_h * 0.68)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

        combined = np.hstack([left, right])

        # Downscale for smooth display on web and screens
        cur_h, cur_w = combined.shape[:2]
        if cur_w > target_display_width:
            factor = target_display_width / cur_w
            new_h = int(cur_h * factor)
            combined = cv2.resize(combined, (target_display_width, new_h), interpolation=cv2.INTER_AREA)

        return combined

    @staticmethod
    def plot_agronomic_correlation(
        samples: List[LeafSample],
        report: AgronomicAnalysisReport,
        output_plot_path: str | Path
    ) -> None:
        """
        Plots scientific correlation chart (SPAD Chlorophyll vs Planococcus citri count).
        """
        p = Path(output_plot_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        pests = [s.pest_count for s in samples]
        spads = [s.spad_reading for s in samples]
        treatments = [s.treatment for s in samples]

        plt.figure(figsize=(9, 6), dpi=150)
        treatment_palette = {"T0": "forestgreen", "T1": "gold", "T2": "darkorange", "T3": "crimson"}

        for t in ["T0", "T1", "T2", "T3"]:
            xs = [pests[i] for i in range(len(samples)) if treatments[i] == t]
            ys = [spads[i] for i in range(len(samples)) if treatments[i] == t]
            if xs:
                plt.scatter(xs, ys, color=treatment_palette.get(t, "blue"), label=f"Tratamento {t}", s=70, alpha=0.85, edgecolors="black", linewidths=0.5)

        # Draw regression line
        x_vals = np.linspace(min(pests), max(pests), 100)
        y_vals = report.slope * x_vals + report.intercept
        plt.plot(x_vals, y_vals, color="black", linestyle="--", linewidth=2.0, label=f"Regressão: y = {report.slope:.2f}x + {report.intercept:.1f} (R²={report.r_squared:.2f})")

        plt.title("Correlação Fisiológica: Infestação de Planococcus citri vs Índice SPAD-502", fontsize=12, fontweight="bold")
        plt.xlabel("Contagem de Cochonilhas por Folha (Visão Computacional)", fontsize=11)
        plt.ylabel("Teor de Clorofila Relativo (Índice SPAD-502)", fontsize=11)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(frameon=True, facecolor="white", edgecolor="none")
        plt.tight_layout()

        plt.savefig(p)
        plt.close()
