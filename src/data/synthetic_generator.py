"""
Synthetic High-Resolution Agronomic Data Generator.

Generates realistic procedural grapevine leaf (Vitis vinifera) canvases with:
1. Procedural primary and secondary leaf venation.
2. Chlorophyll spatial gradients and chlorotic patches.
3. Planococcus citri (cochonilha-farinhenta) colonies with waxy filament textures,
   adhering biologically to leaf veins and petiole axils.
4. SPAD-502 chlorophyll sensor readings inversely correlated with insect count (Beer-Lambert optical density).
"""

from __future__ import annotations
import math
import random
from typing import List, Tuple, Optional
import numpy as np
import cv2

from src.domain.entities import BoundingBox, Detection, LeafSample, PestClass


class SyntheticVinePestGenerator:
    """
    Procedural generator for high-resolution vine leaf images with microscopic pests.
    Designed for controlled benchmarking of SAHI vs Standard Resize inference.
    """

    def __init__(self, seed: Optional[int] = 42) -> None:
        # Instance-local RNGs — does NOT mutate global random/numpy state.
        # This makes parallel dataset generation thread-safe and reproducible.
        self._rng = np.random.default_rng(seed)
        self._py_rng = random.Random(seed)

    def generate_sample(
        self,
        sample_id: str,
        plant_id: str,
        treatment: str = "T2",
        width: int = 3840,
        height: int = 2160,
        pest_count: Optional[int] = None
    ) -> Tuple[np.ndarray, LeafSample]:
        """
        Creates a high-resolution leaf image with ground-truth pest annotations.
        
        Args:
            sample_id: Unique image identifier.
            plant_id: Biological plant identifier (e.g., 'PLANT_07').
            treatment: Experimental treatment level ('T0', 'T1', 'T2', 'T3').
            width: Image width in pixels (e.g. 3840 for 4K UHD or 8160 for 50MP).
            height: Image height in pixels.
            pest_count: Specific pest count or randomly determined by treatment.
            
        Returns:
            Tuple of (image_bgr_numpy, leaf_sample_entity).
        """
        # Determine pest distribution by treatment level
        if pest_count is None:
            if treatment == "T0":  # Control: zero pests
                n_pests = 0
            elif treatment == "T1":  # Light infestation
                n_pests = self._py_rng.randint(3, 8)
            elif treatment == "T2":  # Moderate infestation
                n_pests = self._py_rng.randint(12, 25)
            elif treatment == "T3":  # Severe infestation
                n_pests = self._py_rng.randint(35, 60)
            else:
                n_pests = self._py_rng.randint(5, 20)
        else:
            n_pests = pest_count

        # Physiological SPAD-502 reading simulation:
        # Healthy grapevine SPAD ~ 46.0 - 52.0. Severe infestation drops down to ~ 22.0 - 28.0
        base_spad = 48.0 - (0.45 * n_pests) + self._py_rng.uniform(-1.5, 1.5)
        spad_value = max(18.0, min(55.0, round(base_spad, 1)))

        # 1. Base leaf canvas with chlorophyll gradient
        # OpenCV uses BGR ordering: [B, G, R]
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Base leaf green (Vitis vinifera healthy parenchyma)
        # Deep forest/olive green with subtle noise
        base_b = self._py_rng.randint(25, 40)
        base_g = self._py_rng.randint(110, 140)
        base_r = self._py_rng.randint(35, 55)

        canvas[:, :] = [base_b, base_g, base_r]

        # Add Gaussian texture noise to simulate cellular cuticle
        noise = self._rng.normal(0, 7, (height, width, 3)).astype(np.int16)
        canvas = np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # 2. Draw procedural leaf veins (primary + secondary)
        # Planococcus citri biologically aggregates along veins to feed on phloem sap
        center_x = width // 2
        petiole_y = int(height * 0.95)
        apex_y = int(height * 0.1)

        vein_color = (base_b + 25, base_g + 35, base_r + 30)  # Lighter pale green

        # Primary central vein (from petiole to apex)
        cv2.line(canvas, (center_x, petiole_y), (center_x, apex_y), vein_color, thickness=max(4, width // 400))

        # Secondary lateral veins radiating outward
        num_lateral_pairs = 7
        vein_endpoints: List[Tuple[int, int]] = []

        for i in range(1, num_lateral_pairs + 1):
            fraction = i / (num_lateral_pairs + 1)
            vy = int(petiole_y - fraction * (petiole_y - apex_y))
            # Angle upwards
            offset_y = int(height * 0.15)
            left_x = int(width * 0.15)
            right_x = int(width * 0.85)

            cv2.line(canvas, (center_x, vy), (left_x, vy - offset_y), vein_color, thickness=max(2, width // 600))
            cv2.line(canvas, (center_x, vy), (right_x, vy - offset_y), vein_color, thickness=max(2, width // 600))

            vein_endpoints.append((center_x, vy))
            vein_endpoints.append((left_x, vy - offset_y))
            vein_endpoints.append((right_x, vy - offset_y))

        # 3. Add chlorosis (yellowing) patches if pest count is high (sooty mold / sap drainage)
        if n_pests > 10:
            for _ in range(min(15, n_pests // 3)):
                cx = self._py_rng.randint(int(width * 0.2), int(width * 0.8))
                cy = self._py_rng.randint(int(height * 0.2), int(height * 0.8))
                radius = self._py_rng.randint(width // 40, width // 15)
                # Chlorotic yellow overlay [B, G, R]
                cv2.circle(canvas, (cx, cy), radius, (30, 180, 200), -1)
            # Blur chlorosis into the leaf tissue
            canvas = cv2.GaussianBlur(canvas, (51, 51), 0)

        # 4. Synthesize Planococcus citri pests
        detections: List[Detection] = []

        for p_idx in range(n_pests):
            # 80% chance to be near a vein, 20% random on lamina
            if self._py_rng.random() < 0.80 and vein_endpoints:
                anchor = self._py_rng.choice(vein_endpoints)
                px = int(np.clip(
                    anchor[0] + self._rng.normal(0, width * 0.05),
                    width * 0.1, width * 0.9
                ))
                py = int(np.clip(
                    anchor[1] + self._rng.normal(0, height * 0.05),
                    height * 0.1, height * 0.9
                ))
            else:
                px = self._py_rng.randint(int(width * 0.15), int(width * 0.85))
                py = self._py_rng.randint(int(height * 0.15), int(height * 0.85))

            # Determine pest development stage
            stage_roll = self._py_rng.random()
            if stage_roll < 0.60:
                stage = PestClass.NYMPH
                cid = 0
                cname = "planococcus_citri_nymph"
                # Nymph size: 1.5 - 2.5 mm -> approx 18 - 32 pixels in high-res
                w_px = self._py_rng.randint(18, 30)
                h_px = int(w_px * self._py_rng.uniform(1.2, 1.5))
            elif stage_roll < 0.85:
                stage = PestClass.ADULT
                cid = 1
                cname = "planococcus_citri_adult"
                # Adult female size: 3.0 - 4.5 mm -> approx 34 - 55 pixels
                w_px = self._py_rng.randint(32, 50)
                h_px = int(w_px * self._py_rng.uniform(1.3, 1.6))
            else:
                stage = PestClass.OVISAC
                cid = 2
                cname = "planococcus_citri_ovisac"
                # Ovisac cottony mass
                w_px = self._py_rng.randint(45, 70)
                h_px = int(w_px * self._py_rng.uniform(1.1, 1.4))

            # Draw pest morphology on canvas
            # Body: oval with chalky white/cream waxy coating
            angle = self._py_rng.uniform(0, 180)
            center = (px, py)
            axes = (w_px // 2, h_px // 2)

            # Waxy white body: [230, 240, 245]
            body_color = (
                self._py_rng.randint(220, 245),
                self._py_rng.randint(235, 250),
                self._py_rng.randint(240, 255)
            )
            cv2.ellipse(canvas, center, axes, angle, 0, 360, body_color, -1)

            # Draw perimeter filaments (characteristic lateral wax tassels)
            for _ in range(6):
                filament_len = self._py_rng.randint(3, 7)
                fx = int(px + (w_px // 2 + filament_len) * math.cos(math.radians(self._py_rng.uniform(0, 360))))
                fy = int(py + (h_px // 2 + filament_len) * math.sin(math.radians(self._py_rng.uniform(0, 360))))
                cv2.line(canvas, (px, py), (fx, fy), (240, 250, 255), 1)

            # Compute exact tight bounding box of rotated ellipse with margin for wax filaments
            rad = math.radians(angle)
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)
            a = w_px / 2.0
            b = h_px / 2.0
            half_w = math.sqrt((a * cos_a) ** 2 + (b * sin_a) ** 2) + 2.0
            half_h = math.sqrt((a * sin_a) ** 2 + (b * cos_a) ** 2) + 2.0

            xmin = max(0.0, px - half_w)
            ymin = max(0.0, py - half_h)
            xmax = min(float(width), px + half_w)
            ymax = min(float(height), py + half_h)

            bbox = BoundingBox(xmin=round(xmin, 1), ymin=round(ymin, 1), xmax=round(xmax, 1), ymax=round(ymax, 1))
            detections.append(
                Detection(
                    bbox=bbox,
                    class_id=cid,
                    class_name=cname,
                    confidence=1.0
                )
            )

        sample = LeafSample(
            sample_id=sample_id,
            plant_id=plant_id,
            treatment=treatment,
            spad_reading=spad_value,
            image_shape=(height, width, 3),
            ground_truth=detections,
            metadata={
                "n_pests": n_pests,
                "sensor": "Galaxy_S25_SPAD502",
                "biological_target": "Planococcus citri (Risso, 1813)"
            }
        )

        return canvas, sample

