"""
Agronomic Physiological Analyzer.

Integrates Computer Vision pest counts with SPAD-502 Chlorophyll Index readings.
Computes Pearson correlation, linear regression parameters, and fitosanitary
economic injury severity classification (NDEE / EIL).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from src.domain.entities import LeafSample, Detection


@dataclass(frozen=True)
class InfestationSeverity:
    """Fitosanitary diagnosis and agronomic recommendation."""
    level: str  # "SÃO / CONTROL", "LEVE", "MODERADO", "CRÍTICO"
    pest_count: int
    spad_reading: float
    chlorophyll_loss_percent: float
    action_required: bool
    recommended_control: str


@dataclass(frozen=True)
class AgronomicAnalysisReport:
    """Aggregated statistical study across plants and treatments."""
    sample_count: int
    pearson_r: float
    p_value: float
    r_squared: float
    slope: float
    intercept: float
    severity_breakdown: Dict[str, int]
    mean_spad_healthy: float
    mean_spad_infested: float


class AgronomicPestAnalyzer:
    """
    Bio-mathematical analysis engine correlating Planococcus citri infestation
    with grapevine physiological stress (SPAD-502 chlorophyll degradation).
    """

    HEALTHY_SPAD_BASELINE = 48.0  # Average healthy Vitis vinifera SPAD reading

    @classmethod
    def evaluate_sample_severity(
        cls,
        detected_pests: int,
        spad_reading: float
    ) -> InfestationSeverity:
        """
        Determines the fitosanitary severity level according to integrated pest
        management (MIP - Manejo Integrado de Pragas) standards for viticulture.
        """
        chlorophyll_loss = max(0.0, (cls.HEALTHY_SPAD_BASELINE - spad_reading) / cls.HEALTHY_SPAD_BASELINE * 100.0)

        if detected_pests == 0:
            level = "SÃO / CONTROLE"
            action = False
            recommendation = "Monitoramento quinzenal de rotina. Nenhuma intervenção química necessária."
        elif detected_pests <= 10:
            level = "INFESTAÇÃO LEVE"
            action = False
            recommendation = "Monitoramento intensificado. Liberação preventiva de inimigos naturais (parasitoides como Anagyrus pseudococci)."
        elif detected_pests <= 30:
            level = "INFESTAÇÃO MODERADA"
            action = True
            recommendation = "Atingiu Nível de Ação (NA). Aplicação de óleo mineral ou sabão potássico nas axilas foliares."
        else:
            level = "INFESTAÇÃO CRÍTICA (SURTO)"
            action = True
            recommendation = "Nível de Dano Econômico (NDE) superado. Risco de fumagina severa. Intervenção fitossanitária curativa direcionada."

        return InfestationSeverity(
            level=level,
            pest_count=detected_pests,
            spad_reading=spad_reading,
            chlorophyll_loss_percent=round(chlorophyll_loss, 2),
            action_required=action,
            recommended_control=recommendation
        )

    @classmethod
    def analyze_dataset_correlation(
        cls,
        samples: List[LeafSample],
        predictions: Optional[List[List[Detection]]] = None
    ) -> AgronomicAnalysisReport:
        """
        Performs linear regression and Pearson correlation between pest counts and SPAD readings.
        """
        if len(samples) < 2:
            raise ValueError("At least 2 samples are required for correlation analysis.")

        if predictions is not None:
            if len(predictions) != len(samples):
                raise ValueError(
                    f"Sample count mismatch: got {len(samples)} samples but {len(predictions)} prediction lists."
                )
            pest_counts = np.array([len(preds) for preds in predictions], dtype=np.float64)
        else:
            pest_counts = np.array([s.pest_count for s in samples], dtype=np.float64)

        spads = np.array([s.spad_reading for s in samples], dtype=np.float64)

        # Pearson correlation and p-value calculation
        std_x = float(np.std(pest_counts))
        std_y = float(np.std(spads))

        if std_x > 0.0 and std_y > 0.0:
            from scipy import stats
            res = stats.pearsonr(pest_counts, spads)
            pearson_r = float(res.statistic)
            p_val = float(res.pvalue)
        else:
            pearson_r = 0.0
            p_val = 1.0

        # Linear regression: SPAD = slope * pest_count + intercept
        if std_x > 0.0:
            slope, intercept = np.polyfit(pest_counts, spads, deg=1)
            # R-squared
            fitted = slope * pest_counts + intercept
            ss_res = np.sum((spads - fitted) ** 2)
            ss_tot = np.sum((spads - np.mean(spads)) ** 2)
            r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
        else:
            slope = 0.0
            intercept = float(np.mean(spads))
            r2 = 0.0

        # Breakdown
        severity_counts = {"SÃO / CONTROLE": 0, "INFESTAÇÃO LEVE": 0, "INFESTAÇÃO MODERADA": 0, "INFESTAÇÃO CRÍTICA (SURTO)": 0}
        healthy_spads = []
        infested_spads = []

        for p_count, spad in zip(pest_counts, spads):
            diag = cls.evaluate_sample_severity(int(p_count), spad)
            severity_counts[diag.level] = severity_counts.get(diag.level, 0) + 1
            if p_count == 0:
                healthy_spads.append(spad)
            else:
                infested_spads.append(spad)

        return AgronomicAnalysisReport(
            sample_count=len(samples),
            pearson_r=round(pearson_r, 4),
            p_value=round(p_val, 4),
            r_squared=round(r2, 4),
            slope=round(float(slope), 4),
            intercept=round(float(intercept), 2),
            severity_breakdown=severity_counts,
            mean_spad_healthy=round(float(np.mean(healthy_spads)), 2) if healthy_spads else 0.0,
            mean_spad_infested=round(float(np.mean(infested_spads)), 2) if infested_spads else 0.0
        )
