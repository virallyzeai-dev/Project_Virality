"""Causal inference models for understanding what drives virality."""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Causal inference libraries
try:
    import dowhy
    from dowhy import CausalModel

    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False
    print(
        "Warning: DoWhy not available. Some causal analysis features will be limited."
    )

try:
    from causalml.inference.meta import SLearner, TLearner, XLearner
    from causalml.inference.tree import UpliftTreeClassifier

    CAUSALML_AVAILABLE = True
except ImportError:
    CAUSALML_AVAILABLE = False
    print(
        "Warning: CausalML not available. Some causal analysis features will be limited."
    )

from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from ..core.models import CausalAnalysisResult, CausalEffect, ContentData
from .predictive import FeatureProcessor


class CausalAnalyzer:
    """Analyze causal relationships between content features and virality."""

    def __init__(self, virality_threshold: float = 2.0):
        """Initialize the causal analyzer."""
        self.virality_threshold = virality_threshold
        self.feature_processor = FeatureProcessor()
        self.causal_model = None
        self.data_df = None

    def prepare_causal_data(self, content_data_list: List[ContentData]) -> pd.DataFrame:
        """Prepare data for causal analysis."""
        data_rows = []

        for content_data in content_data_list:
            # Extract features
            feature_vector = self.feature_processor.extract_features(content_data)

            # Combine all features
            row = {}
            row.update(feature_vector.text_features)
            row.update(feature_vector.visual_features)
            row.update(feature_vector.engagement_features)
            row.update(feature_vector.platform_features)

            # Add outcome variable
            row["virality_coefficient"] = (
                self.feature_processor.engagement_extractor.metrics_calculator.calculate_virality_coefficient(
                    content_data.metrics.shares,
                    content_data.metrics.unique_users_reached
                    or content_data.metrics.impressions,
                )
            )
            row["is_viral"] = (
                1 if row["virality_coefficient"] >= self.virality_threshold else 0
            )

            # Add content ID
            row["content_id"] = content_data.content_id

            data_rows.append(row)

        return pd.DataFrame(data_rows)

    def identify_confounders(
        self, data: pd.DataFrame, treatment: str, outcome: str
    ) -> List[str]:
        """Identify potential confounding variables."""
        # Simple correlation-based confounder identification
        confounders = []

        for col in data.columns:
            if col not in [treatment, outcome, "content_id"]:
                # Check if variable is correlated with both treatment and outcome
                treatment_corr = abs(data[treatment].corr(data[col]))
                outcome_corr = abs(data[outcome].corr(data[col]))

                # If correlated with both (threshold of 0.1), consider as confounder
                if treatment_corr > 0.1 and outcome_corr > 0.1:
                    confounders.append(col)

        return confounders

    def estimate_treatment_effect_simple(
        self, data: pd.DataFrame, treatment: str, outcome: str, confounders: List[str]
    ) -> Dict[str, float]:
        """Estimate treatment effect using simple regression adjustment."""
        # Create treatment groups
        treated = data[data[treatment] > data[treatment].median()]
        control = data[data[treatment] <= data[treatment].median()]

        # Simple difference in means
        simple_effect = treated[outcome].mean() - control[outcome].mean()

        # Regression adjustment
        if confounders:
            # Include confounders in regression
            X_cols = confounders + [treatment]
            X = data[X_cols]
            y = data[outcome]

            # Fit regression model
            model = LinearRegression()
            model.fit(X, y)

            # Treatment effect is the coefficient of the treatment variable
            treatment_idx = X_cols.index(treatment)
            adjusted_effect = model.coef_[treatment_idx]
        else:
            adjusted_effect = simple_effect

        return {
            "simple_effect": simple_effect,
            "adjusted_effect": adjusted_effect,
            "treatment_group_size": len(treated),
            "control_group_size": len(control),
        }

    def dowhy_causal_analysis(
        self, data: pd.DataFrame, treatment: str, outcome: str, confounders: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Perform causal analysis using DoWhy library."""
        if not DOWHY_AVAILABLE:
            return None

        try:
            # Create causal graph
            causal_graph = f"""
            digraph {{
                {treatment} -> {outcome};
                {" -> ".join([f"{c} -> {treatment}; {c} -> {outcome}" for c in confounders])};
            }}
            """

            # Create causal model
            model = CausalModel(
                data=data,
                treatment=treatment,
                outcome=outcome,
                graph=causal_graph,
                common_causes=confounders,
            )

            # Identify causal effect
            identified_estimand = model.identify_effect(
                proceed_when_unidentifiable=True
            )

            # Estimate causal effect
            estimate = model.estimate_effect(
                identified_estimand, method_name="backdoor.linear_regression"
            )

            # Refute the estimate
            refutation_results = []
            try:
                # Random common cause
                refute_random = model.refute_estimate(
                    identified_estimand, estimate, method_name="random_common_cause"
                )
                refutation_results.append(refute_random)

                # Placebo treatment
                refute_placebo = model.refute_estimate(
                    identified_estimand,
                    estimate,
                    method_name="placebo_treatment_refuter",
                )
                refutation_results.append(refute_placebo)
            except:
                pass  # Some refutations might fail

            return {
                "causal_effect": estimate.value,
                "confidence_interval": getattr(estimate, "confidence_intervals", None),
                "p_value": getattr(estimate, "p_value", None),
                "refutation_results": refutation_results,
            }

        except Exception as e:
            print(f"DoWhy analysis failed: {e}")
            return None

    def analyze_feature_causality(
        self,
        content_data_list: List[ContentData],
        key_features: Optional[List[str]] = None,
    ) -> CausalAnalysisResult:
        """Analyze causal effects of features on virality."""
        # Prepare data
        data = self.prepare_causal_data(content_data_list)
        self.data_df = data

        outcome = "virality_coefficient"
        causal_effects = []

        # Default key features if not provided
        if key_features is None:
            key_features = [
                "sentiment_polarity",
                "emotional_intensity",
                "readability_score",
                "hashtag_count",
                "posting_hour",
                "face_count",
                "brightness",
            ]

        # Filter features that exist in data
        available_features = [f for f in key_features if f in data.columns]

        for feature in available_features:
            if data[feature].nunique() > 1:  # Skip constant features
                # Identify confounders
                confounders = self.identify_confounders(data, feature, outcome)

                # Estimate treatment effect
                effect_results = self.estimate_treatment_effect_simple(
                    data, feature, outcome, confounders
                )

                # Statistical significance test
                treated = data[data[feature] > data[feature].median()][outcome]
                control = data[data[feature] <= data[feature].median()][outcome]

                t_stat, p_value = stats.ttest_ind(treated, control)

                # Calculate confidence interval (approximate)
                pooled_std = np.sqrt(
                    (
                        (len(treated) - 1) * treated.var()
                        + (len(control) - 1) * control.var()
                    )
                    / (len(treated) + len(control) - 2)
                )
                se = pooled_std * np.sqrt(1 / len(treated) + 1 / len(control))
                ci_lower = effect_results["simple_effect"] - 1.96 * se
                ci_upper = effect_results["simple_effect"] + 1.96 * se

                causal_effect = CausalEffect(
                    factor=feature,
                    effect_size=effect_results["adjusted_effect"],
                    p_value=p_value,
                    confidence_interval=(ci_lower, ci_upper),
                    is_significant=p_value < 0.05,
                )

                causal_effects.append(causal_effect)

                # DoWhy analysis if available
                if DOWHY_AVAILABLE and len(confounders) > 0:
                    dowhy_results = self.dowhy_causal_analysis(
                        data, feature, outcome, confounders
                    )
                    if dowhy_results:
                        # Update effect size with DoWhy estimate
                        causal_effect.effect_size = dowhy_results["causal_effect"]

        # Sort effects by significance and effect size
        causal_effects.sort(
            key=lambda x: (x.is_significant, abs(x.effect_size)), reverse=True
        )

        return CausalAnalysisResult(
            content_id="batch_analysis",
            causal_effects=causal_effects,
            confounders=self.identify_confounders(
                data, "sentiment_polarity", outcome
            ),  # Example
        )

    def analyze_treatment_heterogeneity(
        self,
        content_data_list: List[ContentData],
        treatment_feature: str,
        segment_features: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Analyze how treatment effects vary across different segments."""
        data = self.prepare_causal_data(content_data_list)
        outcome = "virality_coefficient"

        heterogeneity_results = {}

        for segment_feature in segment_features:
            if segment_feature in data.columns:
                # Create segments based on median split
                high_segment = data[
                    data[segment_feature] > data[segment_feature].median()
                ]
                low_segment = data[
                    data[segment_feature] <= data[segment_feature].median()
                ]

                # Estimate treatment effect in each segment
                if len(high_segment) > 10 and len(low_segment) > 10:
                    high_effect = self.estimate_treatment_effect_simple(
                        high_segment, treatment_feature, outcome, []
                    )
                    low_effect = self.estimate_treatment_effect_simple(
                        low_segment, treatment_feature, outcome, []
                    )

                    heterogeneity_results[segment_feature] = {
                        "high_segment_effect": high_effect["simple_effect"],
                        "low_segment_effect": low_effect["simple_effect"],
                        "difference": high_effect["simple_effect"]
                        - low_effect["simple_effect"],
                    }

        return heterogeneity_results

    def causal_mediation_analysis(
        self,
        content_data_list: List[ContentData],
        treatment: str,
        mediator: str,
        outcome: str = "virality_coefficient",
    ) -> Dict[str, float]:
        """Analyze mediation effects (how much of treatment effect goes through mediator)."""
        data = self.prepare_causal_data(content_data_list)

        if not all(col in data.columns for col in [treatment, mediator, outcome]):
            return {}

        # Simple mediation analysis using regression
        # Total effect (treatment -> outcome)
        total_model = LinearRegression()
        total_model.fit(data[[treatment]], data[outcome])
        total_effect = total_model.coef_[0]

        # Direct effect (treatment -> outcome, controlling for mediator)
        direct_model = LinearRegression()
        direct_model.fit(data[[treatment, mediator]], data[outcome])
        direct_effect = direct_model.coef_[0]  # Coefficient of treatment

        # Indirect effect (mediated effect)
        # treatment -> mediator
        mediator_model = LinearRegression()
        mediator_model.fit(data[[treatment]], data[mediator])
        a_path = mediator_model.coef_[0]

        # mediator -> outcome (from direct model)
        b_path = direct_model.coef_[1]  # Coefficient of mediator

        indirect_effect = a_path * b_path

        return {
            "total_effect": total_effect,
            "direct_effect": direct_effect,
            "indirect_effect": indirect_effect,
            "mediation_proportion": indirect_effect / total_effect
            if total_effect != 0
            else 0,
        }
