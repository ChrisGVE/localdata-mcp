"""
Spatial data processor base class for the geospatial analysis domain.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from ...logging_manager import get_logger
from ...pipeline.base import (
    AnalysisPipelineBase,
    CompositionMetadata,
    PipelineResult,
    PipelineState,
    StreamingConfig,
)
from ._base import _dependency_status
from ._data import SpatialDataFrame, SpatialDataValidator, SpatialPoint

logger = get_logger(__name__)


@dataclass
class SpatialAnalysisResult:
    """Container for spatial analysis results."""

    analysis_type: str
    results: Dict[str, Any]
    metadata: Dict[str, Any]
    spatial_data: Optional[SpatialDataFrame] = None
    computation_time: float = 0.0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "analysis_type": self.analysis_type,
            "results": self.results,
            "metadata": self.metadata,
            "computation_time": self.computation_time,
            "warnings": self.warnings,
            "has_spatial_data": self.spatial_data is not None,
        }


class SpatialDataProcessor(AnalysisPipelineBase):
    """
    Base processor for spatial data operations with sklearn compatibility.

    This class provides the foundation for all geospatial analysis components,
    handling dependency management, data validation, and result formatting.
    """

    def __init__(
        self,
        validate_input: bool = True,
        enable_fallbacks: bool = True,
        coordinate_columns: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize spatial data processor.

        Parameters
        ----------
        validate_input : bool, default True
            Whether to validate spatial input data.
        enable_fallbacks : bool, default True
            Whether to use fallback methods when libraries are missing.
        coordinate_columns : dict, optional
            Mapping of coordinate column names {'x': 'longitude', 'y': 'latitude'}.
        """
        super().__init__()
        self.validate_input = validate_input
        self.enable_fallbacks = enable_fallbacks
        self.coordinate_columns = coordinate_columns or {"x": "x", "y": "y"}

        self.dependency_checker_ = None
        self.spatial_validator_ = None
        self.fitted_data_info_ = None

    def _validate_spatial_data(
        self, data: Union[pd.DataFrame, SpatialDataFrame]
    ) -> SpatialDataFrame:
        """Convert and validate spatial data."""
        if isinstance(data, SpatialDataFrame):
            spatial_df = data
        else:
            spatial_df = SpatialDataFrame(data)

        if self.validate_input:
            points = spatial_df.get_points()
            validation_results = SpatialDataValidator.validate_coordinates(points)

            if validation_results["warnings"]:
                logger.warning(
                    f"Spatial data validation warnings: {validation_results['warnings']}"
                )

            if validation_results["valid_points"] == 0:
                raise ValueError("No valid spatial coordinates found in data")

        return spatial_df

    def _create_result(
        self,
        analysis_type: str,
        results: Dict[str, Any],
        spatial_data: Optional[SpatialDataFrame] = None,
        computation_time: float = 0.0,
        warnings: Optional[List[str]] = None,
    ) -> SpatialAnalysisResult:
        """Create standardized spatial analysis result."""
        metadata = {
            "timestamp": time.time(),
            "analysis_type": analysis_type,
            "dependency_status": _dependency_status.available_libraries.copy(),
            "fallback_active": _dependency_status.fallback_active,
            "data_shape": spatial_data.data.shape if spatial_data else None,
            "coordinate_columns": self.coordinate_columns,
        }

        return SpatialAnalysisResult(
            analysis_type=analysis_type,
            results=results,
            metadata=metadata,
            spatial_data=spatial_data,
            computation_time=computation_time,
            warnings=warnings or [],
        )
