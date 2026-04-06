"""Type detection - Result dataclasses."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ...type_conversion import DataType
from ..interfaces import DataFormat


@dataclass
class FormatDetectionResult:
    """Result of data format detection analysis."""

    detected_format: DataFormat
    confidence_score: float
    alternative_formats: List[Tuple[DataFormat, float]] = field(default_factory=list)
    detection_details: Dict[str, Any] = field(default_factory=dict)
    schema_info: Optional["SchemaInfo"] = None
    warnings: List[str] = field(default_factory=list)
    detection_time: float = 0.0
    sample_size: int = 0


@dataclass
class SchemaInfo:
    """Comprehensive schema information for detected data formats."""

    data_format: DataFormat
    structure_type: str  # 'tabular', 'array', 'nested', 'scalar'

    # Tabular data schema
    columns: Optional[Dict[str, str]] = None  # column_name -> data_type
    column_types: Optional[Dict[str, DataType]] = None

    # Array/tensor schema
    shape: Optional[Tuple[int, ...]] = None
    element_type: Optional[str] = None

    # General properties
    size_info: Dict[str, Any] = field(default_factory=dict)
    null_info: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)

    # Metadata
    creation_time: datetime = field(default_factory=datetime.now)
    inference_confidence: float = 1.0
    additional_properties: Dict[str, Any] = field(default_factory=dict)
