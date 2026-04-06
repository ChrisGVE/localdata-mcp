"""
Spatial indexing for performance optimization.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ...logging_manager import get_logger
from ._data import SpatialPoint

logger = get_logger(__name__)


class SpatialIndexer:
    """
    Spatial indexing for performance optimization.

    Provides spatial indexing capabilities using R-tree or grid-based methods
    to accelerate spatial queries on large datasets.
    """

    def __init__(self, method: str = "grid"):
        """
        Initialize spatial indexer.

        Parameters
        ----------
        method : str, default 'grid'
            Indexing method ('grid', 'quadtree', 'rtree').
        """
        self.method = method
        self.index = None
        self.geometries = {}
        self.bounds = None

        if method == "rtree":
            try:
                from rtree import index as rtree_index

                self.rtree_index = rtree_index
                self.rtree_available = True
            except ImportError:
                logger.warning("R-tree not available, falling back to grid indexing")
                self.method = "grid"
                self.rtree_available = False
        else:
            self.rtree_available = False

    def build_index(self, geometries: Dict[int, Any]):
        """
        Build spatial index from geometries.

        Parameters
        ----------
        geometries : dict
            Dictionary mapping IDs to geometry objects.
        """
        self.geometries = geometries.copy()

        if self.method == "rtree" and self.rtree_available:
            self._build_rtree_index()
        elif self.method == "grid":
            self._build_grid_index()
        else:
            raise ValueError(f"Unsupported indexing method: {self.method}")

    def _build_rtree_index(self):
        """Build R-tree spatial index."""
        self.index = self.rtree_index.Index()

        for geom_id, geometry in self.geometries.items():
            if hasattr(geometry, "bounds"):
                bounds = geometry.bounds
                self.index.insert(geom_id, bounds)

    def _build_grid_index(self):
        """Build grid-based spatial index."""
        if not self.geometries:
            return

        all_bounds = []
        for geometry in self.geometries.values():
            if hasattr(geometry, "bounds"):
                all_bounds.append(geometry.bounds)
            elif isinstance(geometry, SpatialPoint):
                all_bounds.append((geometry.x, geometry.y, geometry.x, geometry.y))

        if not all_bounds:
            return

        minx = min(b[0] for b in all_bounds)
        miny = min(b[1] for b in all_bounds)
        maxx = max(b[2] for b in all_bounds)
        maxy = max(b[3] for b in all_bounds)

        self.bounds = (minx, miny, maxx, maxy)

        grid_size = int(np.sqrt(len(self.geometries))) + 1
        self.grid_size = max(grid_size, 10)
        self.cell_width = (maxx - minx) / self.grid_size
        self.cell_height = (maxy - miny) / self.grid_size

        self.index = {}
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.index[(i, j)] = []

        for geom_id, geometry in self.geometries.items():
            bounds = self._get_geometry_bounds(geometry)
            if bounds:
                cells = self._get_intersecting_cells(bounds)
                for cell in cells:
                    if cell in self.index:
                        self.index[cell].append(geom_id)

    def _get_geometry_bounds(self, geometry):
        """Get bounds for any geometry type."""
        if hasattr(geometry, "bounds"):
            return geometry.bounds
        elif isinstance(geometry, SpatialPoint):
            return (geometry.x, geometry.y, geometry.x, geometry.y)
        elif isinstance(geometry, dict) and "bounds" in geometry:
            return geometry["bounds"]
        return None

    def _get_intersecting_cells(self, bounds):
        """Get grid cells that intersect with bounds."""
        if not self.bounds:
            return []

        minx, miny, maxx, maxy = bounds
        base_minx, base_miny, base_maxx, base_maxy = self.bounds

        min_i = max(0, int((minx - base_minx) / self.cell_width))
        max_i = min(self.grid_size - 1, int((maxx - base_minx) / self.cell_width))
        min_j = max(0, int((miny - base_miny) / self.cell_height))
        max_j = min(self.grid_size - 1, int((maxy - base_miny) / self.cell_height))

        cells = []
        for i in range(min_i, max_i + 1):
            for j in range(min_j, max_j + 1):
                cells.append((i, j))

        return cells

    def query(self, bounds: Tuple[float, float, float, float]) -> List[int]:
        """
        Query spatial index for intersecting geometries.

        Parameters
        ----------
        bounds : tuple
            Query bounds (minx, miny, maxx, maxy).

        Returns
        -------
        geometry_ids : list
            IDs of potentially intersecting geometries.
        """
        if self.index is None:
            return list(self.geometries.keys())

        if self.method == "rtree" and self.rtree_available:
            return list(self.index.intersection(bounds))
        elif self.method == "grid":
            cells = self._get_intersecting_cells(bounds)
            candidates = set()
            for cell in cells:
                if cell in self.index:
                    candidates.update(self.index[cell])
            return list(candidates)
        else:
            return list(self.geometries.keys())
