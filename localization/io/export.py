"""CSV export for Ekeland expansion results (drone/satellite uniform schema)."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


class FeatureCsvExporter:
    """Write expansion+Ekeland records as a uniform CSV schema."""

    def export_expansion(self, expansion_results: Iterable[dict], output_path: str = "ekeland_expansion.csv") -> pd.DataFrame:
        rows = []
        for result in expansion_results:
            filename = result.get("filename", "unknown")
            seed_idx = result["seed_triangle_idx"]
            vertices = result["seed_vertices"]
            coords = result["seed_coordinates"]
            ekeland = result["ekeland_angles"]
            ekeland_full = result.get("ekeland_full_results", {})

            ek_angles = [ekeland.get(v, 0.0) for v in vertices]
            axis_names = [ekeland_full.get(v, {}).get("axis_name", "Unknown") for v in vertices]

            rows.append(
                {
                    "filename": filename,
                    "seed_triangle_idx": seed_idx,
                    "vertex_0_idx": vertices[0],
                    "vertex_1_idx": vertices[1],
                    "vertex_2_idx": vertices[2],
                    "vertex_0_x": coords[0][0],
                    "vertex_0_y": coords[0][1],
                    "vertex_1_x": coords[1][0],
                    "vertex_1_y": coords[1][1],
                    "vertex_2_x": coords[2][0],
                    "vertex_2_y": coords[2][1],
                    "ekeland_v0": ek_angles[0],
                    "ekeland_v1": ek_angles[1],
                    "ekeland_v2": ek_angles[2],
                    "axis_v0": axis_names[0],
                    "axis_v1": axis_names[1],
                    "axis_v2": axis_names[2],
                    "mean_ekeland": float(np.mean(ek_angles)),
                    "expansion_steps": result["expansion_steps"],
                    "num_triangles_in_region": result["num_triangles_in_region"],
                    "num_boundary_vertices": result["num_boundary_vertices"],
                }
            )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        return df

    def export_drone_style(self, expansion_results: Iterable[dict], output_path: str) -> pd.DataFrame:
        df = self.export_expansion(expansion_results, output_path=output_path)
        return df


def export_expansion_ekeland_to_csv(expansion_results, output_path: str = "ekeland_expansion.csv") -> pd.DataFrame:
    return FeatureCsvExporter().export_expansion(expansion_results=expansion_results, output_path=output_path)


def save_drone_style_csv(expansion_results, output_path: str) -> pd.DataFrame:
    return FeatureCsvExporter().export_drone_style(expansion_results=expansion_results, output_path=output_path)
