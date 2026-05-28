import numpy as np
import pandas as pd


class FeatureCsvExporter:
    """CSV exporter for Ekeland results (class-based facade)."""

    def export_expansion(self, expansion_results, output_path='ekeland_expansion.csv'):
        rows = []

        for result in expansion_results:
            filename = result.get('filename', 'unknown')
            seed_idx = result['seed_triangle_idx']
            vertices = result['seed_vertices']
            coords = result['seed_coordinates']
            ekeland = result['ekeland_angles']
            ekeland_full = result.get('ekeland_full_results', {})

            ek_angles = [ekeland.get(v, 0.0) for v in vertices]
            mean_ek = np.mean(ek_angles)
            axis_names = [ekeland_full.get(v, {}).get('axis_name', 'Unknown') for v in vertices]

            row = {
                'filename': filename,
                'seed_triangle_idx': seed_idx,
                'vertex_0_idx': vertices[0],
                'vertex_1_idx': vertices[1],
                'vertex_2_idx': vertices[2],
                'vertex_0_x': coords[0][0],
                'vertex_0_y': coords[0][1],
                'vertex_1_x': coords[1][0],
                'vertex_1_y': coords[1][1],
                'vertex_2_x': coords[2][0],
                'vertex_2_y': coords[2][1],
                'ekeland_v0': ek_angles[0],
                'ekeland_v1': ek_angles[1],
                'ekeland_v2': ek_angles[2],
                'axis_v0': axis_names[0],
                'axis_v1': axis_names[1],
                'axis_v2': axis_names[2],
                'mean_ekeland': mean_ek,
                'expansion_steps': result['expansion_steps'],
                'num_triangles_in_region': result['num_triangles_in_region'],
                'num_boundary_vertices': result['num_boundary_vertices'],
            }

            rows.append(row)

        if not rows:
            print('   Warning: No data to export.')
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        cols = ['filename', 'seed_triangle_idx', 'mean_ekeland'] + [
            c for c in df.columns if c not in ['filename', 'seed_triangle_idx', 'mean_ekeland']
        ]
        df = df[cols]

        df.to_csv(output_path, index=False)
        print(f'   Saved {len(df)} rows to: {output_path}')
        return df

    def export_drone_style(self, expansion_results, output_path):
        rows = []

        for result in expansion_results:
            seed_idx = result['seed_triangle_idx']
            vertices = result['seed_vertices']
            coords = result['seed_coordinates']
            ekeland_angles = result['ekeland_angles']
            ekeland_full = result.get('ekeland_full_results', {})

            angles = [ekeland_angles.get(v, 0.0) for v in vertices]
            axes = [ekeland_full.get(v, {}).get('axis_name', 'Unknown') for v in vertices]

            row = {
                'seed_triangle_idx': seed_idx,
                'vertex_0_idx': vertices[0],
                'vertex_1_idx': vertices[1],
                'vertex_2_idx': vertices[2],
                'vertex_0_x': coords[0][0],
                'vertex_0_y': coords[0][1],
                'vertex_1_x': coords[1][0],
                'vertex_1_y': coords[1][1],
                'vertex_2_x': coords[2][0],
                'vertex_2_y': coords[2][1],
                'ekeland_v0': angles[0],
                'ekeland_v1': angles[1],
                'ekeland_v2': angles[2],
                'axis_v0': axes[0],
                'axis_v1': axes[1],
                'axis_v2': axes[2],
                'mean_ekeland': np.mean(angles),
                'expansion_steps': result['expansion_steps'],
                'num_triangles_in_region': result['num_triangles_in_region'],
                'num_boundary_vertices': result['num_boundary_vertices'],
            }
            rows.append(row)

        if not rows:
            print(f'Warning: No data to save for {output_path}')
            return

        cols = [
            'seed_triangle_idx',
            'vertex_0_idx',
            'vertex_1_idx',
            'vertex_2_idx',
            'vertex_0_x',
            'vertex_0_y',
            'vertex_1_x',
            'vertex_1_y',
            'vertex_2_x',
            'vertex_2_y',
            'ekeland_v0',
            'ekeland_v1',
            'ekeland_v2',
            'axis_v0',
            'axis_v1',
            'axis_v2',
            'mean_ekeland',
            'expansion_steps',
            'num_triangles_in_region',
            'num_boundary_vertices',
        ]

        df = pd.DataFrame(rows)
        df = df[cols]
        df.to_csv(output_path, index=False)
        print(f'   Saved CSV: {output_path} ({len(df)} rows)')


def export_expansion_ekeland_to_csv(expansion_results, output_path='ekeland_expansion.csv'):
    """Backward-compatible functional API."""
    return FeatureCsvExporter().export_expansion(expansion_results=expansion_results, output_path=output_path)


def save_drone_style_csv(expansion_results, output_path):
    """Backward-compatible functional API."""
    return FeatureCsvExporter().export_drone_style(expansion_results=expansion_results, output_path=output_path)
