"""Variación sintética explícita del campo inicial y de las emisiones viales."""

import json
from dataclasses import asdict, dataclass, replace

import numpy as np
from lima_pm25.domain import AmbientBaseline, RegionalBoundaryForcing
from lima_pm25.simulation import RegionalBoundarySchedule, RoadLineSource


@dataclass(frozen=True)
class FieldVariation:
    """Hipótesis de contraste, no inventario de emisiones ni reconstrucción histórica.

    Las plumas iniciales representan concentración ya presente, no fuentes de
    emisión adicionales. Su evolución posterior corresponde al transporte de core.
    """

    ambient_range_ug_m3: tuple[float, float]
    road_source_scale: float
    initial_plume_count: int
    initial_plume_amplitude_range_ug_m3: tuple[float, float]
    initial_plume_sigma_range_m: tuple[float, float]
    provenance: str

    def __post_init__(self):
        for name in (
            "ambient_range_ug_m3",
            "initial_plume_amplitude_range_ug_m3",
            "initial_plume_sigma_range_m",
        ):
            values = tuple(getattr(self, name))
            if (
                len(values) != 2
                or not np.isfinite(values).all()
                or not 0 <= values[0] <= values[1]
            ):
                raise ValueError(f"{name} debe ser un intervalo finito no negativo")
            object.__setattr__(self, name, values)
        if self.initial_plume_sigma_range_m[0] <= 0:
            raise ValueError("El ancho de las plumas debe ser positivo")
        if not np.isfinite(self.road_source_scale) or self.road_source_scale <= 0:
            raise ValueError("La escala de emisiones debe ser finita y positiva")
        if type(self.initial_plume_count) is not int or self.initial_plume_count < 0:
            raise ValueError("La cantidad de plumas debe ser un entero no negativo")
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError("La variación del campo requiere procedencia explícita")

    def apply(self, request, seed: int):
        parameters = json.loads(request.episode_parameters_json)
        partition = parameters.get("partition")
        if partition not in {"training", "evaluation"}:
            raise ValueError("La variación requiere un episodio generado con partición")
        if type(seed) is not int or seed < 0:
            raise ValueError("La semilla debe ser un entero no negativo")
        # Independiente de rutas, decisiones, ruido instrumental y generador de viento.
        rng = np.random.default_rng([seed, 19, int(partition == "evaluation")])
        grid = request.model.grid
        cells = np.argwhere(grid.active_mask)
        if self.initial_plume_count > len(cells):
            raise ValueError(
                "No hay suficientes celdas activas para las plumas iniciales"
            )
        ambient = float(rng.uniform(*self.ambient_range_ug_m3))
        field = np.zeros(grid.shape)
        field[grid.active_mask] = ambient
        coordinates = np.asarray([grid.cell_center(int(r), int(c)) for r, c in cells])
        plumes = []
        for index in rng.choice(len(cells), self.initial_plume_count, replace=False):
            x, y = coordinates[index]
            amplitude = float(rng.uniform(*self.initial_plume_amplitude_range_ug_m3))
            sigma = float(rng.uniform(*self.initial_plume_sigma_range_m))
            field[grid.active_mask] += amplitude * np.exp(
                -np.sum((coordinates - [x, y]) ** 2, axis=1) / (2 * sigma**2)
            )
            plumes.append(
                {
                    "x_m": float(x),
                    "y_m": float(y),
                    "amplitude_ug_m3": amplitude,
                    "sigma_m": sigma,
                }
            )
        sources = []
        for source in request.sources:
            if isinstance(source, RoadLineSource):
                source = replace(
                    source,
                    emission_rate_ug_m_s=source.emission_rate_ug_m_s
                    * self.road_source_scale,
                    provenance=f"{source.provenance}; escala de contraste sintético={self.road_source_scale:g}",
                )
            sources.append(source)
        baseline = AmbientBaseline("rl-synthetic-background", ambient, self.provenance)
        parameters["field_variation"] = {
            "configuration": asdict(self),
            "ambient_ug_m3": ambient,
            "initial_plumes": plumes,
            "seed_namespace": "scenario_seed/19/partition; independent_of_sensors_and_policy",
        }
        return replace(
            request,
            initial_state=request.model.initialize_from_array(
                field, ambient_baseline=baseline
            ),
            regional_boundary=RegionalBoundarySchedule.constant(
                RegionalBoundaryForcing(
                    provider_id="rl-synthetic-background",
                    concentration_ug_m3=ambient,
                    provenance=self.provenance,
                )
            ),
            sources=tuple(sources),
            episode_parameters_json=json.dumps(parameters, sort_keys=True),
        )
