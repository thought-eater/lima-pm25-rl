"""Ubicación de productos y lectura/escritura de los archivos de la tesis."""

import csv
import json
import os
import shutil
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONTROL_CONFIG = PROJECT_ROOT / "config/controles-bilg.json"


def product(name):
    """Resolver una entrada explícita del inventario, sin elegir por antigüedad."""
    products = json.loads((PROJECT_ROOT / "config/productos.json").read_text())
    if name not in products:
        raise ValueError(
            f"El producto {name!r} todavía no está registrado en config/productos.json"
        )
    return PROJECT_ROOT / products[name]


def output_directory(relative):
    """Fecha UTC hasta segundos; los escritores rechazan destinos existentes."""
    return (
        PROJECT_ROOT
        / "resultados"
        / relative
        / datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
    )


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")


@contextmanager
def atomic_output(path, *, overwrite=False):
    """Publicar solo archivos completos; los nombres por episodio son inmutables."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as file:
        try:
            yield file
            file.flush()
            os.fsync(file.fileno())
        except BaseException:
            temporary.unlink()
            raise
    try:
        if overwrite:
            temporary.replace(path)
        else:
            # link falla si el destino existe, sin ventana de sobrescritura.
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def copy_atomic(source, destination, *, overwrite=False):
    with (
        Path(source).open("rb") as source_file,
        atomic_output(destination, overwrite=overwrite) as destination_file,
    ):
        shutil.copyfileobj(source_file, destination_file)


def write_json_atomic(path, value, *, overwrite=False):
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with atomic_output(path, overwrite=overwrite) as file:
        file.write(data.encode("utf-8"))


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("x", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_files(path, names, *, calibration=False, cases=False):
    """Leer únicamente los productos solicitados de una carpeta de ejecución."""
    path = Path(path)
    if not path.is_dir():
        raise ValueError(f"Se requiere una carpeta de resultados: {path}")
    result = {}
    for file in sorted(path.rglob("*")):
        relative = file.relative_to(path)
        if (
            file.is_file()
            and file.name in names
            and (not calibration or "calibration" in relative.parts)
            and (not cases or file.parent.name.startswith("case-"))
        ):
            result[str(relative)] = file.read_bytes()
    if not result:
        raise ValueError(f"Faltan los productos solicitados en {path}")
    return result


def unique_file(files, name):
    matches = [data for path, data in files.items() if Path(path).name == name]
    if len(matches) != 1:
        raise ValueError(f"Se requiere exactamente un {name}")
    return matches[0]
