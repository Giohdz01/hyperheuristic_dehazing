# ============================================================
# HIPERHEURÍSTICA PARA IMAGE DEHAZING
# Base de datos: RESIDE-SOTS
# PARTE 1/2: configuración, dataset, dehazing y métricas
# ============================================================

# ============================================================
# 1. IMPORTACIÓN DE BIBLIOTECAS
# ============================================================
import csv
import json
import time
import math
import random
import zipfile
import matplotlib.pyplot as plt

from pathlib import Path

import cv2
import numpy as np

from scipy.ndimage import minimum_filter


from huggingface_hub import hf_hub_download


# ============================================================
# 2. CONFIGURACIÓN GENERAL
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "results_dehazing"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. DESCARGA Y EXTRACCIÓN DE RESIDE-SOTS
# ============================================================

DATASET_NAME = "desonglll/reside-SOTS"

DATASET_DIR = BASE_DIR / "Data" / "RESIDE-SOTS"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

ZIP_PATH = DATASET_DIR / "SOTS.zip"

print("=" * 60)
print("HIPERHEURÍSTICA PARA IMAGE DEHAZING")
print("=" * 60)

# ------------------------------------------------------------
# Descargar el ZIP si todavía no existe
# ------------------------------------------------------------
if not ZIP_PATH.exists():
    print("\nDescargando RESIDE-SOTS desde Hugging Face...")
    downloaded_path = hf_hub_download(
        repo_id=DATASET_NAME,
        filename="SOTS.zip",
        repo_type="dataset",
        local_dir=DATASET_DIR,
    )
    ZIP_PATH = Path(downloaded_path)
else:
    print("\nEl archivo SOTS.zip ya existe.")

# ------------------------------------------------------------
# Extraer si todavía no se ha extraído
# ------------------------------------------------------------
EXTRACT_MARKER = DATASET_DIR / ".extracted"

if not EXTRACT_MARKER.exists():
    print("\nExtrayendo las imágenes...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_file:
        zip_file.extractall(DATASET_DIR)
    EXTRACT_MARKER.touch()
    print("Extracción finalizada.")
else:
    print("\nEl conjunto de datos ya está extraído.")


# ============================================================
# 4. LOCALIZACIÓN DE SOTS Y DEL SUBCONJUNTO
# ============================================================

# Detectar la raíz real de SOTS dentro del directorio extraído
SOTS_DIR = DATASET_DIR / "SOTS"

if not SOTS_DIR.exists():
    for candidate in DATASET_DIR.rglob("*"):
        if candidate.is_dir() and (
            (candidate / "outdoor").exists()
            or (candidate / "indoor").exists()
        ):
            SOTS_DIR = candidate
            break

# Subconjunto: "outdoor" o "indoor"
SUBSET = "outdoor"

HAZY_DIR = SOTS_DIR / SUBSET / "hazy"
CLEAR_DIR = SOTS_DIR / SUBSET / "gt"

if not HAZY_DIR.exists() or not CLEAR_DIR.exists():
    raise FileNotFoundError(
        f"No se encontraron los directorios:\n"
        f"  hazy : {HAZY_DIR}\n"
        f"  clear: {CLEAR_DIR}\n"
        f"Revisa la estructura del dataset en {DATASET_DIR}"
    )

print(f"\nDirectorio hazy : {HAZY_DIR}")
print(f"Directorio clear: {CLEAR_DIR}")


# ============================================================
# 5. CONFIGURACIÓN DE LA HIPERHEURÍSTICA
# ============================================================

POPULATION_SIZE = 6
ITERATIONS = 30
SEED = 42

TRAIN_LIMIT = 8
TEST_LIMIT = 10


# ============================================================
# 6. PARÁMETROS DEL ALGORITMO DE DEHAZING
# ============================================================
#
# theta = [omega, t0, patch_size, guided_radius, guided_eps]
#
BOUNDS = np.array([
    [0.70,  0.99],   # omega: intensidad de eliminación de neblina
    [0.03,  0.30],   # t0: transmisión mínima
    [3.0,  31.0],    # tamaño de ventana del canal oscuro (impar)
    [3.0,  80.0],    # radio del filtro guiado
    [1e-5, 5e-2],    # regularización del filtro guiado
], dtype=float)


# ============================================================
# 7. REPRODUCIBILIDAD E INFO INICIAL
# ============================================================

np.random.seed(SEED)
random.seed(SEED)

print("\nConfiguración:")
print("  Base de datos:", DATASET_NAME)
print("  Tamaño de población:", POPULATION_SIZE)
print("  Iteraciones:", ITERATIONS)
print("  Límite de entrenamiento:", TRAIN_LIMIT)
print("  Límite de prueba:", TEST_LIMIT)
print("  Directorio de resultados:", OUTPUT_DIR)

print("\nInicialización completada.")


# ============================================================
# 8. EXTENSIONES Y EMPAREJAMIENTO HAZY ↔ CLEAR
# ============================================================

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
}


def _image_prefix(path: Path) -> str:
    """
    Devuelve el prefijo numérico del nombre del archivo.
    Ejemplo: '0001_0.8_0.2.jpg' -> '0001'.
    """
    return path.stem.split("_")[0]


def pair_dataset():
    """
    Empareja imágenes hazy con sus ground-truth usando el prefijo
    numérico. Si no hay coincidencias por prefijo, cae a orden
    alfabético.
    """
    hazy_files = sorted(
        p for p in HAZY_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    clear_files = sorted(
        p for p in CLEAR_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not hazy_files or not clear_files:
        raise RuntimeError(
            "No se encontraron imágenes en hazy/ o clear/."
        )

    clear_map = {_image_prefix(c): c for c in clear_files}

    pairs = []
    for h in hazy_files:
        key = _image_prefix(h)
        if key in clear_map:
            pairs.append((h, clear_map[key]))

    # Fallback: emparejamiento por orden
    if not pairs:
        pairs = list(zip(hazy_files, clear_files))

    return pairs


def load_pair(pair):
    """
    Carga un par (hazy, clear) como imágenes BGR uint8 del mismo
    tamaño y con 3 canales.
    """
    hazy_path, clear_path = pair

    hazy = cv2.imread(str(hazy_path))
    clear = cv2.imread(str(clear_path))

    if hazy is None:
        raise RuntimeError(f"No se pudo leer hazy: {hazy_path}")
    if clear is None:
        raise RuntimeError(f"No se pudo leer clear: {clear_path}")

    if hazy.ndim == 2:
        hazy = cv2.cvtColor(hazy, cv2.COLOR_GRAY2BGR)
    if clear.ndim == 2:
        clear = cv2.cvtColor(clear, cv2.COLOR_GRAY2BGR)

    if hazy.shape[:2] != clear.shape[:2]:
        clear = cv2.resize(
            clear,
            (hazy.shape[1], hazy.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    return hazy, clear


# ============================================================
# 9. REPARACIÓN Y DECODIFICACIÓN DEL VECTOR
# ============================================================

def repair(x):
    x = np.asarray(x, dtype=float).copy()

    x = np.clip(x, BOUNDS[:, 0], BOUNDS[:, 1])

    # Tamaño impar para el filtro mínimo
    x[2] = int(round(x[2]))
    if int(x[2]) % 2 == 0:
        x[2] += 1 if x[2] < 31 else -1

    # Radio entero para el filtro guiado
    x[3] = int(round(x[3]))

    return x


def decode(x):
    x = repair(x)
    return {
        "omega":         float(x[0]),
        "t0":            float(x[1]),
        "patch_size":    int(x[2]),
        "guided_radius": int(x[3]),
        "guided_eps":    float(x[4]),
    }


# ============================================================
# 10. FILTRO GUIADO (corregido: ksize = 2*radius + 1)
# ============================================================

def guided_filter(guide, source, radius, eps):
    r = int(radius)
    size = (2 * r + 1, 2 * r + 1)

    mean_i = cv2.boxFilter(guide, cv2.CV_32F, size)
    mean_p = cv2.boxFilter(source, cv2.CV_32F, size)

    mean_ip = cv2.boxFilter(guide * source, cv2.CV_32F, size)
    cov_ip = mean_ip - mean_i * mean_p

    mean_ii = cv2.boxFilter(guide * guide, cv2.CV_32F, size)
    var_i = np.maximum(mean_ii - mean_i * mean_i, 0.0)

    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i

    mean_a = cv2.boxFilter(a, cv2.CV_32F, size)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, size)

    return mean_a * guide + mean_b


# ============================================================
# 11. ESTIMACIÓN DE LUZ ATMOSFÉRICA
# ============================================================

def estimate_atmospheric_light(image, dark):
    count = max(1, int(0.001 * dark.size))
    indices = np.argpartition(dark.ravel(), -count)[-count:]
    candidates = image.reshape(-1, 3)[indices]
    best_index = int(np.argmax(np.sum(candidates, axis=1)))
    return candidates[best_index].astype(np.float32)


# ============================================================
# 12. ALGORITMO DE ELIMINACIÓN DE NEBLINA
# ============================================================

def dehaze(hazy_bgr, params):
    image = hazy_bgr.astype(np.float32) / 255.0

    dark = minimum_filter(
        np.min(image, axis=2),
        size=params["patch_size"],
        mode="nearest",
    )

    atmosphere = estimate_atmospheric_light(image, dark)
    atmosphere = np.maximum(atmosphere, 1e-3)

    normalized = image / atmosphere.reshape(1, 1, 3)

    dark_normalized = minimum_filter(
        np.min(normalized, axis=2),
        size=params["patch_size"],
        mode="nearest",
    )

    transmission = 1.0 - params["omega"] * dark_normalized
    transmission = np.clip(transmission, 0.0, 1.0)

    guide = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    transmission = guided_filter(
        guide,
        transmission.astype(np.float32),
        params["guided_radius"],
        params["guided_eps"],
    )

    transmission = np.clip(transmission, params["t0"], 1.0)

    restored = np.empty_like(image)
    for channel in range(3):
        restored[:, :, channel] = (
            (image[:, :, channel] - atmosphere[channel])
            / transmission + atmosphere[channel]
        )

    return np.clip(restored * 255.0, 0, 255).astype(np.uint8)


# ============================================================
# 13. MÉTRICAS PSNR Y SSIM
# ============================================================

def calculate_psnr(reference, output):
    ref = reference.astype(np.float32) / 255.0
    out = output.astype(np.float32) / 255.0

    mse = float(np.mean((ref - out) ** 2))

    if mse <= 1e-12:
        return 120.0

    return float(10.0 * math.log10(1.0 / mse))


def calculate_ssim(reference, output):
    x = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    y = cv2.cvtColor(output,    cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)

    sigma_x  = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x * mu_x
    sigma_y  = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y * mu_y
    sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    numerator   = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)

    return float(np.mean(numerator / (denominator + 1e-12)))


def evaluate_image(clear, hazy, params):
    restored = dehaze(hazy, params)

    psnr = calculate_psnr(clear, restored)
    ssim = calculate_ssim(clear, restored)

    quality = (
        0.5 * min(max(psnr, 0.0), 50.0) / 50.0
        + 0.5 * ssim
    )

    return quality, restored, psnr, ssim


def objective(vector, training_data):
    params = decode(vector)

    scores = []
    for hazy, clear in training_data:
        quality, _, _, _ = evaluate_image(clear, hazy, params)
        scores.append(quality)

    return float(np.mean(scores))


# ============================================================
# 14. HEURÍSTICAS DE BAJO NIVEL
# ============================================================

def h_random(current, best, population, rng):
    """Reinicio aleatorio uniforme dentro de los límites."""
    return rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1])


def h_local_mutation(current, best, population, rng):
    """
    Mutación gaussiana local.
    Las primeras 4 componentes se mueven en escala lineal y
    guided_eps en escala logarítmica (positivo por construcción).
    """
    candidate = current.copy()
    scale = 0.12 * (BOUNDS[:, 1] - BOUNDS[:, 0])

    candidate[:4] += rng.normal(0.0, scale[:4])
    candidate[4] = current[4] * np.exp(rng.normal(0.0, 0.7))

    return repair(candidate)


def h_differential_move(current, best, population, rng):
    """Movimiento diferencial tipo DE/rand/1/bin con reparación."""
    if len(population) < 3:
        return h_local_mutation(current, best, population, rng)

    indices = rng.choice(len(population), size=3, replace=False)
    a, b, c = population[indices]

    factor = rng.uniform(0.4, 0.9)
    donor = a + factor * (b - c)

    mask = rng.random(len(current)) < 0.7
    if not np.any(mask):
        mask[rng.integers(len(current))] = True

    candidate = current.copy()
    candidate[mask] = donor[mask]

    return repair(candidate)


def h_intensification(current, best, population, rng):
    """Explotación alrededor del mejor individuo con ruido pequeño."""
    candidate = best.copy()
    scale = 0.04 * (BOUNDS[:, 1] - BOUNDS[:, 0])

    candidate[:4] += rng.normal(0.0, scale[:4])
    candidate[4] = best[4] * np.exp(rng.normal(0.0, 0.25))

    return repair(candidate)


HEURISTICS = {
    "random":            h_random,
    "local_mutation":    h_local_mutation,
    "differential_move": h_differential_move,
    "intensification":   h_intensification,
}


# ============================================================
# 15. HIPERHEURÍSTICA DE SELECCIÓN ADAPTATIVA
# ============================================================

def run_hyperheuristic(training_data):
    rng = np.random.default_rng(SEED)

    names = list(HEURISTICS.keys())
    probabilities = np.ones(len(names)) / len(names)

    rewards = np.zeros(len(names))
    uses = np.zeros(len(names), dtype=int)

    # --- Población inicial ---
    population = np.array([
        repair(rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1]))
        for _ in range(POPULATION_SIZE)
    ])

    scores = np.array([
        objective(x, training_data) for x in population
    ])

    best_index = int(np.argmax(scores))
    best = population[best_index].copy()
    best_score = float(scores[best_index])

    history = []
    print(f"\nMejor calidad inicial: {best_score:.6f}")

    for iteration in range(ITERATIONS):

        index = int(rng.integers(len(population)))
        current = population[index].copy()
        current_score = float(scores[index])

        # La hiperheurística selecciona una heurística de bajo nivel
        operator_index = int(rng.choice(len(names), p=probabilities))
        operator_name = names[operator_index]

        candidate = HEURISTICS[operator_name](
            current, best, population, rng
        )
        candidate = repair(candidate)

        candidate_score = objective(candidate, training_data)

        # Recompensa: mejora local + mejora global
        reward = (
            max(0.0, candidate_score - current_score)
            + 2.0 * max(0.0, candidate_score - best_score)
        )
        rewards[operator_index] += reward
        uses[operator_index] += 1

        accepted = candidate_score >= current_score
        if accepted:
            population[index] = candidate
            scores[index] = candidate_score

        if candidate_score > best_score:
            best = candidate.copy()
            best_score = float(candidate_score)

        # Adaptación de probabilidades (softmax sobre recompensa media)
        mean_rewards = rewards / np.maximum(uses, 1)
        shifted = mean_rewards - np.max(mean_rewards)
        weights = np.exp(np.clip(shifted, -30.0, 0.0)) + 0.05
        probabilities = weights / np.sum(weights)

        history.append({
            "iteration":            iteration + 1,
            "operator":             operator_name,
            "candidate_score":      float(candidate_score),
            "best_score":           float(best_score),
            "accepted":             bool(accepted),
            "operator_probability": float(probabilities[operator_index]),
        })

        if (iteration + 1) % 5 == 0 or iteration == 0:
            print(
                f"Iteración {iteration + 1:02d}/{ITERATIONS} | "
                f"mejor={best_score:.6f} | "
                f"heurística={operator_name}"
            )

    statistics = {}
    for i, name in enumerate(names):
        statistics[name] = {
            "uses":              int(uses[i]),
            "mean_reward":       float(rewards[i] / max(uses[i], 1)),
            "final_probability": float(probabilities[i]),
        }

    return best, best_score, history, statistics


# ============================================================
# COMPARACIÓN VISUAL (sin matplotlib / sin PIL)
# ============================================================

def _build_comparison(hazy, dehazed, clear, title=None):
    """
    Construye una imagen BGR (uint8) con los tres paneles:
        [ Hazy | Dehazed | Ground Truth ]
    Añade una barra superior con las etiquetas y, si se indica,
    una barra inferior con el título (métricas).
    """
    h, w = hazy.shape[:2]

    # Igualar tamaños por seguridad
    if dehazed.shape[:2] != (h, w):
        dehazed = cv2.resize(dehazed, (w, h), interpolation=cv2.INTER_AREA)
    if clear.shape[:2] != (h, w):
        clear = cv2.resize(clear, (w, h), interpolation=cv2.INTER_AREA)

    # --- Barra superior con etiquetas ---
    banner_h = 40
    banner = np.zeros((banner_h, w * 3, 3), dtype=np.uint8)

    labels = ["Hazy (input)", "Dehazed (result)", "Ground Truth"]
    for i, label in enumerate(labels):
        cv2.putText(
            banner, label, (i * w + 12, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (255, 255, 255), 2, cv2.LINE_AA,
        )

    # --- Concatenar paneles ---
    grid = np.hstack([hazy, dehazed, clear])
    comparison = np.vstack([banner, grid])

    # --- Barra inferior con el título / métricas (opcional) ---
    if title:
        title_h = 36
        title_bar = np.zeros((title_h, comparison.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            title_bar, title, (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 255, 0), 2, cv2.LINE_AA,
        )
        comparison = np.vstack([comparison, title_bar])

    return comparison


def save_comparison(hazy, dehazed, clear, output_path, title=None):
    """Guarda la imagen comparativa en disco (BGR, sin matplotlib)."""
    comparison = _build_comparison(hazy, dehazed, clear, title=title)
    cv2.imwrite(str(output_path), comparison)


def show_comparison(hazy, dehazed, clear, title=None):
    """
    Muestra la comparación en una ventana de OpenCV.
    Presiona cualquier tecla para cerrar.
    """
    comparison = _build_comparison(hazy, dehazed, clear, title=title)

    # Reducir si es demasiado grande para la pantalla
    max_w = 1800
    if comparison.shape[1] > max_w:
        scale = max_w / comparison.shape[1]
        comparison = cv2.resize(
            comparison, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    win = "Hazy | Dehazed | Ground Truth"
    cv2.imshow(win, comparison)
    print("Presiona cualquier tecla en la ventana para continuar...")
    cv2.waitKey(0)
    cv2.destroyWindow(win)

# ============================================================
# 16. EXPERIMENTO COMPLETO
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("HIPERHEURÍSTICA PARA IMAGE DEHAZING")
    print("=" * 60)
    print("Directorio hazy :", HAZY_DIR)
    print("Directorio clear:", CLEAR_DIR)

    # --- Emparejar y dividir train/test ---
    pairs = pair_dataset()
    if len(pairs) < 2:
        raise RuntimeError(
            "Se necesitan al menos dos pares de imágenes."
        )

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(pairs))
    shuffled_pairs = [pairs[i] for i in order]

    test_count = min(
        max(1, round(0.25 * len(shuffled_pairs))),
        len(shuffled_pairs) - 1,
    )
    test_pairs  = shuffled_pairs[:test_count]
    train_pairs = shuffled_pairs[test_count:]

    if TRAIN_LIMIT is not None:
        train_pairs = train_pairs[:TRAIN_LIMIT]
    if TEST_LIMIT is not None:
        test_pairs = test_pairs[:TEST_LIMIT]

    training_data = [load_pair(p) for p in train_pairs]
    testing_data  = [load_pair(p) for p in test_pairs]

    print(f"\nImágenes de entrenamiento: {len(training_data)}")
    print(f"Imágenes de prueba       : {len(testing_data)}")
    print("Las imágenes de prueba no participan en la búsqueda.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Búsqueda hiperheurística ---
    start = time.time()
    best_vector, best_score, history, statistics = (
        run_hyperheuristic(training_data)
    )
    elapsed = time.time() - start

    best_params = decode(best_vector)

     # --- Evaluación final sobre el conjunto de prueba ---
    rows = []
    image_dir      = OUTPUT_DIR / "test_images"
    comparison_dir = OUTPUT_DIR / "comparisons"
    image_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    SHOW_IMAGES = False   # ← pon True si quieres que se abra cada figura

    print("\nRESULTADOS DE PRUEBA")
    for i, (pair, data) in enumerate(
        zip(test_pairs, testing_data), start=1
    ):
        hazy_path, clear_path = pair
        hazy, clear = data

        quality, restored, psnr, ssim = evaluate_image(
            clear, hazy, best_params
        )

        # Guardar imagen restaurada
        output_name = f"{i:03d}_{hazy_path.stem}_dehazed.png"
        cv2.imwrite(str(image_dir / output_name), restored)

        # Guardar comparación (hazy | dehazed | GT)
        cmp_name = f"{i:03d}_{hazy_path.stem}_comparison.png"
        save_comparison(
            hazy, restored, clear,
            comparison_dir / cmp_name,
            title=f"{hazy_path.name}  |  PSNR={psnr:.2f} dB  SSIM={ssim:.4f}",
        )

        # Mostrar en pantalla (opcional)
        if SHOW_IMAGES:
            show_comparison(
                hazy, restored, clear,
                title=f"{hazy_path.name}  |  PSNR={psnr:.2f} dB  SSIM={ssim:.4f}",
            )

        rows.append({
            "hazy_image":      hazy_path.name,
            "clear_reference": clear_path.name,
            "PSNR_dB":         psnr,
            "SSIM":            ssim,
            "quality":         quality,
            "output_image":    output_name,
            "comparison_image": cmp_name,
        })

        print(
            f"{hazy_path.name}: "
            f"PSNR={psnr:.3f} dB, SSIM={ssim:.5f}"
        )
    # --- Resumen ---
    summary = {
        "method":               "Adaptive Selection Hyper-Heuristic",
        "training_images":      len(training_data),
        "testing_images":       len(testing_data),
        "iterations":           ITERATIONS,
        "population_size":      POPULATION_SIZE,
        "seed":                 SEED,
        "best_training_score":  float(best_score),
        "mean_test_PSNR":       float(np.mean([r["PSNR_dB"] for r in rows])),
        "mean_test_SSIM":       float(np.mean([r["SSIM"]    for r in rows])),
        "best_parameters":      best_params,
        "search_time_seconds":  float(elapsed),
        "operator_statistics":  statistics,
    }

    # --- Persistencia ---
    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(
        OUTPUT_DIR / "test_metrics.csv", "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with open(
        OUTPUT_DIR / "search_history.csv", "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    print("\nRESUMEN FINAL")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nResultados guardados en: {OUTPUT_DIR}")


# ============================================================
# 17. PUNTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    main()