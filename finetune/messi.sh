#!/usr/bin/env bash
# =============================================================================
# setup_and_run_finetune.sh — Instala dependencias y lanza el fine-tuning de
# YOLO11n-obb en un VPS Ubuntu (con o sin GPU NVIDIA).
#
# Uso (desde la carpeta con este script y finetune_yolo_obb.py):
#   chmod +x setup_and_run_finetune.sh
#   ./setup_and_run_finetune.sh                 # instala y entrena
#   ./setup_and_run_finetune.sh --skip-setup    # solo entrena
#   ./setup_and_run_finetune.sh --setup-only    # solo instala
#
# Sobrescribir hiperparametros:  EPOCHS=80 BATCH_SIZE=64 ./setup_and_run_finetune.sh
#
# Recomendado en VPS:  tmux new -s train  antes de lanzarlo, para que el
# entrenamiento sobreviva si se cae la conexion SSH.
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# CONFIGURACION
# -----------------------------------------------------------------------------
DATA_ROOT="/home/ubuntu/exp/data"
DATASET_SIZES=(2500)
OUTPUT_DIR="/home/ubuntu/exp/reports/paper_metrics/finetune"
YOLO_DATA_DIR="/home/ubuntu/exp/yolo_datasets"
TRAIN_SCRIPT="finetune_yolo_obb.py"
VENV_DIR="venv-ft"     # venv propio: no mezcla con el del pipeline baseline
LOG_DIR="logs"

EPOCHS="${EPOCHS:-70}"
BATCH_SIZE="${BATCH_SIZE:-32}"
IMG_SIZE="${IMG_SIZE:-640}"
BENCHMARK_THREADS="${BENCHMARK_THREADS:-4}"   # 4 hilos ~ Raspberry Pi 4/5
DEFAULT_WORKERS=$(( $(nproc) > 1 ? $(nproc) - 1 : 1 ))
NUM_WORKERS="${NUM_WORKERS:-$(( DEFAULT_WORKERS < 8 ? DEFAULT_WORKERS : 8 ))}"

# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------
if [ -t 1 ]; then
  BOLD="\033[1m"; GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; RESET="\033[0m"
else
  BOLD=""; GREEN=""; YELLOW=""; RED=""; RESET=""
fi
info()  { echo -e "${GREEN}[INFO]${RESET} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${RESET} $*"; }
error() { echo -e "${RED}[ERROR]${RESET} $*" >&2; }
step()  { echo -e "\n${BOLD}==> $*${RESET}"; }

SKIP_SETUP=0
SETUP_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --skip-setup) SKIP_SETUP=1 ;;
    --setup-only) SETUP_ONLY=1 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//' | head -n 16; exit 0 ;;
    *) error "Flag desconocido: $arg"; exit 1 ;;
  esac
done

cd "$(dirname "$0")"
SCRIPT_DIR="$(pwd)"
mkdir -p "$LOG_DIR"

# =============================================================================
# FASE 1 — INSTALACION
# =============================================================================
setup() {
  step "Fase 1/3: dependencias de sistema"
  export DEBIAN_FRONTEND=noninteractive
  sudo apt-get update -y
  sudo apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip build-essential \
    libgl1 libglib2.0-0 ca-certificates pciutils

  step "Fase 2/3: verificacion de GPU (leccion aprendida: hardware != driver)"
  # Un 'nvidia-smi' ausente NO significa que no haya GPU: puede faltar el
  # driver. Distinguimos ambos casos con lspci antes de decidir.
  local has_hw=0 has_driver=0
  if lspci | grep -qi -E "nvidia"; then has_hw=1; fi
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then has_driver=1; fi

  if [ "$has_hw" -eq 1 ] && [ "$has_driver" -eq 0 ]; then
    error "Se detecto hardware NVIDIA en el bus PCI, pero el driver NO esta instalado."
    error "Instala el driver ANTES de continuar (de lo contrario se entrenaria en CPU):"
    error "    sudo apt-get install -y ubuntu-drivers-common"
    error "    sudo ubuntu-drivers install"
    error "    sudo reboot"
    error "Tras el reboot, verifica con 'nvidia-smi' y relanza este script."
    exit 1
  fi
  if [ "$has_driver" -eq 1 ]; then
    info "GPU lista:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/       /'
  else
    warn "Sin GPU NVIDIA (ni hardware ni driver). Se entrenara en CPU: sera lento."
  fi

  step "Fase 3/3: entorno virtual + Ultralytics"
  if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    info "Entorno virtual creado en $SCRIPT_DIR/$VENV_DIR"
  else
    info "Reutilizando entorno virtual $SCRIPT_DIR/$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  python -m pip install --upgrade pip wheel setuptools

  if [ "$has_driver" -eq 1 ]; then
    # En Linux, el torch por defecto de PyPI trae CUDA; ultralytics lo arrastra.
    pip install ultralytics
  else
    # Sin GPU: instalar torch CPU-only primero evita bajar ~2GB de CUDA inutil.
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install ultralytics
  fi
  # onnx para el export a la Pi (ncnn lo instala Ultralytics on-demand).
  pip install onnx

  step "Verificacion"
  python - <<'PY'
import torch, ultralytics
print(f"       PyTorch     : {torch.__version__}")
print(f"       Ultralytics : {ultralytics.__version__}")
print(f"       CUDA disp.  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"       GPU         : {torch.cuda.get_device_name(0)}")
PY
  info "Instalacion completada."
}

# =============================================================================
# FASE 2 — VERIFICACIONES PREVIAS
# =============================================================================
preflight() {
  step "Verificaciones previas al fine-tuning"
  if [ ! -f "$TRAIN_SCRIPT" ]; then
    error "No se encontro '$TRAIN_SCRIPT' en $SCRIPT_DIR."
    exit 1
  fi
  if [ ! -d "$DATA_ROOT" ]; then
    error "No existe el directorio de datos: $DATA_ROOT"
    exit 1
  fi
  local missing=0
  for size in "${DATASET_SIZES[@]}"; do
    local d="$DATA_ROOT/dataset_${size}"
    if [ ! -f "$d/etiquetas_${size}.csv" ] || [ ! -d "$d/images" ]; then
      error "Dataset incompleto: $d"
      missing=1
    else
      local n
      n=$(find "$d/images" -maxdepth 1 -name '*.jpg' | wc -l)
      info "dataset_${size}: OK ($n imagenes .jpg)"
    fi
  done
  [ "$missing" -eq 0 ] || exit 1

  # Los datasets YOLO se crean con symlinks (casi no ocupan disco), pero los
  # runs de Ultralytics (checkpoints, plots) si pesan: avisar si queda poco.
  local avail_gb
  avail_gb=$(df --output=avail -BG "$SCRIPT_DIR" | tail -1 | tr -dc '0-9')
  info "Espacio libre: ${avail_gb} GB"
  if [ "${avail_gb:-0}" -lt 5 ]; then
    warn "Menos de 5 GB libres; los runs de Ultralytics podrian llenar el disco."
  fi
}

# =============================================================================
# FASE 3 — FINE-TUNING
# =============================================================================
train() {
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  local timestamp logfile
  timestamp="$(date +%Y%m%d_%H%M%S)"
  logfile="$LOG_DIR/finetune_${timestamp}.log"

  step "Lanzando fine-tuning de YOLO11n-obb"
  info "Datasets : ${DATASET_SIZES[*]}"
  info "Epocas   : $EPOCHS   Batch: $BATCH_SIZE   Img: $IMG_SIZE   Workers: $NUM_WORKERS"
  info "Salida   : $OUTPUT_DIR"
  info "Log      : $SCRIPT_DIR/$logfile"

  stdbuf -oL python "$TRAIN_SCRIPT" \
    --data-root "$DATA_ROOT" \
    --yolo-data-dir "$YOLO_DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --dataset-sizes "${DATASET_SIZES[@]}" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --img-size "$IMG_SIZE" \
    --workers "$NUM_WORKERS" \
    --benchmark-threads "$BENCHMARK_THREADS" \
    2>&1 | tee "$logfile"

  info "Fine-tuning finalizado."
  info "Resumen del paper : $OUTPUT_DIR/paper_experiment_summary.md"
  info "Para la Raspberry : busca las carpetas *_ncnn_model junto a cada best.pt"
}

main() {
  info "Directorio de trabajo: $SCRIPT_DIR"
  if [ "$SKIP_SETUP" -eq 0 ]; then setup; else info "Se omite la instalacion (--skip-setup)."; fi
  if [ "$SETUP_ONLY" -eq 1 ]; then info "Instalacion lista (--setup-only)."; exit 0; fi
  preflight
  train
}

main