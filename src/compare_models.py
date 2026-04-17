from evaluation import compute_metrics, print_metrics
from energy_vad_demo import run_energy_vad   # encapsulation
from dl_inference import run_dl_inference

# ===== Energy VAD =====
gt_energy, pred_energy = run_energy_vad()

metrics_energy = compute_metrics(gt_energy, pred_energy)
print_metrics("Energy VAD", metrics_energy)

# ===== DL VAD =====
gt_dl, pred_dl = run_dl_inference(...)

metrics_dl = compute_metrics(gt_dl, pred_dl)
print_metrics("CNN VAD", metrics_dl)