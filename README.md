# try_cotracker

Comandi minimi da lanciare dalla root del repo.

DINO in breve: CoTracker produce le track 2D nel tempo; DINO estrae i descrittori sulle track e il matching finale tra camere usa quei descrittori DINO, filtrati con vincoli geometrici.

## 1. Estrai track + feature DINO su tutte le camere

```bash
python3 main.py \
  --all-cams \
  --dataset-root blade_103706 \
  --output-root outputs/blade_103706
```

## 2. Visualizza le feature DINO multiview

```bash
python3 dino_feature_viewer.py \
  --multiview \
  --dataset-root blade_103706 \
  --output-root outputs/blade_103706
```

Output principale:

```text
outputs/blade_103706/multiview_dino/all_cams_shared_pca.mp4
```

## 3. Calcola il matching finale globale tra tutte le camere

```bash
python3 pair_camera_matching.py \
  --dataset-root blade_103706 \
  --output-root outputs/blade_103706 \
  --min-dino-similarity 0.05 \
  --min-common-dino-similarity 0.00 \
  --min-common-frames 3 \
  --max-world-distance 0.10 \
  --max-reprojection-error 20 \
  --global-min-cameras 3
```

Output principali:

```text
outputs/blade_103706/global_matches/global_match_table.csv
outputs/blade_103706/global_matches/global_matches.json
outputs/blade_103706/global_matches/all_cams_global_matches.mp4
```
