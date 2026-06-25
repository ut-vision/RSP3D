# Prosthesis-Aware 3D Human Pose Estimation: A Dataset and Benchmark for RSP Users

Official implementation of the paper by Yilin Wen, Kechuan Dong, Fumiya Suginaka, Ken Endo, and Yusuke Sugano: "Prosthesis-Aware 3D Human Pose Estimation: A Dataset and Benchmark for RSP Users", ECCV 2026.

[[Paper (arXiv)]]() | [[Supplementary]]() | [[Dataset]](https://ut-vision.github.io/RSP3D/download.html) | [[Project Page]](https://ut-vision.github.io/RSP3D/)


## Citation

If you find this work helpful, please consider citing:
```bibtex
@inproceedings{wen2026rsp3d,
  title     = {Prosthesis-Aware 3D Human Pose Estimation: A Dataset and Benchmark for RSP Users},
  author    = {Wen, Yilin and Dong, Kechuan and Suginaka, Fumiya and Endo, Ken and Sugano, Yusuke},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026},
}
```


## Visualize Data

### 1. Set the dataset path

Download the dataset from the [Dataset page](https://ut-vision.github.io/RSP3D/download.html) and set the path in `utils/constants.py`:

```python
PATH_DATASET = '/path/to/RSP3D/'   # root containing P1/, P2/, … subdirs
```

### 2. Install dependencies

No heavy dependencies (PyTorch is not required).

```bash
pip install numpy opencv-python matplotlib h5py
```

### 3. Browse and visualize

The script lists all action segments and saves GT annotation visualizations (skeleton + RSP blade edges projected onto video frames) to `./visualizations/`.

```bash
cd data
python dataset.py                    # all subjects
python dataset.py --subject_tag P1   # single subject
```

Output images are written to `./data/visualizations/<subject>/<action>_<camera>_<frame>.jpg`.


## Hybrid Alignment and Evaluation

We propose a hybrid alignment pipeline that combines **SAM3D** (model-based body estimation) and **SpatialTracker-v2** (model-free reconstruction) to produce 3D body and RSP blade estimation.

### 1. Download assets

Download the pre-computed detection results and model-based / model-free outputs from the link below, and set the path in `utils/constants.py`:

```
[Google Drive link — to be provided]
```

```python
PATH_ASSETS = '/path/to/assets/'   # root containing detection/, sam3d/, spatial_tracker/ subdirs
```

### 2. Run alignment

`scripts/alignment.py` fuses per-frame SAM3D body pose and SpatialTracker-v2 point-map estimates. It aligns the model-based skeleton into the model-free coordinate frame using a two-stage PnP + root-aligned scale fitting, and writes results to HDF5.

```bash
cd scripts
python alignment.py                      # all subjects
python alignment.py --subject_tag P3     # single subject
```

Results are written to `PATH_ASSETS/alignment/<subject>/<action>_<camera>.h5`.

### 3. Run evaluation

`scripts/eval.py` loads the alignment HDF5 outputs, computes per-frame body pose (MPJPE, PA-MPJPE) and RSP blade metrics (Chamfer distance, F1 score at multiple thresholds), and saves per-subject NPZ files.

```bash
python eval.py                           # all subjects
python eval.py --subject_tag P3          # single subject
```

Results are written to `PATH_ASSETS/eval/<subject_tag>.npz`.

### 4. Report results

`scripts/report_eval.py` aggregates all per-subject NPZ files and prints a summary table with frame-weighted means and a ready-to-paste LaTeX row.

```bash
python report_eval.py
```

Edit the `results_dir` variable at the top of `main()` to point to your eval output folder if needed.

