# SVM Kernel Trick 3D Interactive Demo

Live Demo: https://l13-svmdemo.streamlit.app/

This is a Python teaching project for visualizing the intuition behind the SVM
Kernel Trick with an RBF Kernel SVM.

The project includes:

- A Streamlit + Plotly interactive app
- A scikit-learn `SVC(kernel="rbf")` classifier
- A 2D decision boundary plot with support vectors
- A 3D `decision_function` surface over the original 2D input space
- A simplified 2D-to-3D teaching visualization
- A Manim animation scene for explaining the same idea

## Project Introduction

The app generates nonlinear circle data with `sklearn.datasets.make_circles`,
trains an RBF Kernel SVM, and shows how the learned classifier behaves. The goal
is not to prove the full math of the RBF feature space, but to build a visual
bridge from a hard 2D classification problem to the intuition of a lifted
feature representation.

## What Is The Kernel Trick?

The Kernel Trick lets an algorithm work as if data had been mapped into a
higher-dimensional feature space, without explicitly computing every coordinate
in that space. Instead, a kernel function computes similarity values between
samples.

For RBF SVM, the model uses:

```text
K(x, x') = exp(-gamma * ||x - x'||^2)
```

This kernel can represent very flexible nonlinear decision boundaries.

## What Is RBF SVM?

An SVM tries to separate classes with a margin. With an RBF kernel, the
classifier can create nonlinear boundaries in the original input space.

Main parameters:

- `C`: controls how strictly the model tries to classify training points.
- `gamma`: controls how local the RBF similarity is.
- Support vectors: training samples that directly shape the margin and decision
  boundary.

## Important Mathematical Note

This project uses simplified 3D mappings such as:

```text
z = x1^2 + x2^2
z = exp(-(x1^2 + x2^2))
```

to visually explain the intuition behind the Kernel Trick.

However, these mappings are only for educational demonstration.

The actual RBF kernel used by sklearn SVM does not explicitly transform data
into this simple 3D space. Instead, the RBF kernel computes similarity in an
implicit high-dimensional feature space.

The 3D decision function surface shown in this project represents the classifier
score over the original 2D input space, not the actual high-dimensional feature
representation.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For local Manim rendering, install the optional development dependencies:

```bash
pip install -r requirements-dev.txt
```

Streamlit Community Cloud only needs `requirements.txt` to run the app. Manim is
kept in `requirements-dev.txt` because the deployed app displays an already
rendered video when one is available.

## Run Streamlit App

From the project root:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Run Manim Animation

Because this project intentionally contains a folder named `manim/`, the most
reliable Windows command is to run Manim from the parent directory:

```bash
cd ..
python -m manim -pql svm-kernel-trick-demo/manim/svm_kernel_trick_scene.py SVMKernelTrickScene --media_dir svm-kernel-trick-demo/assets/manim_output
```

For a high-quality preview:

```bash
python -m manim -pqh svm-kernel-trick-demo/manim/svm_kernel_trick_scene.py SVMKernelTrickScene --media_dir svm-kernel-trick-demo/assets/manim_output
```

Manim may require FFmpeg and a working text rendering stack. The `manim`
Python package installs the Manim Community Edition tools, but system-level
video or font dependencies can still vary by machine.

Rendered videos are written under `assets/manim_output/`. A low-quality render
usually appears at:

```text
assets/manim_output/videos/svm_kernel_trick_scene/480p15/SVMKernelTrickScene.mp4
```

## Project Structure

```text
svm-kernel-trick-demo/
|
|-- app.py
|-- requirements.txt
|-- README.md
|
|-- src/
|   |-- data_utils.py
|   |-- svm_model.py
|   |-- plot_utils.py
|   |-- kernel_mapping.py
|
|-- manim/
|   |-- svm_kernel_trick_scene.py
|
|-- assets/
    |-- manim_output/
```

## Learning Objectives

After using this project, students should be able to:

- Explain why concentric circle data is not linearly separable in the original
  2D input space.
- Describe the intuition of mapping data into a higher-dimensional view.
- Distinguish the simplified 3D teaching map from the true implicit RBF kernel
  feature space.
- Interpret support vectors, `C`, `gamma`, and the SVM `decision_function`.
- Explain why the 3D score surface is a classifier output over 2D inputs, not
  the actual kernel-transformed feature representation.

## Screenshots Placeholder

Add screenshots here after running the Streamlit app:

- Kernel Trick Concept tab
- RBF SVM Classifier tab
- Decision Function Surface tab
- Manim Animation tab

## Teaching Flow

1. Start with the 2D circle dataset.
2. Show that a straight line cannot separate the classes well in 2D.
3. Show a simplified 3D mapping to build intuition.
4. Explain that RBF Kernel SVM does not explicitly use that 3D mapping.
5. Show the 2D decision boundary and support vectors.
6. Show the 3D `decision_function` surface as model score over the original 2D
   input grid.
7. Use the Manim scene as a short animated explanation.

## Controls

The Streamlit sidebar includes:

- `n_samples`: number of generated circle samples
- `noise`: noise level in the generated data
- `factor`: inner circle size relative to the outer circle
- `random_state`: reproducible random seed
- `C`: SVM regularization parameter
- `gamma`: RBF kernel locality parameter
- `grid_size`: decision boundary and surface resolution
- `mapping_method`: simplified 3D teaching formula

## Project Placeholder

The `assets/manim_output/` folder is a placeholder for rendered Manim files.
The Streamlit app can run without a rendered video. If the Manim output video is
present, the app displays it in the Manim tab.
