# Project Log

Project: SVM Kernel Trick 3D Interactive Demo  
Live Demo: https://l13-svmdemo.streamlit.app/  
Repository: https://github.com/ian0629082/L13-SVM.git

## 2026-06-18

### Initial Project Build

- Created the Python project structure:
  - `app.py`
  - `requirements.txt`
  - `README.md`
  - `src/data_utils.py`
  - `src/svm_model.py`
  - `src/plot_utils.py`
  - `src/kernel_mapping.py`
  - `manim/svm_kernel_trick_scene.py`
  - `assets/manim_output/`
- Built a Streamlit app for teaching the SVM Kernel Trick.
- Added Plotly visualizations:
  - 2D RBF SVM decision boundary
  - support vectors
  - 3D `decision_function` surface
  - simplified 2D-to-3D teaching mapping
- Used `sklearn.svm.SVC(kernel="rbf")` for the classifier.
- Used `sklearn.datasets.make_circles` for nonlinear 2D data.

### Concept Clarification

- Added an important warning in both the app and README:
  - The simplified 3D mapping is only an educational visualization.
  - The RBF Kernel does not explicitly transform data into this 3D space.
  - The 3D `decision_function` surface is a classifier score over the original
    2D input space, not the true kernel feature space.

### Design Review Updates

- Read and applied the project design requirements from `design.md`.
- Adjusted the app to use the specified sidebar controls:
  - `n_samples`
  - `noise`
  - `factor`
  - `C`
  - `gamma`
  - `mapping_method`
  - `random_state`
  - `grid_size`
- Added two simplified mapping options:
  - `z = x1^2 + x2^2`
  - `z = exp(-(x1^2 + x2^2))`
- Added C and gamma explanation tables for beginners.
- Added a simple concept separation plane to the 3D mapping visualization.
- Added beginner-friendly Chinese comments in key source files.

### Manim Animation

- Created `SVMKernelTrickScene`.
- Animated:
  - 2D concentric circle data
  - simplified lift using `z = x^2 + y^2`
  - concept separation plane
  - Kernel Trick explanation text
  - return to 2D with nonlinear decision boundary concept
- Rendered the low-quality preview video to:
  - `assets/manim_output/videos/svm_kernel_trick_scene/480p15/SVMKernelTrickScene.mp4`

### Deployment Preparation

- Added `.gitignore`.
- Split dependencies:
  - `requirements.txt` for Streamlit Cloud deployment
  - `requirements-dev.txt` for local Manim rendering
- Kept Streamlit Cloud dependencies minimal:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `streamlit`
  - `plotly`
- Added README deployment and Manim instructions.

### GitHub And Streamlit Deployment

- Initialized git repository.
- Created initial commit:
  - `bce764d Initial SVM kernel trick Streamlit demo`
- Pushed project to:
  - https://github.com/ian0629082/L13-SVM.git
- Added live demo link to README.
- Created second commit:
  - `0c7dd3d Add live demo link to README`
- Streamlit app deployed at:
  - https://l13-svmdemo.streamlit.app/

### Verification

- Confirmed Python syntax compilation:
  - `python -m compileall .`
- Confirmed core imports and model training.
- Confirmed Plotly figure generation.
- Confirmed local Streamlit response:
  - `http://localhost:8501`
- Confirmed Manim video render.

## Notes

- The 3D mapping is intentionally simplified for teaching.
- The app should remain usable even if Manim is not installed.
- The rendered Manim video is optional and shown only when available.
