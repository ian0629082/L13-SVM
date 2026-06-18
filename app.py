from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.data_utils import generate_dataset
from src.kernel_mapping import MAPPING_METHODS, simplified_3d_mapping
from src.plot_utils import (
    plot_2d_decision_boundary,
    plot_3d_decision_function,
    plot_simplified_kernel_mapping,
)
from src.svm_model import train_rbf_svm


MANIM_VIDEO_PATH = Path(
    "assets/manim_output/videos/svm_kernel_trick_scene/480p15/SVMKernelTrickScene.mp4"
)


st.set_page_config(
    page_title="SVM Kernel Trick 3D Interactive Demo",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)


st.title("SVM Kernel Trick 3D Interactive Demo")
st.markdown(
    """
This project uses an RBF Kernel SVM to classify nonlinear 2D circle data and to
visualize the intuition behind the Kernel Trick.

**Important note:** This project uses a simplified 3D mapping to visualize the
intuition of the Kernel Trick. In reality, the RBF Kernel computes similarities
in an implicit high-dimensional feature space through a kernel function. It does
not explicitly convert each sample into 3D or any finite coordinate system.

**Decision function note:** The 3D `decision_function` surface is not the real
high-dimensional kernel feature space. It is the model's classification score at
each location in the original 2D input space.
"""
)


with st.sidebar:
    st.header("Dataset")
    n_samples = st.slider("n_samples", 50, 1000, 300, 50)
    noise = st.slider("noise", 0.0, 0.5, 0.1, 0.01)
    factor = st.slider("factor", 0.1, 0.9, 0.5, 0.05)
    random_state = st.number_input("random_state", min_value=0, max_value=9999, value=42)

    st.header("RBF SVM")
    c_value = st.slider("C", 0.1, 100.0, 1.0, 0.1)
    gamma_value = st.slider("gamma", 0.01, 10.0, 1.0, 0.01)
    grid_size = st.slider("grid_size", 30, 120, 60, 10)

    st.header("Kernel Trick Visualization")
    mapping_method = st.selectbox("mapping_method", MAPPING_METHODS, index=0)

    if n_samples > 500 or grid_size > 100:
        st.warning(
            "Large sample counts or dense grids may make Streamlit redraws slower."
        )


X, y = generate_dataset(
    n_samples=n_samples,
    noise=noise,
    factor=factor,
    random_state=int(random_state),
)
result = train_rbf_svm(X, y, c_value=c_value, gamma_value=gamma_value)
model = result.model
mapped = simplified_3d_mapping(X, method=mapping_method)
class_counts = pd.Series(y).map({0: "Class 0", 1: "Class 1"}).value_counts()

metric_cols = st.columns(4)
metric_cols[0].metric("Training Accuracy", f"{result.accuracy:.3f}")
metric_cols[1].metric("Support Vectors", len(result.support_vectors))
metric_cols[2].metric("C", f"{c_value:.2f}")
metric_cols[3].metric("gamma", f"{gamma_value:.2f}")

tab_concept, tab_classifier, tab_surface, tab_manim, tab_data = st.tabs(
    [
        "Kernel Trick Concept",
        "RBF SVM Classifier",
        "Decision Function Surface",
        "Manim Animation",
        "Data",
    ]
)

with tab_concept:
    st.subheader("From 2D nonlinear data to a simplified 3D view")
    st.markdown(
        """
The circle dataset is not linearly separable in the original 2D plane. A common
teaching idea is to imagine lifting points into a higher-dimensional view where
a simpler separating surface becomes easier to understand.

This demo lets you choose one simplified explicit mapping:

- `z = x1^2 + x2^2`
- `z = exp(-(x1^2 + x2^2))`

That visual map is only a teaching aid. The RBF SVM used here still relies on
the kernel function to compare samples in an implicit feature space. The green
plane in the 3D view is also only a concept demonstration, not the actual SVM
hyperplane in RBF feature space.
"""
    )
    st.plotly_chart(
        plot_simplified_kernel_mapping(
            original_X=X,
            mapped_X=mapped,
            y=y,
        ),
        use_container_width=True,
    )

with tab_classifier:
    st.subheader("2D RBF SVM decision boundary and support vectors")
    st.plotly_chart(
        plot_2d_decision_boundary(
            model=model,
            X=X,
            y=y,
            resolution=grid_size,
        ),
        use_container_width=True,
    )
    st.markdown(
        """
Support vectors are the training points that shape the SVM margin. Increasing
`C` usually makes the classifier fit training data more strictly. Decreasing
`C` usually allows a wider margin with more tolerance for mistakes.
"""
    )
    st.table(
        pd.DataFrame(
            [
                {
                    "Parameter": "Small C",
                    "Meaning": "Allows more classification errors; smoother boundary.",
                },
                {
                    "Parameter": "Large C",
                    "Meaning": "Tries harder to classify correctly; boundary may become more complex.",
                },
            ]
        )
    )

with tab_surface:
    st.subheader("3D decision_function surface over the 2D input grid")
    st.plotly_chart(
        plot_3d_decision_function(
            model=model,
            X=X,
            y=y,
            resolution=grid_size,
        ),
        use_container_width=True,
    )
    st.markdown(
        """
The zero-score contour is the decision boundary. Positive and negative scores
represent which side of the classifier a point falls on. A larger `gamma` makes
the RBF similarity more local, which can create a more detailed boundary; a
smaller `gamma` produces a smoother, broader boundary.
"""
    )
    st.table(
        pd.DataFrame(
            [
                {
                    "Parameter": "Small gamma",
                    "Meaning": "Each point has a larger influence range; smoother boundary.",
                },
                {
                    "Parameter": "Large gamma",
                    "Meaning": "Each point has a smaller influence range; may overfit.",
                },
            ]
        )
    )

with tab_manim:
    st.subheader("Manim animation")
    st.markdown(
        """
The included Manim scene explains the same 2D-to-3D teaching metaphor. It should
be read as intuition, not as the exact RBF feature map.

Render command from the parent directory:

```bash
python -m manim -pql svm-kernel-trick-demo/manim/svm_kernel_trick_scene.py SVMKernelTrickScene --media_dir svm-kernel-trick-demo/assets/manim_output
```
"""
    )
    if MANIM_VIDEO_PATH.exists():
        st.video(str(MANIM_VIDEO_PATH))
    else:
        st.info(
            "No rendered Manim video was found yet. Run the command above to "
            "generate it under assets/manim_output."
        )

with tab_data:
    st.subheader("Generated samples and model outputs")
    data = pd.DataFrame(
        {
            "x1": X[:, 0],
            "x2": X[:, 1],
            "class": np.where(y == 0, "Class 0", "Class 1"),
            "prediction": np.where(result.predictions == 0, "Class 0", "Class 1"),
            "decision_function": result.decision_scores,
            "mapped_x": mapped[:, 0],
            "mapped_y": mapped[:, 1],
            "mapped_z": mapped[:, 2],
        }
    )
    left, right = st.columns([1, 2])
    with left:
        st.write("Class counts")
        st.dataframe(class_counts.rename("count"), use_container_width=True)
    with right:
        st.dataframe(data, use_container_width=True)
