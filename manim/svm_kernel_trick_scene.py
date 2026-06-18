import numpy as np
from manim import *


class SVMKernelTrickScene(ThreeDScene):
    def construct(self):
        title = Text("SVM Kernel Trick: 2D intuition to 3D view", font_size=34)
        subtitle = Text(
            "The 3D lift is a teaching metaphor, not the full RBF feature map.",
            font_size=22,
            color=YELLOW,
        ).next_to(title, DOWN)

        self.play(Write(title), FadeIn(subtitle, shift=DOWN))
        self.wait(1.5)
        self.play(FadeOut(title), FadeOut(subtitle))

        axes_2d = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=6,
            y_length=6,
            tips=False,
        )
        axes_2d_labels = axes_2d.get_axis_labels(Text("x1"), Text("x2"))

        inner_points = VGroup()
        outer_points = VGroup()
        for angle in range(0, 360, 24):
            theta = angle * DEGREES
            inner_points.add(
                Dot(
                    axes_2d.c2p(0.9 * np.cos(theta), 0.9 * np.sin(theta)),
                    color=BLUE,
                    radius=0.06,
                )
            )
            outer_points.add(
                Dot(
                    axes_2d.c2p(2.0 * np.cos(theta), 2.0 * np.sin(theta)),
                    color=RED,
                    radius=0.06,
                )
            )

        boundary = Circle(radius=1.45, color=WHITE).move_to(axes_2d.c2p(0, 0))
        boundary_label = Text("Nonlinear boundary in 2D", font_size=26).to_edge(UP)

        self.play(Create(axes_2d), FadeIn(axes_2d_labels))
        self.play(LaggedStartMap(FadeIn, inner_points), LaggedStartMap(FadeIn, outer_points))
        self.play(Create(boundary), Write(boundary_label))
        self.wait(1.5)

        self.play(
            FadeOut(boundary),
            FadeOut(boundary_label),
            axes_2d.animate.scale(0.7).to_edge(LEFT),
            axes_2d_labels.animate.scale(0.7).to_edge(LEFT),
            inner_points.animate.scale(0.7).to_edge(LEFT),
            outer_points.animate.scale(0.7).to_edge(LEFT),
        )

        lift_text = Text(
            "Simplified lift: z = x^2 + y^2",
            font_size=28,
            color=YELLOW,
        ).to_edge(UP)
        note = Text(
            "RBF SVM uses kernel similarities instead of explicitly building this 3D map.",
            font_size=21,
        ).next_to(lift_text, DOWN)
        self.play(Write(lift_text), FadeIn(note))
        self.wait(1)

        self.move_camera(phi=65 * DEGREES, theta=-50 * DEGREES, run_time=1.5)

        axes_3d = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[0, 2.5, 0.5],
            x_length=6,
            y_length=6,
            z_length=3,
            tips=False,
        ).shift(RIGHT * 1.6)
        z_label = Text("z", font_size=24).move_to(axes_3d.c2p(0, 0, 2.7))

        mapped_inner = VGroup()
        mapped_outer = VGroup()
        for angle in range(0, 360, 24):
            theta = angle * DEGREES
            xi, yi = 0.9 * np.cos(theta), 0.9 * np.sin(theta)
            xo, yo = 2.0 * np.cos(theta), 2.0 * np.sin(theta)
            zi = 0.45 * (xi**2 + yi**2)
            zo = 0.45 * (xo**2 + yo**2)
            mapped_inner.add(Dot3D(axes_3d.c2p(xi, yi, zi), color=BLUE, radius=0.055))
            mapped_outer.add(Dot3D(axes_3d.c2p(xo, yo, zo), color=RED, radius=0.055))

        plane = Surface(
            lambda u, v: axes_3d.c2p(u, v, 0.75),
            u_range=[-2.4, 2.4],
            v_range=[-2.4, 2.4],
            resolution=(2, 2),
            fill_opacity=0.25,
            checkerboard_colors=[GREEN_E, GREEN_E],
        )
        plane_label = Text("Linear cut becomes plausible", font_size=24, color=GREEN).to_edge(DOWN)

        self.play(Create(axes_3d), FadeIn(z_label))
        self.play(TransformFromCopy(inner_points, mapped_inner), TransformFromCopy(outer_points, mapped_outer))
        self.play(Create(plane), Write(plane_label))
        self.wait(2)

        final_note = Text(
            "Kernel Trick: compute inner products through K(x, x'), avoid explicit coordinates.",
            font_size=25,
            color=YELLOW,
        ).to_edge(UP)
        self.play(Transform(lift_text, final_note), FadeOut(note), FadeOut(plane_label))
        self.begin_ambient_camera_rotation(rate=0.15)
        self.wait(4)
        self.stop_ambient_camera_rotation()

        self.play(
            FadeOut(axes_2d),
            FadeOut(axes_2d_labels),
            FadeOut(inner_points),
            FadeOut(outer_points),
            FadeOut(axes_3d),
            FadeOut(z_label),
            FadeOut(mapped_inner),
            FadeOut(mapped_outer),
            FadeOut(plane),
            FadeOut(lift_text),
        )
        self.move_camera(phi=0 * DEGREES, theta=-90 * DEGREES, run_time=1)

        final_axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=6,
            y_length=6,
            tips=False,
        )
        final_inner = VGroup()
        final_outer = VGroup()
        for angle in range(0, 360, 24):
            theta = angle * DEGREES
            final_inner.add(
                Dot(
                    final_axes.c2p(0.9 * np.cos(theta), 0.9 * np.sin(theta)),
                    color=BLUE,
                    radius=0.06,
                )
            )
            final_outer.add(
                Dot(
                    final_axes.c2p(2.0 * np.cos(theta), 2.0 * np.sin(theta)),
                    color=RED,
                    radius=0.06,
                )
            )

        final_boundary = Circle(radius=1.45, color=YELLOW).move_to(final_axes.c2p(0, 0))
        return_title = Text("Back in 2D: the learned boundary is nonlinear", font_size=28).to_edge(UP)
        return_note = Text(
            "Kernel Trick gives the SVM this nonlinear behavior without explicit 3D coordinates.",
            font_size=22,
            color=YELLOW,
        ).to_edge(DOWN)

        self.play(Create(final_axes), FadeIn(final_inner), FadeIn(final_outer))
        self.play(Create(final_boundary), Write(return_title), FadeIn(return_note))
        self.wait(2)
