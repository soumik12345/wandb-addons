import numpy as np
import plotly.graph_objects as go
import wandb


class AnnotatedImage(wandb.Html):
    def __init__(self, image: np.array, mask=np.array) -> None:
        fig = go.Figure(
            [
                go.Image(name="image", z=image, opacity=1),  # trace 0
                go.Image(name="mask", z=mask, opacity=0.85),  # trace 1
            ]
        )
        fig.update_layout(
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=False),
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    buttons=[
                        dict(
                            label="All",
                            method="update",
                            args=[{"visible": [True, True]}],
                        ),
                    ],
                    y=1,
                    yanchor="top",
                ),
                dict(
                    type="buttons",
                    direction="right",
                    buttons=[
                        dict(
                            label="Image",
                            method="update",
                            args=[{"visible": [True, False]}],
                        ),
                    ],
                    y=0.9,
                    yanchor="top",
                ),
                dict(
                    type="buttons",
                    direction="right",
                    buttons=[
                        dict(
                            label="Mask",
                            method="update",
                            args=[{"visible": [False, True]}],
                        ),
                    ],
                    y=0.8,
                    yanchor="top",
                ),
            ],
        )
        super().__init__(fig.to_html(), inject=True)
