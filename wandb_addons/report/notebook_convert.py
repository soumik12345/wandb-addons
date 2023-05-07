import re
import yaml
import json
from typing import Any, Dict, List, Optional, Union

from tqdm.auto import tqdm

import wandb
import wandb.apis.reports as wr

import nbformat
from nbformat import NotebookNode


def _check_cell_for_panelgrid(cell_source) -> Union[str, None]:
    pattern = re.compile(r"---\n(.+?)\n---", re.DOTALL)
    match = pattern.search(cell_source)
    return match.group(1) if match else None


def _parse_notebook_cells(filepath: str) -> List[str]:
    with open(filepath, "r") as f:
        nb = nbformat.read(f, as_version=4)

    cells = []
    for cell in nb.cells:
        if cell.cell_type in ["code", "markdown"]:
            metadata_match = _check_cell_for_panelgrid(cell.source)
            cells.append(
                {"source": yaml.safe_load(metadata_match), "type": "panel_metadata"}
                if metadata_match
                else {"source": cell.source, "type": cell.cell_type}
            )

    return cells


def _convert_metadata_to_panelgrid(metadata: Dict[str, Any]) -> wr.PanelGrid:
    runsets, line_plots = [], []
    for runset_metadata in tqdm(
        metadata["panelgrid"]["runsets"], leave=False, desc="Creating Run Sets"
    ):
        runsets.append(
            wr.Runset(
                project=runset_metadata["project"],
                entity=runset_metadata["entity"],
                name=runset_metadata["name"],
            )
        )
    for line_plot_metadata in tqdm(
        metadata["panelgrid"]["lineplots"], leave=False, desc="Creating Line Plots"
    ):
        line_plots.append(
            wr.LinePlot(x=line_plot_metadata["x"], y=line_plot_metadata["y"])
        )
    return wr.PanelGrid(runsets=runsets, panels=line_plots)


def convert_to_wandb_report(
    filepath: str,
    wandb_project: str,
    wandb_entity: str,
    report_title: Optional[str] = "Untitled Report",
    description: Optional[str] = "",
    width: Optional[str] = "readable",
):
    """Convert an IPython Notebook to a [Weights & Biases Report](https://docs.wandb.ai/guides/reports).

    **Usage:**

    ```python
    from wandb_addons.report import convert_to_wandb_report

    convert_to_wandb_report(
        filepath="./Use_WandbMetricLogger_in_your_Keras_workflow.ipynb",
        wandb_project="report-to-notebook",
        wandb_entity="geekyrakshit",
        report_title="Use WandbMetricLogger in your Keras Workflow",
        description="A guide to using the WandbMetricLogger callback in your Keras and TensorFlow training worflow"
    )
    ```

    !!! note
        In order to include panel grids with runsets and line plots in your report, you need to include
        YAML metadata regarding the runsets and line plots you want to include in a panel grid in a
        markdown cell of your notebook in the following format:

        ```yaml
        ---
        panelgrid:
        runsets:
        - project: report-to-notebook
            entity: geekyrakshit
            name: Training-Logs
        lineplots:
        - x: batch/batch_step
            y: batch/accuracy
        - x: batch/batch_step
            y: batch/learning_rate
        - x: batch/batch_step
            y: batch/loss
        - x: batch/batch_step
            y: batch/top@5_accuracy
        - x: epoch/epoch
            y: epoch/accuracy
        - x: epoch/epoch
            y: epoch/learning_rate
        - x: epoch/epoch
            y: epoch/loss
        - x: epoch/epoch
            y: epoch/top@5_accuracy
        - x: epoch/epoch
            y: epoch/val_accuracy
        - x: epoch/epoch
            y: epoch/val_loss
        - x: epoch/epoch
            y: epoch/val_top@5_accuracy
        ---
        ```

        Currently only line plots are supported inside panel grids.
    
    ??? example "Example"
        The following report was generated for [this](https://github.com/wandb/examples/blob/master/colabs/keras/Use_WandbMetricLogger_in_your_Keras_workflow.ipynb) notebook:
        <iframe src="https://wandb.ai/geekyrakshit/report-to-notebook/reports/Use-WandbMetricLogger-in-your-Keras-Workflow--Vmlldzo0Mjg4NTM2" style="border:none;height:1024px;width:100%">

    Args:
        filepath (str): Path to an IPython notebook.
        wandb_project (str): The name of the Weights & Biases project where you're creating the project.
        wandb_entity (str): The name of the Weights & Biases entity (username or team name).
        report_title (Optional[str]): The title of the report.
        description (Optional[str]): The description of the report.
        width (Optional[str]): Width of the report, one of `"readable"`, `"fixed"`, or `"fluid"`.
    """
    notebook_cells = _parse_notebook_cells(filepath)

    report = wr.Report(
        project=wandb_project,
        title=report_title,
        entity=wandb_entity,
        description=description,
        width=width,
    )

    blocks = []
    for cell in tqdm(notebook_cells, desc="Converting notebook cells to report cells"):
        if cell["type"] == "markdown":
            blocks.append(wr.MarkdownBlock(text=cell["source"]))
        elif cell["type"] == "code":
            blocks.append(wr.MarkdownBlock(text=f"```python\n{cell['source']}\n```"))
        elif cell["type"] == "panel_metadata":
            blocks.append(_convert_metadata_to_panelgrid(metadata=cell["source"]))

    report.blocks = blocks
    report.save()
    wandb.termlog(
        "Report {report_title} created successfully. "
        + f"Check list of reports at {report.url}."
    )
