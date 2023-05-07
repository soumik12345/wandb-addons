from typing import List, Optional

import wandb
import wandb.apis.reports as wr

import nbformat
from nbformat import NotebookNode


def _parse_notebook_cells(filepath: str) -> List[str]:
    with open(filepath, "r") as f:
        nb = nbformat.read(f, as_version=4)

    cells = []
    for cell in nb.cells:
        if cell.cell_type in ["code", "markdown"]:
            cells.append((cell.source, cell.cell_type))

    return cells


def convert_to_wandb_report(
    filepath: str,
    wandb_project: str,
    wandb_entity: str,
    report_title: Optional[str] = "Untitled Report",
    description: Optional[str] = "",
    width: Optional[str] = "readable",
):
    notebook_cells = _parse_notebook_cells(filepath)

    report = wr.Report(
        project=wandb_project,
        title=report_title,
        entity=wandb_entity,
        description=description,
        width=width,
    )

    blocks = []
    for cell in notebook_cells:
        if cell[1] == "markdown":
            blocks.append(wr.MarkdownBlock(text=cell[0]))
        elif cell[1] == "code":
            blocks.append(wr.MarkdownBlock(text=f"```python\n{cell[0]}\n```"))
            wr.Runset()

    report.blocks = blocks
    report.save()
    wandb.termlog(
        f"Report {report_title} created successfully. "
        + "Check list of reports at https://wandb.ai/{wandb_entity}/{wandb_project}/reportlist."
    )
