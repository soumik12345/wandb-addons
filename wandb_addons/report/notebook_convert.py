import re
import yaml
from typing import List, Optional, Union

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
    for cell in tqdm(notebook_cells, desc="Converting notebook cells to report cells"):
        if cell["type"] == "markdown":
            blocks.append(wr.MarkdownBlock(text=cell["source"]))
        elif cell["type"] == "code":
            blocks.append(wr.MarkdownBlock(text=f"```python\n{cell['source']}\n```"))
            wr.Runset()

    report.blocks = blocks
    report.save()
    wandb.termlog(
        "Report {report_title} created successfully. "
        + f"Check list of reports at {report.url}."
    )
