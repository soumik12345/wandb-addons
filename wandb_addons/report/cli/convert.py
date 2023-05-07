import click

from ..notebook_convert import convert_to_wandb_report


@click.command()
@click.option("--filepath", help="Path to an IPython notebook")
@click.option(
    "--wandb_project",
    help="The name of the Weights & Biases project where you're creating the project",
)
@click.option(
    "--wandb_entity",
    help="The name of the Weights & Biases entity (username or team name)",
)
@click.option(
    "--report_title", default="Untitled Report", help="The title of the report"
)
@click.option("--description", default="", help="The description of the report")
@click.option(
    "--width",
    default="readable",
    help="Width of the report, one of `'readable'`, `'fixed'`, or `'fluid'`",
)
def convert(filepath, wandb_project, wandb_entity, report_title, description, width):
    convert_to_wandb_report(
        filepath=filepath,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        report_title=report_title,
        description=description,
        width=width,
    )


if __name__ == "__main__":
    convert()
