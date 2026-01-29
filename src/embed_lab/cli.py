import typer
from pathlib import Path
from typing import Annotated
from embed_lab import templates

app = typer.Typer(name="emb", add_completion=True)


@app.command()
def init(
    path: Annotated[
        Path,
        typer.Argument(
            help="Directory to initialize the lab in"
        ),
    ] = Path("."),
) -> None:
    """
    Initialize a new embed-lab project structure with inventory and experiment templates.
    """
    base_path: Path = path.resolve()

    # Define the project structure: Path -> Content
    structure: dict[Path, str | None] = {
        # Directories (Content is None)
        base_path / "inventory": None,
        base_path / "experiments": None,
        base_path / "results": None,
        # Files (Content is Template)
        base_path / "inventory" / "__init__.py": "",
        base_path
        / "inventory"
        / "embedders.py": templates.TEMPLATE_EMBEDDERS,
        base_path
        / "inventory"
        / "chunkers.py": templates.TEMPLATE_CHUNKERS,
        base_path / "inventory" / "datasets.py": "",
        base_path / "experiments" / "__init__.py": "",
        base_path
        / "experiments"
        / "exp_01_baseline.py": templates.TEMPLATE_EXP_BASELINE,
        base_path / "results" / ".gitkeep": "",
        base_path
        / ".gitignore": templates.TEMPLATE_GITIGNORE,
        base_path / "main.py": templates.TEMPLATE_MAIN,
    }

    typer.secho(
        f"🧪 Initializing Embed Lab in {base_path.name}...",
        fg=typer.colors.BLUE,
    )

    for file_path, content in structure.items():
        # 1. Handle Directories
        if content is None:
            if not file_path.exists():
                file_path.mkdir(parents=True, exist_ok=True)
            continue

        # 2. Handle Files
        # Ensure parent directory exists for files (safety check)
        if not file_path.parent.exists():
            file_path.parent.mkdir(
                parents=True, exist_ok=True
            )

        # Write content if file doesn't exist to prevent overwriting work
        if not file_path.exists():
            file_path.write_text(content, encoding="utf-8")
            typer.secho(
                f"  + Created {file_path.relative_to(base_path)}",
                fg=typer.colors.GREEN,
            )
        else:
            typer.secho(
                f"  . Skipped {file_path.relative_to(base_path)} (exists)",
                fg=typer.colors.YELLOW,
            )

    typer.secho(
        "\n✨ Done! Run your first experiment:",
        fg=typer.colors.BLUE,
        bold=True,
    )
    typer.echo(
        "   uv run embed-lab train experiments/exp_01_baseline.py"
    )


if __name__ == "__main__":
    app()
