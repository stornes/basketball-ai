"""Basketball Video Analysis Platform - CLI entrypoint."""

from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv(override=True)  # Load API keys from .env (override empty shell vars)

from app.pipeline.pipeline_config import PipelineConfig, detect_device


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """Basketball AI - Video Analysis & Training Platform."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--output-dir", default="data/outputs", help="Output directory")
@click.option("--device", default="auto", help="Compute device: auto|mps|cpu")
@click.option("--sample-rate", default=6, help="Process every Nth frame (6=fast, 3=detailed)")
@click.option("--no-clips", is_flag=True, help="Skip clip generation")
@click.option("--no-agent", is_flag=True, help="Skip AI coaching report")
@click.option("--model", default="yolov8n.pt", help="YOLO model weights path")
@click.option(
    "--class-map",
    default=None,
    help='Class mapping JSON, e.g. \'{"0":"ball","1":"player"}\'',
)
@click.option(
    "--llm-backend",
    default="template",
    type=click.Choice(["gemini", "grok", "template"]),
    help="LLM backend for coaching report",
)
@click.option("--roster", default=None, type=click.Path(exists=True), help="Path to roster JSON file")
@click.option("--quarter-duration", default=600, help="Quarter duration in seconds (600=FIBA U16, 720=NBA)")
@click.option("--game-start", default=None, help="Game start as wall-clock time (HH:MM:SS) from scoreboard, or seconds offset")
@click.option("--vlm-backend", default="gemini", type=click.Choice(["anthropic", "gemini", "grok"]), help="VLM backend for jersey number recognition (gemini=Gemini 3.1 Pro, anthropic=Claude Sonnet, grok=Grok-2 Vision)")
@click.option("--profile", default="youth", type=click.Choice(["professional", "youth"]), help="Box score profile (professional=full NBA, youth=coaching-focused)")
@click.option("--scorekeeper", default=None, type=click.Path(exists=True), help="Path to scorekeeper JSON for manual stats (OREB, DREB, AST, TO, STL, BLK, PF, MIN, +/-)")
def analyse(video_path, output_dir, device, sample_rate, no_clips, no_agent, model, class_map, llm_backend, roster, quarter_duration, game_start, vlm_backend, profile, scorekeeper):
    """Analyse a basketball game video.

    Produces: shot_chart.png, player_stats.json, possessions.json,
    highlight clips, and game_report.md
    """
    import json

    from app.pipeline.run_analysis import PipelineOrchestrator

    parsed_class_map = None
    if class_map:
        parsed_class_map = {int(k): v for k, v in json.loads(class_map).items()}

    # Parse game start time
    game_start_sec = 0.0
    if game_start:
        if ":" in game_start:
            # Wall-clock format HH:MM:SS — compute offset from video start
            parts = game_start.split(":")
            start_abs = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            # Try to get video start time from filename (e.g., "2026-03-14 14:07:24.MP4")
            import re as _re
            video_name = Path(video_path).stem
            m = _re.search(r"(\d{2}):(\d{2}):(\d{2})", video_name)
            if m:
                video_abs = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
                game_start_sec = start_abs - video_abs
                print(f"Game start: {game_start} (video starts at "
                      f"{m.group(0)}, offset={game_start_sec:.0f}s)")
            else:
                click.echo(f"Warning: Could not parse video start time from filename. "
                           f"Using {game_start} as absolute seconds.", err=True)
                game_start_sec = start_abs
        else:
            game_start_sec = float(game_start)

    config = PipelineConfig(
        device=device if device != "auto" else detect_device(),
        yolo_model=model,
        frame_sample_rate=sample_rate,
        output_dir=output_dir,
        enable_clips=not no_clips,
        enable_coaching_agent=not no_agent,
        llm_backend=llm_backend,
        class_map=parsed_class_map,
        roster_path=roster,
        quarter_duration_sec=quarter_duration,
        game_start_sec=game_start_sec,
        vlm_backend=vlm_backend,
        box_score_profile=profile,
        scorekeeper_path=scorekeeper,
    )

    print("Basketball AI Analysis")
    print(f"Device: {config.device}")
    print(f"Model: {config.yolo_model}")
    print(f"Video: {video_path}")
    print(f"Output: {config.output_dir}")
    print()

    orchestrator = PipelineOrchestrator(config)
    result = orchestrator.run(video_path)

    print()
    print("=== Results ===")
    print(f"Shot charts: {len(result.chart_paths)} files")
    for p in result.chart_paths:
        print(f"  {p}")
    print(f"Player stats: {result.stats_path}")
    print(f"Possessions: {result.possessions_path}")
    if result.clip_paths:
        print(f"Highlights: {len(result.clip_paths)} clips")
    if result.report_path:
        print(f"Coach report: {result.report_path}")


@cli.command()
@click.option("--output-dir", default="data/datasets", help="Dataset output directory")
@click.option("--workspace", default="basketball-yolo-dataset", help="Roboflow workspace")
@click.option("--project", default="basketball-yolo-dataset", help="Roboflow project")
@click.option("--version", default=1, help="Dataset version")
def download(output_dir, workspace, project, version):
    """Download basketball dataset from Roboflow."""
    from app.training.download_dataset import download_dataset

    download_dataset(
        output_dir=output_dir,
        workspace=workspace,
        project=project,
        version=version,
    )


@cli.command()
@click.option("--data", default=None, help="Path to data.yaml (auto-detected if omitted)")
@click.option("--base-model", default="yolov8n.pt", help="Base model weights")
@click.option("--epochs", default=50, help="Training epochs")
@click.option("--batch", default=16, help="Batch size")
@click.option("--imgsz", default=640, help="Image size")
@click.option("--device", default="auto", help="Device: auto|mps|cuda|cpu")
@click.option("--freeze", default=0, help="Number of backbone layers to freeze")
def train(data, base_model, epochs, batch, imgsz, device, freeze):
    """Fine-tune YOLOv8 on basketball dataset."""
    from app.training.train import TrainingConfig, find_dataset_yaml, train as run_train

    # Auto-detect data.yaml if not provided
    if data is None:
        data = find_dataset_yaml("data/datasets/basketball-yolo-dataset")
        if data is None:
            raise click.ClickException(
                "No data.yaml found. Run 'python main.py download' first, "
                "or specify --data path/to/data.yaml"
            )
        print(f"Auto-detected dataset config: {data}")

    config = TrainingConfig(
        data_yaml=data,
        base_model=base_model,
        epochs=epochs,
        batch_size=batch,
        img_size=imgsz,
        device=device if device != "auto" else detect_device(),
        freeze=freeze,
    )

    best_weights = run_train(config)
    print(f"\nTo use fine-tuned model:")
    print(f"  python main.py analyse video.mp4 --model {best_weights}")


if __name__ == "__main__":
    cli()
