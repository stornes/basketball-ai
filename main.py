"""Basketball Video Analysis Platform - CLI entrypoint."""

import click

from app.pipeline.pipeline_config import PipelineConfig, detect_device
from app.pipeline.run_analysis import PipelineOrchestrator


@click.command()
@click.argument("video_path", type=click.Path(exists=True))
@click.option("--output-dir", default="data/outputs", help="Output directory")
@click.option("--device", default="auto", help="Compute device: auto|mps|cpu")
@click.option("--sample-rate", default=2, help="Process every Nth frame")
@click.option("--no-clips", is_flag=True, help="Skip clip generation")
@click.option("--no-agent", is_flag=True, help="Skip AI coaching report")
@click.option(
    "--llm-backend",
    default="template",
    type=click.Choice(["gemini", "template"]),
    help="LLM backend for coaching report",
)
def main(video_path, output_dir, device, sample_rate, no_clips, no_agent, llm_backend):
    """Analyze a basketball game video.

    Produces: shot_chart.png, player_stats.json, possessions.json,
    highlight clips, and game_report.md
    """
    config = PipelineConfig(
        device=device if device != "auto" else detect_device(),
        frame_sample_rate=sample_rate,
        output_dir=output_dir,
        enable_clips=not no_clips,
        enable_coaching_agent=not no_agent,
        llm_backend=llm_backend,
    )

    print(f"Basketball AI Analysis")
    print(f"Device: {config.device}")
    print(f"Video: {video_path}")
    print(f"Output: {config.output_dir}")
    print()

    orchestrator = PipelineOrchestrator(config)
    result = orchestrator.run(video_path)

    print()
    print("=== Results ===")
    print(f"Shot chart: {result.chart_path}")
    print(f"Player stats: {result.stats_path}")
    print(f"Possessions: {result.possessions_path}")
    if result.clip_paths:
        print(f"Highlights: {len(result.clip_paths)} clips")
    if result.report_path:
        print(f"Coach report: {result.report_path}")


if __name__ == "__main__":
    main()
