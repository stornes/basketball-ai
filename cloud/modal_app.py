"""HoopsVision Cloud Pipeline — Modal deployment.

Run the full basketball analysis pipeline on cloud GPUs.
Zero local compute after video upload.

Usage:
    # Upload model weights (one-time)
    modal run cloud/modal_app.py::upload_model

    # Analyse a game
    modal run cloud/modal_app.py::analyse --video /path/to/game.mp4 \
        --roster /path/to/roster.json --match-id 8254973

    # Full pipeline (upload + analyse + report)
    modal run cloud/modal_app.py::full_pipeline --video /path/to/game.mp4 \
        --roster /path/to/roster.json --match-id 8254973
"""

import modal
import os

# ── App and infrastructure ──────────────────────────────────────

app = modal.App("hoopsvision")

# Persistent volumes
model_volume = modal.Volume.from_name("hoopsvision-models", create_if_missing=True)
video_volume = modal.Volume.from_name("hoopsvision-videos", create_if_missing=True)
output_volume = modal.Volume.from_name("hoopsvision-outputs", create_if_missing=True)

# Container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "ultralytics>=8.4.0",
        "opencv-python-headless>=4.8.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0",
        "deep-sort-realtime>=1.3.0",
        "supervision>=0.27.0",
        "anthropic>=0.40.0",
        "openai>=1.0.0",
        "python-dotenv>=1.0.0",
        "langchain-google-genai>=2.0.0",
        "langchain-core>=0.3.0",
        "langgraph>=0.1.0",
        "click>=8.0.0",
        "pyyaml>=6.0",
        "easyocr>=1.7.0",
        "imageio-ffmpeg>=0.4.0",
    )
    .add_local_dir("app", remote_path="/app/app")
    .add_local_dir("scripts", remote_path="/app/scripts")
    .add_local_file("main.py", remote_path="/app/main.py")
)


# ── Upload model weights (one-time setup) ───────────────────────

@app.function(volumes={"/models": model_volume}, image=image)
def upload_model():
    """Upload YOLO model weights to persistent volume."""
    import shutil
    # This function is called locally with modal run
    # The weights are copied via the local entrypoint below
    print("Model volume ready at /models")
    model_volume.commit()


@app.local_entrypoint()
def setup_models(
    weights: str = "runs/detect/runs/detect/yolo26m_basketball_640/weights/best.pt",
):
    """Upload model weights from local machine to Modal volume."""
    import subprocess
    print(f"Uploading {weights} to Modal volume...")
    # Use modal volume put
    subprocess.run(
        ["python", "-m", "modal", "volume", "put", "hoopsvision-models", weights, "/best.pt"],
        check=True,
    )
    print("Done. Model weights uploaded.")


# ── Transcode (4K → 1080p) ──────────────────────────────────────

@app.function(
    volumes={"/videos": video_volume},
    image=image,
    cpu=4,
    memory=8192,
    timeout=1800,  # 30 min for large 4K videos
)
def transcode(game_id: str) -> str:
    """Transcode 4K video to 1080p for faster processing."""
    import subprocess

    input_path = f"/videos/{game_id}/original.mp4"
    output_path = f"/videos/{game_id}/1080p.mp4"

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Video not found: {input_path}")

    if os.path.exists(output_path):
        print(f"Already transcoded: {output_path}")
        return output_path

    print(f"Transcoding {input_path} -> 1080p...")
    subprocess.run([
        "ffmpeg", "-i", input_path,
        "-vf", "scale=1920:1080",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "copy",
        "-y", output_path,
    ], check=True, capture_output=True)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"Transcoded: {size_mb:.0f} MB")
    video_volume.commit()
    return output_path


# ── Analyse (GPU pipeline) ───────────────────────────────────────

@app.function(
    volumes={
        "/models": model_volume,
        "/videos": video_volume,
        "/outputs": output_volume,
    },
    image=image,
    gpu="A10G",
    cpu=4,
    memory=32768,
    timeout=3600,  # 1 hour for full pipeline including VLM/Sherlock
    secrets=[
        modal.Secret.from_name("anthropic-key", required_keys=["ANTHROPIC_API_KEY"]),
        modal.Secret.from_name("xai-key", required_keys=["XAI_API_KEY"]),
    ],
)
def analyse(
    game_id: str,
    roster: dict,
    game_start: float = 150.0,
    quarter_duration: int = 600,
    sample_rate: int = 6,
) -> dict:
    """Run the full analysis pipeline on cloud GPU."""
    import sys
    import json
    sys.path.insert(0, "/app")

    from app.pipeline.pipeline_config import PipelineConfig
    from app.pipeline.run_analysis import PipelineOrchestrator

    # Debug: list what's on the video volume
    print(f"Listing /videos/...")
    for item in os.listdir("/videos"):
        print(f"  /videos/{item}")
        subpath = f"/videos/{item}"
        if os.path.isdir(subpath):
            for sub in os.listdir(subpath):
                size_mb = os.path.getsize(f"{subpath}/{sub}") / 1024 / 1024
                print(f"    {sub} ({size_mb:.0f} MB)")

    # Prefer original (guaranteed complete). 1080p may be corrupt from failed transcode.
    video_path = f"/videos/{game_id}/original.mp4"
    if not os.path.exists(video_path):
        video_path = f"/videos/{game_id}/1080p.mp4"
    if not os.path.exists(video_path):
        # Try any mp4 in the game directory
        game_dir = f"/videos/{game_id}"
        if os.path.isdir(game_dir):
            mp4s = [f for f in os.listdir(game_dir) if f.endswith(".mp4") or f.endswith(".MP4")]
            if mp4s:
                video_path = f"{game_dir}/{mp4s[0]}"
                print(f"Using fallback video: {video_path}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"No video found for game {game_id}. Checked /videos/{game_id}/")

    output_dir = f"/outputs/{game_id}"
    os.makedirs(output_dir, exist_ok=True)

    # Write roster to temp location (not output_dir, to avoid SameFileError when pipeline copies it)
    roster_path = f"/tmp/roster_{game_id}.json"
    with open(roster_path, "w") as f:
        json.dump(roster, f)

    # Configure pipeline
    config = PipelineConfig(
        device="cuda",  # Modal A10G provides CUDA
        yolo_model="/models/best.pt",
        class_map={0: "ball", 1: "basket", 2: "person"},
        frame_sample_rate=sample_rate,
        output_dir=output_dir,
        enable_clips=False,
        enable_coaching_agent=False,
        roster_path=roster_path,
        game_start_sec=game_start,
        quarter_duration_sec=quarter_duration,
        vlm_backend="skip",  # Skip VLM in cloud to avoid 60+ min Sherlock timeout
        tracker_type="iou",  # IoU is faster and sufficient with track merger
        use_possession_state_machine=True,
    )

    # Run pipeline
    orchestrator = PipelineOrchestrator(config)
    result = orchestrator.run(video_path)

    output_volume.commit()

    # Return summary
    with open(f"{output_dir}/box_score.json") as f:
        box_score = json.load(f)

    return {
        "game_id": game_id,
        "output_dir": output_dir,
        "home": box_score["home"]["team_name"],
        "away": box_score["away"]["team_name"],
        "total_shots": box_score.get("detection_summary", {}).get("total_shots", 0),
    }


# ── Compile report (CPU) ─────────────────────────────────────────

@app.function(
    volumes={"/outputs": output_volume},
    image=image,
    cpu=2,
    memory=4096,
    timeout=600,
    secrets=[
        modal.Secret.from_name("xai-key", required_keys=["XAI_API_KEY"]),
    ],
)
def compile_report(
    game_id: str,
    match_id: int,
    llm_backend: str = "grok",
) -> dict:
    """Compile game report with API ground truth merge."""
    import sys
    import json
    sys.path.insert(0, "/app")

    output_dir = f"/outputs/{game_id}"

    from scripts.compile_film_report import load_box_score, clean_box_score, merge_api_data, main as compile_main
    from app.analytics.box_score import GameBoxScore
    from app.reporting.film_report import FilmReportGenerator

    # Load and process
    game = load_box_score(output_dir)
    game = clean_box_score(game, output_dir)

    if match_id:
        game = merge_api_data(game, match_id, output_dir)

    # Generate report
    llm_client = None
    if llm_backend == "grok":
        from app.reporting.coach_agent import GrokClient
        llm_client = GrokClient()

    generator = FilmReportGenerator(llm_client=llm_client)
    report = generator.generate(game)

    # Save outputs
    report_dir = f"{output_dir}/report"
    os.makedirs(report_dir, exist_ok=True)

    safe_name = f"{report.home_name} {report.home_score} - {report.away_name} {report.away_score}"

    json_path = f"{report_dir}/{safe_name}.json"
    generator.save_json(report, json_path)

    md_path = f"{report_dir}/{safe_name}.md"
    generator.save_markdown(report, md_path)

    output_volume.commit()

    return {
        "game_id": game_id,
        "score": f"{report.home_name} {report.home_score} - {report.away_name} {report.away_score}",
        "report_path": report_dir,
    }


# ── Full pipeline (local entrypoint) ─────────────────────────────

@app.local_entrypoint()
def full_pipeline(
    video: str,
    roster: str,
    match_id: int = 0,
    game_start: float = 150.0,
    quarter_duration: int = 600,
    sample_rate: int = 6,
    llm_backend: str = "grok",
    skip_upload: bool = False,
    skip_transcode: bool = False,
):
    """Upload video, run pipeline, compile report, download results.

    Usage:
        # Full pipeline
        modal run cloud/modal_app.py::full_pipeline \
            --video game.mp4 --roster roster.json --match-id 8254973

        # Skip upload (video already on volume)
        modal run cloud/modal_app.py::full_pipeline \
            --video game.mp4 --roster roster.json --match-id 8254973 \
            --skip-upload --skip-transcode
    """
    import json
    from pathlib import Path

    video_path = Path(video)
    roster_path = Path(roster)

    if not skip_upload and not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video}")
    if not roster_path.exists():
        raise FileNotFoundError(f"Roster not found: {roster}")

    # Generate game ID from filename
    game_id = video_path.stem

    # Load roster
    with open(roster_path) as f:
        roster_data = json.load(f)

    print(f"\n{'='*60}")
    print(f"HOOPSVISION CLOUD PIPELINE")
    print(f"{'='*60}")
    print(f"\nGame: {game_id}")

    # Step 1: Upload video
    if skip_upload:
        print(f"\n[1/4] Skipping upload (video already on volume)")
    else:
        size_gb = video_path.stat().st_size / 1024 / 1024 / 1024
        print(f"\n[1/4] Uploading video ({size_gb:.1f} GB)...")
        vol = modal.Volume.from_name("hoopsvision-videos")
        with vol.batch_upload() as batch:
            batch.put_file(video_path, f"/{game_id}/original.mp4")
        print(f"  Uploaded.")

    # Step 2: Transcode
    if skip_transcode:
        print(f"\n[2/4] Skipping transcode (using original video)")
    else:
        print(f"\n[2/4] Transcoding to 1080p...")
        transcode_result = transcode.remote(game_id)
        print(f"  Done: {transcode_result}")

    # Step 3: Analyse
    print(f"\n[3/4] Analysing on cloud GPU...")
    analyse_result = analyse.remote(
        game_id=game_id,
        roster=roster_data,
        game_start=game_start,
        quarter_duration=quarter_duration,
        sample_rate=sample_rate,
    )
    print(f"  Shots detected: {analyse_result['total_shots']}")
    print(f"  {analyse_result['home']} vs {analyse_result['away']}")

    # Step 4: Compile report
    if match_id:
        print(f"\n[4/4] Compiling report (API merge + {llm_backend})...")
        report_result = compile_report.remote(
            game_id=game_id,
            match_id=match_id,
            llm_backend=llm_backend,
        )
        print(f"  Score: {report_result['score']}")
    else:
        print(f"\n[4/4] Skipping report (no match-id provided)")
        report_result = None

    # Download results
    print(f"\n{'='*60}")
    print(f"DOWNLOAD RESULTS")
    print(f"{'='*60}")
    local_output = Path(f"data/outputs/{game_id}-cloud")
    local_output.mkdir(parents=True, exist_ok=True)

    out_vol = modal.Volume.from_name("hoopsvision-outputs")
    try:
        for entry in out_vol.iterdir(f"/{game_id}"):
            remote_path = entry.path
            local_path = local_output / Path(remote_path).name
            local_path.parent.mkdir(parents=True, exist_ok=True)
            out_vol.read_file_into_fileobj(remote_path, open(local_path, "wb"))
            print(f"  Downloaded: {local_path.name}")
    except Exception as e:
        print(f"  Download via SDK failed ({e}), try: modal volume get hoopsvision-outputs /{game_id}/ {local_output}")

    print(f"\nResults saved to: {local_output}")
    print(f"Done.")
