"""AI coaching agent using LangGraph and Google Gemini."""

import os
from pathlib import Path
from typing import Protocol, TypedDict

from app.prompts.loader import format_prompt, load_prompts


class BaseLLMClient(Protocol):
    """Protocol for LLM backends."""
    def generate(self, prompt: str, system: str = "") -> str: ...


class GeminiClient:
    """Google Gemini LLM client via langchain."""

    def __init__(self, model: str = "gemini-2.0-flash"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY not set. Set it in .env or environment."
            )
        self.llm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key)

    def generate(self, prompt: str, system: str = "") -> str:
        from langchain_core.messages import HumanMessage, SystemMessage
        messages = []
        if system:
            messages.append(SystemMessage(content=system))
        messages.append(HumanMessage(content=prompt))
        response = self.llm.invoke(messages)
        return response.content


class GrokClient:
    """xAI Grok LLM client via OpenAI-compatible API."""

    def __init__(self, model: str = "grok-3-mini-fast"):
        from openai import OpenAI

        api_key = os.environ.get("XAI_API_KEY")
        if not api_key:
            raise ValueError(
                "XAI_API_KEY not set. Set it in .env or environment."
            )
        self.client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        self.model = model

    def generate(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            timeout=30,
        )
        return response.choices[0].message.content


class TemplateLLMClient:
    """Fallback report generator using string templates. No LLM needed."""

    def generate(self, prompt: str, system: str = "") -> str:
        return prompt  # Pass-through; the agent nodes handle formatting


class CoachingState(TypedDict):
    game_summary: dict
    tracking_data: dict
    offense_analysis: str
    defense_analysis: str
    technique_analysis: str
    spacing_analysis: str
    transition_analysis: str
    court_vision_analysis: str
    individual_analysis: str
    coaching_report: str


def build_coaching_graph(llm_client: BaseLLMClient):
    """Build LangGraph coaching agent with parallel specialist analyses."""
    from langgraph.graph import END, StateGraph

    prompts = load_prompts("coaching")

    def _tracking_data(state: CoachingState) -> str:
        """Get tracking data string, defaulting to empty note."""
        td = state.get("tracking_data", {})
        return td if td else "No tracking data available for this game."

    def analyze_offense(state: CoachingState) -> dict:
        cfg = prompts["analyze_offense"]
        prompt = format_prompt(
            cfg["user"],
            stats=state["game_summary"],
            tracking_data=_tracking_data(state),
        )
        result = llm_client.generate(prompt, system=cfg["system"])
        return {"offense_analysis": result}

    def analyze_defense(state: CoachingState) -> dict:
        cfg = prompts["analyze_defense"]
        prompt = format_prompt(
            cfg["user"],
            stats=state["game_summary"],
            tracking_data=_tracking_data(state),
        )
        result = llm_client.generate(prompt, system=cfg["system"])
        return {"defense_analysis": result}

    def analyze_technique(state: CoachingState) -> dict:
        cfg = prompts["analyze_technique"]
        prompt = format_prompt(
            cfg["user"],
            stats=state["game_summary"],
            tracking_data=_tracking_data(state),
        )
        result = llm_client.generate(prompt, system=cfg["system"])
        return {"technique_analysis": result}

    def analyze_spacing(state: CoachingState) -> dict:
        cfg = prompts["analyze_spacing"]
        prompt = format_prompt(
            cfg["user"],
            stats=state["game_summary"],
            tracking_data=_tracking_data(state),
        )
        result = llm_client.generate(prompt, system=cfg["system"])
        return {"spacing_analysis": result}

    def analyze_transition(state: CoachingState) -> dict:
        cfg = prompts["analyze_transition"]
        prompt = format_prompt(
            cfg["user"],
            stats=state["game_summary"],
            tracking_data=_tracking_data(state),
        )
        result = llm_client.generate(prompt, system=cfg["system"])
        return {"transition_analysis": result}

    def analyze_court_vision(state: CoachingState) -> dict:
        cfg = prompts["analyze_court_vision"]
        prompt = format_prompt(
            cfg["user"],
            stats=state["game_summary"],
            tracking_data=_tracking_data(state),
        )
        result = llm_client.generate(prompt, system=cfg["system"])
        return {"court_vision_analysis": result}

    def analyze_individual(state: CoachingState) -> dict:
        cfg = prompts["analyze_individual"]
        prompt = format_prompt(
            cfg["user"],
            stats=state["game_summary"],
            tracking_data=_tracking_data(state),
        )
        result = llm_client.generate(prompt, system=cfg["system"])
        return {"individual_analysis": result}

    def synthesize_report(state: CoachingState) -> dict:
        cfg = prompts["synthesize_report"]
        prompt = format_prompt(
            cfg["user"],
            stats=state["game_summary"],
            offense_analysis=state.get("offense_analysis", ""),
            defense_analysis=state.get("defense_analysis", ""),
            technique_analysis=state.get("technique_analysis", ""),
            spacing_analysis=state.get("spacing_analysis", ""),
            transition_analysis=state.get("transition_analysis", ""),
            court_vision_analysis=state.get("court_vision_analysis", ""),
            individual_analysis=state.get("individual_analysis", ""),
        )
        report = llm_client.generate(prompt, system=cfg["system"])
        return {"coaching_report": report}

    graph = StateGraph(CoachingState)

    # Add all analysis nodes
    graph.add_node("analyze_offense", analyze_offense)
    graph.add_node("analyze_defense", analyze_defense)
    graph.add_node("analyze_technique", analyze_technique)
    graph.add_node("analyze_spacing", analyze_spacing)
    graph.add_node("analyze_transition", analyze_transition)
    graph.add_node("analyze_court_vision", analyze_court_vision)
    graph.add_node("analyze_individual", analyze_individual)
    graph.add_node("synthesize_report", synthesize_report)

    # Parallel fan-out: all specialist analyses run concurrently from entry
    # then fan-in to synthesis
    parallel_nodes = [
        "analyze_offense",
        "analyze_defense",
        "analyze_technique",
        "analyze_spacing",
        "analyze_transition",
        "analyze_court_vision",
        "analyze_individual",
    ]

    # Entry fans out to all parallel nodes
    graph.set_entry_point("analyze_offense")
    # Wire sequential chain: offense → defense → technique → spacing →
    # transition → court_vision → individual → synthesize
    # (LangGraph StateGraph doesn't support true fan-out without
    # conditional edges, so we chain them — each is independent and
    # writes to its own state key, so order doesn't matter for correctness)
    graph.add_edge("analyze_offense", "analyze_defense")
    graph.add_edge("analyze_defense", "analyze_technique")
    graph.add_edge("analyze_technique", "analyze_spacing")
    graph.add_edge("analyze_spacing", "analyze_transition")
    graph.add_edge("analyze_transition", "analyze_court_vision")
    graph.add_edge("analyze_court_vision", "analyze_individual")
    graph.add_edge("analyze_individual", "synthesize_report")
    graph.add_edge("synthesize_report", END)

    return graph.compile()


def generate_template_report(summary: dict) -> str:
    """Generate a coaching report using templates (no LLM needed)."""
    fg_pct = summary.get("fg_percentage", 0) * 100
    total_shots = summary.get("total_shots", 0)
    shots_made = summary.get("shots_made", 0)
    possessions = summary.get("total_possessions", 0)
    turnovers = summary.get("turnovers", 0)
    avg_poss = summary.get("avg_possession_duration", 0)
    players = summary.get("player_stats", [])

    player_lines = ""
    for p in players:
        player_lines += (
            f"| Player {p['player_id']} | {p['shots_attempted']} | "
            f"{p['shots_made']} | {p['fg_percentage']*100:.1f}% | "
            f"{p['possessions']} | {p['possession_time_sec']:.1f}s |\n"
        )

    return f"""# Game Analysis Report

## Executive Summary

The team attempted **{total_shots} shots** with **{shots_made} made** for a field goal percentage of **{fg_pct:.1f}%**.
There were **{possessions} possessions** tracked with an average duration of **{avg_poss:.1f} seconds**.
**{turnovers} turnovers** were recorded.

## Shooting Analysis

- Total Shots: {total_shots}
- Shots Made: {shots_made}
- Field Goal %: {fg_pct:.1f}%

## Possession Analysis

- Total Possessions: {possessions}
- Average Duration: {avg_poss:.1f}s
- Turnovers: {turnovers}

## Player Statistics

| Player | FGA | FGM | FG% | Possessions | Possession Time |
|--------|-----|-----|-----|-------------|-----------------|
{player_lines}

## Recommendations

1. {"Improve shot selection - FG% below 40%" if fg_pct < 40 else "Maintain shooting efficiency"}
2. {"Reduce turnovers - high turnover rate detected" if turnovers > possessions * 0.3 else "Continue ball security"}
3. {"Increase pace - short possession durations" if avg_poss < 10 else "Manage clock effectively"}

---
*Generated by Basketball AI Analysis Platform*
"""


def run_coaching_agent(
    summary: dict,
    output_path: str,
    llm_backend: str = "gemini",
    tracking_data: dict | None = None,
) -> str:
    """Run coaching agent and write report to file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if llm_backend in ("gemini", "grok"):
        try:
            client = GrokClient() if llm_backend == "grok" else GeminiClient()
            graph = build_coaching_graph(client)
            result = graph.invoke({
                "game_summary": summary,
                "tracking_data": tracking_data or {},
                "offense_analysis": "",
                "defense_analysis": "",
                "technique_analysis": "",
                "spacing_analysis": "",
                "transition_analysis": "",
                "court_vision_analysis": "",
                "individual_analysis": "",
                "coaching_report": "",
            })
            report = result["coaching_report"]
        except Exception as e:
            print(f"{llm_backend.title()} agent failed ({e}), falling back to template report")
            report = generate_template_report(summary)
    else:
        report = generate_template_report(summary)

    Path(output_path).write_text(report)
    return output_path
