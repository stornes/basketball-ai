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


class TemplateLLMClient:
    """Fallback report generator using string templates. No LLM needed."""

    def generate(self, prompt: str, system: str = "") -> str:
        return prompt  # Pass-through; the agent nodes handle formatting


class CoachingState(TypedDict):
    game_summary: dict
    offense_analysis: str
    defense_analysis: str
    coaching_report: str


def build_coaching_graph(llm_client: BaseLLMClient):
    """Build LangGraph coaching agent."""
    from langgraph.graph import END, StateGraph

    prompts = load_prompts("coaching")

    def analyze_offense(state: CoachingState) -> dict:
        cfg = prompts["analyze_offense"]
        prompt = format_prompt(cfg["user"], stats=state["game_summary"])
        result = llm_client.generate(prompt, system=cfg["system"])
        return {"offense_analysis": result}

    def analyze_defense(state: CoachingState) -> dict:
        cfg = prompts["analyze_defense"]
        prompt = format_prompt(cfg["user"], stats=state["game_summary"])
        result = llm_client.generate(prompt, system=cfg["system"])
        return {"defense_analysis": result}

    def synthesize_report(state: CoachingState) -> dict:
        cfg = prompts["synthesize_report"]
        prompt = format_prompt(
            cfg["user"],
            stats=state["game_summary"],
            offense_analysis=state.get("offense_analysis", ""),
            defense_analysis=state.get("defense_analysis", ""),
        )
        report = llm_client.generate(prompt, system=cfg["system"])
        return {"coaching_report": report}

    graph = StateGraph(CoachingState)
    graph.add_node("analyze_offense", analyze_offense)
    graph.add_node("analyze_defense", analyze_defense)
    graph.add_node("synthesize_report", synthesize_report)

    graph.set_entry_point("analyze_offense")
    graph.add_edge("analyze_offense", "analyze_defense")
    graph.add_edge("analyze_defense", "synthesize_report")
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
) -> str:
    """Run coaching agent and write report to file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if llm_backend == "gemini":
        try:
            client = GeminiClient()
            graph = build_coaching_graph(client)
            result = graph.invoke({
                "game_summary": summary,
                "offense_analysis": "",
                "defense_analysis": "",
                "coaching_report": "",
            })
            report = result["coaching_report"]
        except Exception as e:
            print(f"Gemini agent failed ({e}), falling back to template report")
            report = generate_template_report(summary)
    else:
        report = generate_template_report(summary)

    Path(output_path).write_text(report)
    return output_path
