"""Structured outputs we ask Gemini for (validated with Pydantic)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Phase = Literal["stance", "load", "foot_plant", "contact", "finish"]
Diagram = Literal["sequencing", "overlay", "scores"]
Joint = Literal["lead_shoulder", "rear_shoulder", "lead_elbow", "rear_elbow", "lead_wrist", "rear_wrist",
                "lead_hip", "rear_hip", "lead_knee", "rear_knee", "lead_ankle", "rear_ankle", "neck"]


class Drill(BaseModel):
    name: str
    steps: list[str] = Field(min_length=1, max_length=5)


class TrainingItem(BaseModel):
    exercise_id: str = Field(description="id copied exactly from exercise_options")
    why: str = Field(description="One sentence linking this exercise to this hitter's measured weakness")
    prescription: str = Field(description="Sets/reps; use the option's default unless there is a clear reason")


class SwingReport(BaseModel):
    headline: str = Field(description="One short sentence naming the main issue or strength")
    what_happened: str = Field(description="2-3 sentences, citing only numbers present in the payload")
    why_it_matters: str = Field(description="1-2 sentences linking the issue to bat speed or contact")
    cue: str = Field(description="One coaching cue the hitter can think about during the next swing")
    drill: Drill
    focus_joint: Joint
    focus_phase: Phase
    diagram: Diagram
    caption: str = Field(description="Caption for the chosen diagram")
    training_focus: str = Field(default="", description="One sentence: what the training plan targets and why")
    training_plan: list[TrainingItem] = Field(default_factory=list, max_length=5,
                                              description="3-4 exercises chosen ONLY from exercise_options")
    video_notes: list[str] = Field(default_factory=list, max_length=4,
                                   description="What the video shows about the focus factor that the numbers don't (empty without a video)")
    disagreements: list[str] = Field(default_factory=list, max_length=3,
                                     description="Where the video clearly disagrees with a measured factor; empty if none")


class BlindRating(BaseModel):
    """The AI coach's own look at the swing video, BEFORE it sees any numbers (Explosiveness factor)."""
    rating: int = Field(ge=0, le=100, description="Overall athleticism and explosiveness, 0-100; 100 = like an elite college/pro hitter")
    upper_body: str = Field(description="One sentence on the shoulder and hip turn as seen")
    legs: str = Field(description="One sentence on the stride, plant, front-leg brace and back-foot turn as seen")
    standout: str = Field(description="The one thing that stands out most, good or bad")


class VideoRead(BaseModel):
    """Qualitative second opinion from the swing video (multimodal)."""
    observations: list[str] = Field(description="Things a skeleton cannot see: grip, bat angle/lag, head/eyes, finish balance")
    contact_time_s: float | None = Field(description="Estimated time of contact in the clip, seconds")
    confidence: Literal["low", "medium", "high"]


class SessionReport(BaseModel):
    headline: str = Field(description="One sentence summing up the session")
    trend: str = Field(description="How bat speed and scores changed from the first to the last swings, with numbers from the payload")
    consistency: str = Field(description="Which mechanics were repeatable and which varied, with numbers")
    recurring_issue: str = Field(description="The issue that showed up most often and how many swings")
    best_swing_id: str
    next_session_focus: str = Field(description="One thing to work on next time")
    fatigue_note: str = Field(default="", description="Only if bat speed dropped late in the session")
