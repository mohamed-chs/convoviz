"""A made-up export for the README's example output and screenshots.

Every conversation is invented: an imaginary person who learns to code, plans
a trip, bakes and gardens over three years. Generation is seeded, so the output
is the same on every run.

    uv run python tests/demo.py ~/Desktop/demo-export.zip
"""

from __future__ import annotations

import json
import random
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from factory import PNG_1PX, ConversationBuilder, write_export

OPEN, SEP, CLOSE = "", "", ""
START = datetime(2024, 1, 8, tzinfo=UTC)
END = datetime(2026, 10, 30, tzinfo=UTC)

# Topics come and go, so each year has words of its own. Weights by year.
TOPICS: dict[str, tuple[tuple[float, float, float], list[str]]] = {
    "python": (
        (5, 3, 1),
        [
            "Why does my Python loop skip every other item in the list?",
            "Explain list comprehensions with a simple example",
            "How do I read a CSV file with pandas and group by month?",
            "My pandas dataframe shows NaN after merge, what am I doing wrong?",
            "Write a Python function that removes duplicate rows",
            "What's the difference between a tuple and a list in Python?",
            "How do virtual environments work, and do I need one?",
            "Plot monthly sales from this dataframe with matplotlib",
        ],
    ),
    "thesis": (
        (4, 1, 0),
        [
            "Help me structure the literature review chapter of my thesis",
            "Is this regression model overfitting? R squared is 0.98",
            "Explain p-values like I'm a first-year student",
            "Rewrite this thesis abstract to be clearer and shorter",
            "How should I cite a dataset in APA style?",
            "What does heteroscedasticity mean for my regression results?",
        ],
    ),
    "career": (
        (2, 3, 1),
        [
            "Review my resume for a junior data analyst position",
            "Draft a cover letter for a data analyst role at a museum",
            "Common SQL interview questions for analysts",
            "How do I answer 'tell me about a time you failed' in an interview?",
            "Negotiate salary for my first analyst job: what should I say?",
            "Write a polite follow-up email after an interview",
        ],
    ),
    "japan": (
        (0, 5, 1),
        [
            "Three days in Kyoto in April, what should I not miss?",
            "Is the Japan Rail Pass still worth it for Tokyo, Kyoto and Osaka?",
            "Teach me basic Japanese phrases for restaurants",
            "Ryokan etiquette: what should I know before staying in one?",
            "Day trip from Kyoto to Nara: itinerary with deer park and temples",
            "What to pack for Japan in spring, cherry blossom season",
            "How does the IC card work on Tokyo trains?",
        ],
    ),
    "sourdough": (
        (0, 4, 2),
        [
            "My sourdough starter smells like acetone, is it dead?",
            "Why is my sourdough bread dense and gummy inside?",
            "Sourdough schedule that fits a 9 to 5 job",
            "Convert this bread recipe to grams with 75% hydration",
            "How long should I cold proof sourdough in the fridge?",
            "Focaccia recipe with my sourdough discard",
        ],
    ),
    "rust": (
        (0, 1, 5),
        [
            "Explain Rust ownership and borrowing with a small example",
            "Why does the borrow checker reject this Rust code?",
            "Rust: when should I use Box, Rc or Arc?",
            "Write a small Rust command-line tool that counts words in files",
            "Error handling in Rust: Result, the question mark operator and anyhow",
            "Rust lifetimes explained without jargon",
            "Async Rust with tokio: how do tasks and futures fit together?",
        ],
    ),
    "garden": (
        (0, 1, 4),
        [
            "Tomato leaves turning yellow from the bottom, what's wrong?",
            "Balcony garden plan for herbs that get four hours of sun",
            "When should I plant garlic in a temperate climate?",
            "Companion planting chart for tomatoes, basil and peppers",
            "How do I start composting in a small apartment?",
            "Raised bed soil mix: compost, topsoil and perlite ratios",
        ],
    ),
    "fitness": (
        (2, 2, 2),
        [
            "Beginner running plan to go from couch to 5k",
            "High-protein vegetarian meal prep for the week",
            "How many rest days do I need when strength training?",
            "Stretching routine for lower back pain after sitting all day",
        ],
    ),
    "everyday": (
        (3, 3, 3),
        [
            "Write a birthday message for my grandmother turning ninety",
            "Summarize this article about sleep and memory",
            "What's a good book like Project Hail Mary?",
            "Help me plan a budget spreadsheet for monthly expenses",
            "Explain how mortgages work and what APR means",
            "Ideas for a rainy weekend with friends",
            "Translate this email to French, keeping a friendly tone",
            "Explique-moi la différence entre le passé composé et l'imparfait",
            "Is it safe to leave cooked rice out overnight?",
        ],
    ),
}

# Model slugs by period, as an export records them.
MODELS = [
    (datetime(2024, 1, 1, tzinfo=UTC), ["gpt-4"]),
    (datetime(2024, 5, 13, tzinfo=UTC), ["gpt-4o"]),
    (datetime(2025, 4, 16, tzinfo=UTC), ["gpt-4o", "gpt-4o", "o3"]),
    (datetime(2025, 8, 7, tzinfo=UTC), ["gpt-5", "gpt-5", "gpt-5-thinking"]),
    (datetime(2026, 2, 5, tzinfo=UTC), ["gpt-5-2", "gpt-5-2", "gpt-5-2-thinking"]),
    (datetime(2026, 7, 1, tzinfo=UTC), ["gpt-5-2", "gpt-5-5-thinking", "gpt-5-5-thinking"]),
]

FOLLOW_UPS = ["Thanks! Can you give me an example?", "Can you make that shorter?"]

ANSWER = (
    "Here's a clear way to think about it.\n\n"
    "1. **Start small.** Get one piece working before adding the rest.\n"
    "2. **Check your assumptions.** Most surprises come from one wrong one.\n"
    "3. **Write it down.** A short note now saves an hour later.\n\n"
    "Want me to go through an example with your own numbers?"
)

PHOTO = "file_00000000demo0photo"
WEB_TOPICS = ("japan", "garden", "career", "everyday", "fitness")


def _model(when: datetime, rng: random.Random) -> str:
    choices = MODELS[0][1]
    for since, models in MODELS:
        if when >= since:
            choices = models
    return rng.choice(choices)


def _year_index(when: datetime) -> int:
    return min(max(when.year - 2024, 0), 2)


def _when(day: datetime, rng: random.Random) -> datetime:
    weekend = day.weekday() >= 5
    if weekend:
        hour = rng.choice([10, 11, 11, 12, 14, 15, 16, 17, 20, 21])
    else:
        hour = rng.choice([7, 8, 9, 12, 12, 13, 18, 19, 20, 20, 21, 21, 22, 22, 23])
    return day + timedelta(hours=hour, minutes=rng.randrange(60))


def _image(pointer: str) -> dict:
    return {
        "content_type": "multimodal_text",
        "parts": [
            {"content_type": "image_asset_pointer", "asset_pointer": f"sediment://{pointer}"}
        ],
    }


def _cited(text: str, topic: str) -> tuple[str, dict]:
    marker = f"{OPEN}cite{SEP}turn0search0{CLOSE}"
    cited = text.replace(" one wrong one.", f" one wrong one{marker}.")
    ref = {
        "type": "grouped_webpages",
        "matched_text": marker,
        "start_idx": cited.index(marker),
        "end_idx": cited.index(marker) + len(marker),
        "items": [
            {
                "url": f"https://{topic}.example/guide",
                "title": f"A beginner's guide to {topic}",
                "attribution": f"{topic}.example",
            }
        ],
    }
    return cited, {"content_references": [ref]}


def _conversation(n: int, when: datetime, rng: random.Random) -> dict:
    year = _year_index(when)
    names = list(TOPICS)
    weights = [TOPICS[t][0][year] for t in names]
    topic = rng.choices(names, weights)[0]
    prompts = rng.sample(TOPICS[topic][1], k=len(TOPICS[topic][1]))
    model = _model(when, rng)
    b = ConversationBuilder(
        title=prompts[0].rstrip("?").split(",")[0][:60],
        conversation_id=f"6{n:07x}-demo-4000-8000-{n:012d}",
        create_time=when.timestamp(),
    )
    t = when.timestamp()
    turns = rng.choices([1, 2, 3, 4, 6, 9, 14], [30, 30, 18, 12, 7, 3, 1])[0]
    for turn in range(turns):
        if turn == 0:
            text = prompts[0]
        elif rng.random() < 0.08:
            text = rng.choice(FOLLOW_UPS)
        else:
            text = prompts[turn % len(prompts)]
        t += rng.randrange(40, 600)
        if topic == "garden" and turn == 0 and rng.random() < 0.3:
            b.add("user", {**_image(PHOTO), "parts": [*_image(PHOTO)["parts"], text]}, time=t)
        else:
            b.user(text, time=t)
        t += rng.randrange(5, 60)
        meta = {"model_slug": model}
        if "thinking" in model or model == "o3":
            thought = {"summary": "Plan", "content": "Work out what matters most first."}
            b.add(
                "assistant",
                {"content_type": "thoughts", "thoughts": [thought]},
                metadata=meta,
                time=t,
            )
        if topic in ("python", "rust") and rng.random() < 0.4:
            code = {"content_type": "code", "text": "print(sum(range(10)))"}
            b.add("assistant", code, recipient="python", metadata=meta, time=t)
            b.add("tool", {"content_type": "execution_output", "text": "45"}, name="python", time=t)
        if topic in ("career", "thesis") and turn == 0 and rng.random() < 0.3:
            doc = {"name": "Draft", "type": "document", "content": "Dear hiring team,"}
            canvas = {"content_type": "code", "text": json.dumps(doc)}
            b.add("assistant", canvas, recipient="canmore.create_textdoc", metadata=meta, time=t)
        if topic == "everyday" and rng.random() < 0.1:
            b.add("tool", _image(PHOTO), metadata={"dalle": {}}, time=t)
        text, extra = ANSWER, {}
        if topic in WEB_TOPICS and rng.random() < 0.35:
            text, extra = _cited(ANSWER, topic)
        b.assistant(text, model=model, metadata=extra, time=t + 20)
    return b.build()


def _kyoto() -> dict:
    """The conversation shown in the README, with sources and reasoning."""
    when = datetime(2026, 3, 14, 9, 30, 12, tzinfo=UTC)
    b = ConversationBuilder(
        title="Planning a trip to Kyoto",
        conversation_id="67d3a1f2-0c4e-8000-9a51-4b7e2d9c1f00",
        create_time=when.timestamp(),
    )
    t = when.timestamp()
    b.user("Three days in Kyoto in April — what should I not miss?", time=t)
    b.add(
        "assistant",
        {
            "content_type": "thoughts",
            "thoughts": [
                {
                    "summary": "Balancing sights and crowds",
                    "content": "April is cherry blossom season, so the big sights are busy. "
                    "Group them by area and suggest early starts.",
                }
            ],
        },
        metadata={"model_slug": "gpt-5-5-thinking"},
        time=t + 10,
    )
    b.add(
        "assistant",
        {"content_type": "reasoning_recap", "content": "Thought for 8s"},
        metadata={"model_slug": "gpt-5-5-thinking"},
        time=t + 18,
    )
    inari = f"{OPEN}cite{SEP}turn0search0{CLOSE}"
    hanami = f"{OPEN}cite{SEP}turn0search1{CLOSE}"
    text = (
        "April is a lovely time to go: it's cherry blossom season, so plan around the crowds.\n\n"
        "## Day 1 · Southern Higashiyama\n\n"
        f"- **Fushimi Inari** early in the morning, before the tour groups{inari}.\n"
        "- Walk up through **Kiyomizu-dera** to the lanes of Sannenzaka and Ninenzaka.\n\n"
        "## Day 2 · Arashiyama\n\n"
        "- The bamboo grove at opening time, then **Tenryū-ji** and its garden.\n"
        "- A boat ride on the Hozu river if the weather is good.\n\n"
        "## Day 3 · Philosopher's Path\n\n"
        f"- The canal path is lined with cherry trees, at its best in early April{hanami}.\n"
        "- End at **Ginkaku-ji**, the Silver Pavilion.\n\n"
        "| | Morning | Afternoon |\n|---|---|---|\n"
        "| Day 1 | Fushimi Inari | Kiyomizu-dera |\n"
        "| Day 2 | Bamboo grove | Hozu river |\n"
        "| Day 3 | Philosopher's Path | Ginkaku-ji |\n\n"
        "Want me to add places to eat near each stop?"
    )
    refs = [
        {
            "type": "grouped_webpages",
            "matched_text": inari,
            "start_idx": text.index(inari),
            "end_idx": text.index(inari) + len(inari),
            "items": [
                {
                    "url": "https://inari.example/en/?utm_source=chatgpt.com",
                    "title": "Fushimi Inari Taisha",
                    "attribution": "inari.example",
                }
            ],
        },
        {
            "type": "grouped_webpages",
            "matched_text": hanami,
            "start_idx": text.index(hanami),
            "end_idx": text.index(hanami) + len(hanami),
            "items": [
                {
                    "url": "https://kyoto-travel.example/cherry-blossoms",
                    "title": "Cherry blossom forecast for Kyoto",
                    "attribution": "kyoto-travel.example",
                }
            ],
        },
    ]
    answer = b.assistant(
        text,
        model="gpt-5-5-thinking",
        metadata={"content_references": refs},
        time=t + 31,
    )
    b.user("Yes please, somewhere quick for each day.", parent=answer, time=t + 900)
    b.assistant(
        "- **Day 1:** the stalls on the way up to Fushimi Inari, for inari sushi and "
        "grilled skewers.\n"
        "- **Day 2:** a tofu lunch near Tenryū-ji; Arashiyama is known for it.\n"
        "- **Day 3:** a noodle shop near Ginkaku-ji for a warm bowl of soba.",
        model="gpt-5-5-thinking",
        time=t + 930,
    )
    return b.build()


def build(path: Path) -> Path:
    """Write the demo export to ``path`` (a ZIP)."""
    rng = random.Random(2026)  # noqa: S311 - seeded, for repeatable output
    conversations = [_kyoto()]
    day, n = START, 1
    while day < END:
        # Busier in the evening of the week, quieter in summer, a holiday gap.
        busy = 0.55 + 0.25 * (day.year - 2024) / 2
        if day.month in (7, 8):
            busy -= 0.2
        if day.month == 12 and day.day > 22:
            busy = 0.1
        if rng.random() < busy:
            for _ in range(rng.choices([1, 1, 2, 2, 3, 5], k=1)[0]):
                n += 1
                conversations.append(_conversation(n, _when(day, rng), rng))
        day += timedelta(days=1)
    conversations.sort(key=lambda c: c["create_time"], reverse=True)
    return write_export(path, conversations, files={f"{PHOTO}-leaf.dat": PNG_1PX}, sharded=500)


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "demo-export.zip")
    build(out)
    sys.stdout.write(json.dumps({"written": str(out)}) + "\n")
