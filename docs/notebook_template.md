# Notebook charter template (ML course)

Every notebook teaches ONE concept (in the fundamentals 1–3) or integrates several (the parameters
notebook 4 and the demanding case 5). Plain `.ipynb`, **output-free in git**, ~20 cells.

**Voice:** warm, empowering, celebratory — AND rigorous. No decorative emojis. No "obviously /
simply / trivially / just". Celebrate what the learner just achieved; frame difficulty as growth.
**Never presuppose prior knowledge** — re-lay what the notebook leans on. **Everything in the
notebook is in English** — prose, code, identifiers, docstrings.

## Cell order (target ~20 cells)

1. **Header** (markdown): `# NN — Title`; one-line purpose; **Prerequisites** (notebook ids);
   **What you'll be able to do** (3–5 action-verb objectives). A warm one-line welcome is encouraged.
2. **Imports** (code): stdlib / third-party / `ml_course` local. `np.random.seed(...)` (fixed,
   documented); `from ml_course import viz; viz.use_course_style()`. **Never** hardcode hex — use
   `ml_course.colors`.
3. **Recap / footing** (markdown): briefly re-establish the prerequisite ideas this notebook stands
   on. Short, but real — not "you already know X".
4–15. **Body sections** (one concept per section), each:
   - *Intuition* (markdown) — the picture, before any formula.
   - *Implementation* (code) — first **by hand** (the mechanism), then the library estimator once
     the mechanism is felt. Small, visualizable data in the fundamentals.
   - **"Read the figure"** (markdown) — always explain what we see, kindly and concretely. Every
     figure cell is followed by one of these.
16. **Your turn** (markdown + optional scaffold code): 2–3 exercises, tiered easy → harder.
   These belong to the learner — the notebook does not solve them.
17. **What you built** (markdown): celebrate the accomplishment in concrete bullets + restate what
   the learner can now do.
18. **Going further** (markdown, optional): the formal/deeper view for the curious — clearly marked
   optional, never a prerequisite for the next notebook.
19. **References** (markdown): cited papers with DOI; `Previous:` / `Next:` links in the module.

(The four body slots 4–15 stretch or compress to land near twenty cells total.)

## The four notebook roles

- **Fundamentals (1–3):** one concept each, maximum by-hand, maximum visual. The learner could
  explain the idea afterwards without help.
- **The method & its parameters (4):** the real estimator. Walk each hyperparameter: what it does,
  what too-much/too-little looks like, how to choose it (held-out data / CV). Show failure modes.
- **The demanding case (5):** a realistic problem. Full honest workflow — look at the data →
  preprocess → choose → evaluate on held-out data → analyse errors → state limits.

## Figures

Use `ml_course.viz` helpers and `ml_course.colors`. Fixed dims/DPI from the charter
(`viz.use_course_style()`). Every figure is followed by a "Read the figure" paragraph.

## Before commit

- Clear all outputs.
- `uv run python scripts/check_no_hardcoded_hex.py` passes.
- `uv run python scripts/gen_llms_txt.py` re-run if a notebook was added/renamed.
- `@ml-expert-reviewer` and `@pedagogy-reviewer` both PASS (no BLOCK).
