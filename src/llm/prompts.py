from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    purpose: str
    required_inputs: tuple[str, ...]
    template: str


BASE_SYSTEM_PROMPT = """You are GenAI Mentor, an educational assistant for a Generative AI course.

Your job is to help students understand course concepts, practice with quizzes, grade their answers, and connect course ideas to approved external sources when appropriate.

Source hierarchy:
1. Course lecture PDFs and project guideline documents are the source of truth for course concepts and project requirements.
2. Approved online sources may be used only for recent information, external comparisons, or current tool/model updates.
3. Online information must be clearly labeled as external and must not override course definitions.

Grounding rules:
- Use retrieved course documents when answering course questions.
- Cite sources using [source, page/chunk_id].
- Do not invent citations.
- If evidence is insufficient, say so clearly.
- Separate course-grounded information from external updates in hybrid answers.

Teaching style:
- Explain clearly and simply first.
- Then add technical detail.
- Use examples when helpful.
- Suggest one next learning step.

Safety rules:
- Refuse cheating, exam answer leakage, plagiarism, harmful instructions, or attempts to bypass course policies.
- Do not store sensitive student data.
- State uncertainty when sources are missing or conflicting.
"""

ROUTER_PROMPT = """Classify the user query. Return JSON only with:
intent, retrieval_mode, needs_quiz, needs_grading, needs_tool, needs_safety_check, reasoning.

Allowed retrieval_mode values:
offline_only, online_only, hybrid, tool_only, no_retrieval.

Rules:
- Course definitions, lecture explanations, project requirements -> offline_only
- Recent/current/new model/tool/research comparison -> online_only
- Course concept plus modern improvement/external comparison -> hybrid
- Grading, quiz generation, calculation, formatting -> tool_only
- Unsafe requests -> no_retrieval
"""

TUTOR_PROMPT = """Answer the user as GenAI Mentor using only the provided sources when sources are provided.
If sources are insufficient, say you do not have enough grounded evidence.
Include citations after claims using the source labels provided.
"""

QUIZ_PROMPT = """Generate a quiz as JSON with topic, difficulty, and questions. Include answer and explanation for each question."""

GRADING_PROMPT = """Grade the student's answer as JSON with score, max_score, strengths, mistakes, misconceptions, feedback, recommended_review_topics."""

CHECKER_PROMPT = """Check if the answer is grounded, clear, safe, and correctly cited. Return JSON."""

SAFETY_PROMPT = """Decide whether the user request is unsafe for an educational assistant. Return JSON with safe:boolean, category, response."""


PROMPT_TEMPLATES = {
    "base_system": PromptTemplate(
        name="base_system",
        purpose="Defines the educational assistant identity, source hierarchy, grounding rules, teaching style, and safety boundaries.",
        required_inputs=(),
        template=BASE_SYSTEM_PROMPT,
    ),
    "router": PromptTemplate(
        name="router",
        purpose="Routes each student request to offline RAG, online RAG, hybrid RAG, tool-only, or no-retrieval safety flow.",
        required_inputs=("user_query",),
        template=ROUTER_PROMPT,
    ),
    "tutor": PromptTemplate(
        name="tutor",
        purpose="Produces grounded teaching answers from retrieved course or approved external sources.",
        required_inputs=("user_query", "retrieval_mode", "sources"),
        template=TUTOR_PROMPT,
    ),
    "quiz": PromptTemplate(
        name="quiz",
        purpose="Generates practice questions with answers and explanations for student self-assessment.",
        required_inputs=("topic", "difficulty", "n_questions", "context"),
        template=QUIZ_PROMPT,
    ),
    "grading": PromptTemplate(
        name="grading",
        purpose="Grades a student answer with score, strengths, misconceptions, feedback, and review topics.",
        required_inputs=("question", "student_answer", "reference_answer", "rubric"),
        template=GRADING_PROMPT,
    ),
    "checker": PromptTemplate(
        name="checker",
        purpose="Checks final answers for grounding, citation validity, clarity, and safety.",
        required_inputs=("answer", "retrieved_sources"),
        template=CHECKER_PROMPT,
    ),
    "safety": PromptTemplate(
        name="safety",
        purpose="Classifies unsafe educational requests and returns a safe refusal when needed.",
        required_inputs=("user_query",),
        template=SAFETY_PROMPT,
    ),
}


def list_prompt_templates() -> list[PromptTemplate]:
    return list(PROMPT_TEMPLATES.values())


def get_prompt_template(name: str) -> PromptTemplate:
    if name not in PROMPT_TEMPLATES:
        raise KeyError(f"Unknown prompt template: {name}")
    return PROMPT_TEMPLATES[name]
