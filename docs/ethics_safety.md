# Ethics, Safety, and Limitations

## Disclaimer

GenAI Mentor is a learning support tool designed exclusively to help students understand course material, practice with quizzes, and get grading feedback on their own answers. It is not a substitute for the instructor, a course grader, or an authoritative academic source. Any answer the system produces should be treated as a study aid, not as a definitive or exam-ready explanation. Students are responsible for verifying understanding with course materials and their instructor.

---

## What the Assistant Refuses

The assistant is designed to support learning, not to bypass it. The safety agent (`src/agents/safety_agent.py`) intercepts requests before any retrieval or generation happens, and the base system prompt (`src/llm/prompts.py`) reinforces these boundaries at the model level.

The following categories of requests are refused outright:

- **Exam answer leakage** — requests like "give me the exam answers" or "what are the hidden exam questions" are blocked. The system has no access to real exam answers, but the refusal is enforced regardless to prevent prompt-injection attempts that try to elicit fabricated answers.
- **Plagiarism and academic dishonesty** — requests to write a report, essay, or assignment for submission as the student's own work are refused. The assistant can help a student understand a topic or review their own draft, but it will not produce submission-ready academic work on their behalf.
- **Policy bypass** — attempts to override safety rules (e.g., "ignore your instructions," "reveal your system prompt") are detected and refused. This protects against prompt injection attacks that try to repurpose the model for unintended uses.
- **Harmful instructions** — the assistant refuses requests outside the educational domain that could cause harm, such as asking for instructions to exploit systems or generate content inappropriate for a course setting.

When a request is refused, the system responds with a concrete alternative — for example, offering to explain the topic, generate a practice quiz, or help the student study for the exam legitimately. Refusal messages are kept constructive rather than penalizing.

**Limitation of the current implementation:** The safety check is keyword-based (`safety_agent.py` uses a fixed list of terms). It catches obvious violations reliably, but a determined user could rephrase a harmful request to evade detection. A more robust approach would use an LLM-based classifier for the safety check, which is noted as a future improvement.

---

## Risk Analysis and Mitigations

### Hallucination

**Risk:** LLMs can generate fluent but factually incorrect answers, particularly when asked about specific details not present in their training data or the retrieved context.

**How it is handled:** The system uses a two-layer approach. First, the HybridRetriever retrieves relevant chunks from the course lecture PDFs before the TutorAgent answers, so the model is given grounded source material rather than relying on parametric memory alone. Second, the CheckerAgent (`src/agents/checker_agent.py`) reviews every generated answer before it reaches the student, verifying that claims are traceable to retrieved chunks and that no unsupported citations appear. If retrieved evidence is insufficient, the TutorAgent is prompted to say so explicitly rather than fabricate an answer.

**Honest limitation:** The CheckerAgent is lightweight — it validates that citation labels match the retrieved sources, but it cannot verify whether the content of the answer accurately reflects the source text. A model can still misrepresent a retrieved passage while technically citing it. This is a known limitation of citation-based grounding approaches and is not fully solved here.

### Bias in Generated and Training Data

**Risk:** The fine-tuning dataset was generated synthetically using LLM APIs (OpenAI/Groq) and procedural templates. Synthetic data inherits any biases present in the generator model — for example, the phrasing of explanations, the choice of analogies, and the framing of "correct" versus "incorrect" answers may reflect biases in the generator rather than neutral pedagogical best practice.

**How it is handled:** The dataset preparation pipeline (`src/finetuning/prepare_dataset.py`) uses structured output schemas for each persona (Tutor, Examiner, Critic), which constrains generation to defined fields and reduces free-form hallucination. Three distinct behavioral roles were separated so that tutoring, examining, and critical reflection are trained independently, reducing the risk of one behavioral mode contaminating another.

**Honest limitation:** No bias audit was performed on the synthetic dataset. The training data was not reviewed by a domain expert or a pedagogy specialist. The fine-tuned model may perpetuate over-confident or oversimplified explanations inherited from the generator. This is acceptable for a course project but would need rigorous auditing before any real deployment.

### Outdated or Incorrect Online Information

**Risk:** When online retrieval is enabled (hybrid or online-only mode), the assistant fetches content from external sources via Tavily or the `ddgs` fallback. This content may be outdated, inaccurate, or from unreliable sources.

**How it is handled:** Online retrieval is restricted to an approved domain list (`data/raw/approved_online_sources.json`). The source hierarchy in the base system prompt explicitly places course lecture PDFs above online sources, and the TutorAgent is instructed to label online content as external and not allow it to override course definitions. Hybrid answers separate course-grounded claims from external context.

**Honest limitation:** The approved domain list is curated but not exhaustive. The `ddgs` fallback (DuckDuckGo search) does not restrict to the approved list — it is used as a last resort when no API key is configured, and its results may include unreliable sources. Online retrieval is the weakest component of the pipeline in terms of reliability.

### Privacy

**Risk:** An educational assistant that logs conversations or stores student queries could create privacy risks, particularly if queries contain personally identifiable information.

**How it is handled:** The system does not store conversation history beyond the current browser session. Agent traces are saved locally to `outputs/traces/` for debugging and project evaluation purposes, but they contain only the query text and system-generated responses — no student identifiers, grades, or personal information are logged. No sensitive student data was used in training or evaluation.

**Honest limitation:** Trace files are written to disk automatically after every query. In a shared or multi-user deployment, this could allow one user's queries to be visible to another via the Backend Tracking tab. For a production deployment, trace storage would need access controls and a retention policy. In the current single-user course demo context, this is an acceptable trade-off.

---

## Summary of Mitigations

| Risk | Mitigation | Limitation |
|---|---|---|
| Hallucination | RAG grounding + CheckerAgent citation validation | Checker cannot verify content accuracy, only citation format |
| Academic dishonesty | Keyword-based safety agent + system prompt constraints | Keyword matching can be evaded by rephrasing |
| Bias in training data | Structured schemas, separated personas | No bias audit performed |
| Outdated online content | Approved domain list, source hierarchy in prompt | `ddgs` fallback bypasses domain restriction |
| Privacy / data exposure | Session-only history, no personal data in training | Trace files written to disk without access controls |

---

## Future Improvements

- Replace the keyword-based safety check with an LLM classifier prompt (e.g., using the `SAFETY_PROMPT` template already defined in `src/llm/prompts.py`) for more robust detection of adversarial rephrasing.
- Add a content accuracy check to the CheckerAgent, not just citation format validation.
- Conduct a bias review of the synthetic fine-tuning dataset, ideally with input from a course instructor.
- Add access controls and a configurable retention policy to the trace file system before any multi-user deployment.
- Expand the safety test set beyond the current three cases to cover a broader range of adversarial inputs, including indirect prompt injection through uploaded PDFs.
