from src.agents.adaptation_agent import StudentAdaptationAgent
from src.agents.tutor_agent import TutorAgent
from src.schemas import DocumentChunk, RetrievedChunk


def test_adaptation_agent_maps_levels_to_quiz_difficulty():
    agent = StudentAdaptationAgent()

    assert agent.run("beginner")["quiz_difficulty"] == "easy"
    assert agent.run("intermediate")["quiz_difficulty"] == "medium"
    assert agent.run("advanced")["quiz_difficulty"] == "hard"


def test_tutor_agent_changes_answer_by_student_level():
    chunk = DocumentChunk(
        chunk_id="c1",
        text="RAG retrieves relevant documents and adds them to the prompt.",
        source="lecture.pdf",
        source_type="course_pdf",
        page=1,
    )
    retrieved = [RetrievedChunk(chunk=chunk, keyword_score=1.0)]
    tutor = TutorAgent()

    beginner = tutor.answer("Explain RAG", retrieved, "offline_only", StudentAdaptationAgent().run("beginner"))
    advanced = tutor.answer("Explain RAG", retrieved, "offline_only", StudentAdaptationAgent().run("advanced"))

    assert "Adapted for: Beginner" in beginner
    assert "Analogy" in beginner
    assert "Adapted for: Advanced" in advanced
    assert "Tradeoff" in advanced
