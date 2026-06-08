from src.llm.prompts import get_prompt_template, list_prompt_templates


def test_prompt_templates_are_named_and_documented():
    templates = list_prompt_templates()
    names = {template.name for template in templates}

    assert {"base_system", "router", "tutor", "quiz", "grading", "checker", "safety"} <= names
    assert all(template.purpose for template in templates)
    assert all(template.template.strip() for template in templates)


def test_router_prompt_defines_allowed_retrieval_modes():
    router = get_prompt_template("router")

    for mode in ["offline_only", "online_only", "hybrid", "tool_only", "no_retrieval"]:
        assert mode in router.template
