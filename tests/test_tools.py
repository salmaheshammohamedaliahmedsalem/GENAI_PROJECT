from src.tools.calculator_tool import calculate_expression

def test_calculator():
    r = calculate_expression("8 / 10")
    assert r["ok"]
    assert r["result"] == 0.8