from paged_attention.bench import cpu_smoke


def test_cpu_smoke_is_correctness_not_performance() -> None:
    result = cpu_smoke()
    assert result["status"] == "correctness_only"
    assert result["performance_data"] is False
    assert result["max_absolute_error"] < 2e-5
    assert "milliseconds" not in result
