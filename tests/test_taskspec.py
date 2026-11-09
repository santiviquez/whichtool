from whichtool.taskspec import load_task


def test_who_wrote_each_wording(tmp_path):
    task = tmp_path / "t"
    task.mkdir()
    (task / "jobs.yaml").write_text(
        "jobs:\n"
        "  - id: j1\n    text: Old job.\n"
        "  - id: j2\n    by: ChatGPT\n    text: New job.\n"
    )
    (task / "variants.yaml").write_text(
        "frozen: true\nvariants:\n"
        "  - id: j1-v01\n    job_id: j1\n    text: Old rewording.\n"
        "  - id: j2-v01\n    job_id: j2\n    by: ChatGPT\n    text: New rewording.\n"
    )
    spec = load_task("t", tmp_path)
    assert {v.id: v.by for v in spec.variants} == {
        "j1-v00": "ChatGPT", "j1-v01": "Claude", "j2-v00": "ChatGPT", "j2-v01": "ChatGPT",
    }
    sha = spec.variants_sha
    (task / "variants.yaml").write_text((task / "variants.yaml").read_text().replace("by: ChatGPT", "by: Someone"))
    assert load_task("t", tmp_path).variants_sha == sha  # the credit is not part of the prompts
