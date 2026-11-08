import pytest

from whichtool.score import AliasTable, classify, grade_apis, norm
from whichtool.stats import clustered_mean_ci, paired_diff_ci, pass_hat_k

ALIASES = {
    "azure_di": {"names": ["Azure AI Document Intelligence", "Azure Form Recognizer", "Document Intelligence"],
                 "code": ["azure.ai.documentintelligence", "azure-ai-formrecognizer"]},
    "google_docai": {"names": ["Google Document AI", "Google Cloud Document AI", "Document AI"],
                     "code": ["google.cloud.documentai", "documentai.googleapis.com"]},
    "aws_textract": {"names": ["Amazon Textract", "AWS Textract", "Textract"], "code": ["textract"]},
    "tesseract": {"names": ["Tesseract"], "code": ["pytesseract"], "hosted": False},
    "other_api": {"names": ["Other API"]},
}
GT = {
    "comparison": ["azure_di", "google_docai", "aws_textract"],
    "jobs": {"j1": {"best": "azure_di", "alternates": ["google_docai"]}},
}


@pytest.fixture
def table():
    return AliasTable(ALIASES)


def row(api="", **kw):
    base = {"study": "headline", "condition": "no_search", "job_id": "j1", "status": "ok",
            "refused": "False", "parse_ok": "True", "api": api}
    return {**base, **kw}


def test_norm():
    assert norm("Azure AI Document Intelligence (formerly Form Recognizer)") == \
        "azure ai document intelligence formerly form recognizer"
    assert norm("Amazon Textract™") == "amazon textract"


def test_exact_and_contains(table):
    assert table.match_name("AWS Textract").api_id == "aws_textract"
    m = table.match_name("Microsoft Azure AI Document Intelligence (prebuilt-invoice model)")
    assert m.api_id == "azure_di"


def test_longest_alias_wins(table):
    # "Document AI" must not also fire inside "Google Cloud Document AI"
    assert table.match_name("Google Cloud Document AI").api_id == "google_docai"


def test_two_apis_are_kept_and_graded_on_the_best(table):
    # Naming two APIs is a recommendation, not a hedge, even with a hedge word in it.
    out = classify(row("Either AWS Textract or Google Document AI"), "", table, GT)
    assert out["match_method"] == "multiple"
    assert (out["canonical_api"], out["canonical_api_2"], out["canonical_api_3"]) == ("aws_textract", "google_docai", "")
    assert out["n_apis"] == 2
    assert out["status"] == "alternate" and out["graded_api"] == "google_docai"
    assert out["strict_correct"] == 0 and out["lenient_correct"] == 1
    # API 1 is the one named first
    assert classify(row("Google Document AI or AWS Textract"), "", table, GT)["canonical_api"] == "google_docai"


def test_manual_apis_replace_the_alias_match(table):
    # A hand classification wins over the alias table; Tesseract is self-hosted, so it grades off-list
    # and the answer is graded on Azure.
    out = classify(row("Amazon Textract"), "", table, GT, manual=["tesseract", "azure_di"])
    assert out["match_method"] == "manual"
    assert (out["canonical_api"], out["canonical_api_2"], out["n_apis"]) == ("tesseract", "azure_di", 2)
    assert out["status"] == "correct" and out["graded_api"] == "azure_di"
    # it does not override a failed call
    assert classify(row("", status="api_error"), "", table, GT, manual=["azure_di"])["status"] == "api_error"


def test_multi_api_grading_rules(table):
    ids = ["aws_textract", "azure_di"]
    assert grade_apis(ids, "j1", table, GT, "best") == ("azure_di", "correct")
    assert grade_apis(ids, "j1", table, GT, "first") == ("aws_textract", "wrong")
    # no ground truth: nothing to choose between, so API 1 is reported
    assert grade_apis(ids, "j1", table, None, "best") == ("aws_textract", "unscored")
    with pytest.raises(ValueError):
        grade_apis(ids, "j1", table, GT, "half")


def test_report_labels_multi_api_picks():
    from whichtool.report import pick

    assert pick({"canonical_api": "a", "canonical_api_2": "b", "canonical_api_3": "", "status": "correct"}) == "a + b"
    assert pick({"canonical_api": "a", "status": "wrong"}) == "a"
    assert pick({"canonical_api": "", "status": "hedge"}) == "(hedge)"


@pytest.mark.parametrize(
    "api,status,strict,lenient",
    [
        ("Azure AI Document Intelligence", "correct", 1, 1),
        ("Google Document AI", "alternate", 0, 1),
        ("Amazon Textract", "wrong", 0, 0),
        ("Other API", "off_list", 0, 0),
        ("Tesseract", "off_list", 0, 0),
        ("Some Startup OCR", "unmapped", 0, 0),
        ("It depends on your volume", "hedge", 0, 0),
    ],
)
def test_statuses(table, api, status, strict, lenient):
    out = classify(row(api), "", table, GT)
    assert out["status"] == status
    assert out["strict_correct"] == strict and out["lenient_correct"] == lenient


def test_refusal_and_errors(table):
    refused = classify(row("", refused="True"), "", table, GT)
    assert refused["status"] == "refusal" and refused["strict_correct"] == 0
    assert classify(row("", parse_ok="False"), "", table, GT)["status"] == "format_error"
    failed = classify(row("", status="api_error"), "", table, GT)
    # a failed call says nothing about the model's choice: not graded, so not in accuracy
    assert failed["status"] == "api_error" and failed["strict_correct"] == "" and failed["lenient_correct"] == ""


def test_no_ground_truth_is_unscored(table):
    out = classify(row("Amazon Textract"), "", table, None)
    assert out["status"] == "unscored" and out["canonical_api"] == "aws_textract"
    assert out["strict_correct"] == ""


def test_code_study(table):
    code = "```python\nfrom azure.ai.documentintelligence import DocumentIntelligenceClient\n```"
    r = row("", study="code", condition="write_code", parse_ok="")
    assert classify(r, code, table, GT)["status"] == "correct"
    both = "```python\nfrom azure.ai.documentintelligence import X\nclient('textract')\n```"
    both = both.replace("client('textract')", "boto3.client('textract')")
    t = AliasTable({**ALIASES, "aws_textract": {"code": ["client('textract'"]}})
    out = classify(r, both, t, GT)
    # two hosted APIs: both kept, API 1 in aliases.yaml order, graded on the best
    assert (out["canonical_api"], out["canonical_api_2"], out["status"]) == ("azure_di", "aws_textract", "correct")
    assert classify(r, "Sorry, no code.", table, GT)["status"] == "format_error"


def test_clustered_ci_wider_than_naive():
    # Two variants, each always the same answer: clustering should widen the interval.
    values = [1] * 10 + [0] * 10
    clusters = ["a"] * 10 + ["b"] * 10
    r = clustered_mean_ci(values, clusters)
    assert r["mean"] == 0.5 and r["clusters"] == 2
    naive_se = (0.25 / 20) ** 0.5
    assert r["se"] > naive_se


def test_paired_diff():
    a = {"v1": [1, 1], "v2": [1, 0], "v3": [0, 0]}
    b = {"v1": [1, 0], "v2": [0, 0], "v3": [0, 0]}
    d = paired_diff_ci(a, b)
    assert abs(d["diff"] - (0.5 + 0.5 + 0) / 3) < 1e-9 and d["units"] == 3


def test_pass_hat_k():
    assert pass_hat_k(3, 3, 3) == 1.0
    assert pass_hat_k(2, 4, 2) == pytest.approx(1 / 6)
    assert pass_hat_k(0, 4, 1) == 0.0


def test_t_distribution():
    from whichtool.stats import t95, t_p_two_sided
    assert t95(1) == pytest.approx(12.706, abs=1e-3)
    assert t95(10) == pytest.approx(2.228, abs=1e-3)
    assert t95(1000) == pytest.approx(1.962, abs=1e-3)
    assert t_p_two_sided(2.228, 10) == pytest.approx(0.05, abs=1e-4)


def test_exact_only_aliases():
    t = AliasTable({"unstructured_io": {"names": ["Unstructured API"], "exact": ["Unstructured"]},
                    "azure_di": {"names": ["Azure AI Document Intelligence"]}})
    assert t.match_name("Unstructured").api_id == "unstructured_io"
    # a generic word inside a longer answer must not create a second match
    assert t.match_name("Azure AI Document Intelligence for unstructured scans").api_id == "azure_di"


def test_code_ignores_comments_prose_and_helpers(table):
    text = (
        "Here is the code.\n```python\nimport fitz  # render pages\n"
        "# You could also use pytesseract or Amazon Textract\n"
        "from azure.ai.documentintelligence import DocumentIntelligenceClient\n```\n"
        "Alternatively, google.cloud.documentai works too."
    )
    r = row("", study="code", condition="write_code", parse_ok="")
    t = AliasTable({**ALIASES, "pymupdf": {"code": ["import fitz"], "hosted": False}})
    out = classify(r, text, t, GT)
    assert out["status"] == "correct", out


def test_cut_off_answers_are_truncated_not_format_errors(table):
    for stop in ("max_tokens", "pause_turn", "incomplete:max_output_tokens"):
        out = classify(row("", parse_ok="False", stop_reason=stop), "", table, GT)
        assert out["status"] == "truncated" and out["strict_correct"] == 0
    assert classify(row("", parse_ok="False", stop_reason="end_turn"), "", table, GT)["status"] == "format_error"


def test_report_clusters_by_job_and_pairs_on_shared_wordings():
    from whichtool.report import Config, paired_by_job

    def rows(values):  # values: {variant_id: [0/1, ...]}, job = the part before "-"
        return [{"variant_id": v, "job_id": v.split("-")[0], "strict_correct": str(x)}
                for v, xs in values.items() for x in xs]

    head = Config(rows({"j1-v00": [1, 1], "j1-v01": [1, 1], "j1-v02": [1, 1], "j2-v00": [0, 0], "j2-v01": [1, 0]}),
                  "strict_correct")
    assert head.ci()["clusters"] == 2  # two jobs, not five wordings
    side = Config(rows({"j1-v00": [0, 0], "j2-v00": [1, 1]}), "strict_correct")  # fewer wordings
    d = paired_by_job(side, head)
    # only v00 is shared: j1 0 - 1 = -1, j2 1 - 0 = +1
    assert d["units"] == 2 and abs(d["diff"]) < 1e-9
