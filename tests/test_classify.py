import pytest

from whichtool.classify import RESULT_COLUMNS, AliasTable, classify, norm, result_row

ALIASES = {
    "azure_di": {"names": ["Azure AI Document Intelligence", "Azure Form Recognizer", "Document Intelligence"],
                 "code": ["azure.ai.documentintelligence", "azure-ai-formrecognizer"]},
    "google_docai": {"names": ["Google Document AI", "Google Cloud Document AI", "Document AI"],
                     "code": ["google.cloud.documentai", "documentai.googleapis.com"]},
    "aws_textract": {"names": ["Amazon Textract", "AWS Textract", "Textract"], "code": ["textract"]},
    "tesseract": {"names": ["Tesseract"], "code": ["pytesseract"], "hosted": False},
    "other_api": {"names": ["Other API"]},
}


@pytest.fixture
def table():
    return AliasTable(ALIASES)


def row(api="", **kw):
    base = {"study": "headline", "condition": "no_search", "job_id": "j1", "status": "ok",
            "refused": "False", "parse_ok": "True", "api": api}
    return {**base, **kw}


def apis(out):
    return tuple(x for x in (out["canonical_api"], out["canonical_api_2"], out["canonical_api_3"]) if x)


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


def test_exact_only_aliases():
    t = AliasTable({"unstructured_io": {"names": ["Unstructured API"], "exact": ["Unstructured"]},
                    "azure_di": {"names": ["Azure AI Document Intelligence"]}})
    assert t.match_name("Unstructured").api_id == "unstructured_io"
    # a generic word inside a longer answer must not create a second match
    assert t.match_name("Azure AI Document Intelligence for unstructured scans").api_id == "azure_di"


@pytest.mark.parametrize(
    "api,status,expected",
    [
        ("Azure AI Document Intelligence", "mapped", ("azure_di",)),
        ("Other API", "mapped", ("other_api",)),
        ("Tesseract", "mapped", ("tesseract",)),  # self-hosted is still what the model named
        ("Some Startup OCR", "unmapped", ()),
        ("It depends on your volume", "hedge", ()),
        ("I can't recommend one", "refusal", ()),
    ],
)
def test_statuses(table, api, status, expected):
    out = classify(row(api), "", table)
    assert out["status"] == status and apis(out) == expected


def test_two_apis_are_both_kept(table):
    # Naming two APIs is a recommendation, not a hedge, even with a hedge word in it.
    out = classify(row("Either AWS Textract or Google Document AI"), "", table)
    assert out["match_method"] == "multiple" and out["status"] == "mapped"
    assert apis(out) == ("aws_textract", "google_docai") and out["n_apis"] == 2
    # API 1 is the one named first
    assert apis(classify(row("Google Document AI or AWS Textract"), "", table)) == ("google_docai", "aws_textract")


def test_manual_apis_replace_the_alias_match(table):
    out = classify(row("Amazon Textract"), "", table, manual=["tesseract", "azure_di"])
    assert out["match_method"] == "manual" and out["status"] == "mapped"
    assert apis(out) == ("tesseract", "azure_di") and out["n_apis"] == 2
    # it does not override a failed call
    assert classify(row("", status="api_error"), "", table, manual=["azure_di"])["status"] == "api_error"


def test_refusal_and_errors(table):
    assert classify(row("", refused="True"), "", table)["status"] == "refusal"
    assert classify(row("", parse_ok="False"), "", table)["status"] == "format_error"
    failed = classify(row("Amazon Textract", status="api_error"), "", table)
    assert failed["status"] == "api_error" and apis(failed) == ()


def test_cut_off_answers_are_truncated_not_format_errors(table):
    for stop in ("max_tokens", "pause_turn", "incomplete:max_output_tokens"):
        assert classify(row("", parse_ok="False", stop_reason=stop), "", table)["status"] == "truncated"
    assert classify(row("", parse_ok="False", stop_reason="end_turn"), "", table)["status"] == "format_error"


def test_code_study(table):
    r = row("", study="code", condition="write_code", parse_ok="")
    code = "```python\nfrom azure.ai.documentintelligence import DocumentIntelligenceClient\n```"
    assert apis(classify(r, code, table)) == ("azure_di",)
    both = "```python\nfrom azure.ai.documentintelligence import X\nboto3.client('textract')\n```"
    t = AliasTable({**ALIASES, "aws_textract": {"code": ["client('textract'"]}})
    out = classify(r, both, t)
    # two hosted APIs: both kept, API 1 in aliases.yaml order
    assert out["match_method"] == "code_multiple" and apis(out) == ("azure_di", "aws_textract")
    assert classify(r, "Sorry, no code.", table)["status"] == "format_error"


def test_code_ignores_comments_prose_and_helpers(table):
    text = (
        "Here is the code.\n```python\nimport fitz  # render pages\n"
        "# You could also use pytesseract or Amazon Textract\n"
        "from azure.ai.documentintelligence import DocumentIntelligenceClient\n```\n"
        "Alternatively, google.cloud.documentai works too."
    )
    r = row("", study="code", condition="write_code", parse_ok="")
    t = AliasTable({**ALIASES, "pymupdf": {"code": ["import fitz"], "hosted": False}})
    assert apis(classify(r, text, t)) == ("azure_di",)


def test_result_row_has_every_column(table):
    r = row("Azure AI Document Intelligence", call_id="c1", model="m", effort="high", variant_id="j1-v00", epoch="1")
    out = result_row(r, classify(r, "", table), {"output_text": "", "urls": ["https://x.test/a", "https://x.test/b"]},
                     "Which API?", "accepted", {"aliases": "a", "manual_apis": ""})
    assert set(out) == set(RESULT_COLUMNS)
    assert out["api_1"] == "azure_di" and out["answer"] == "Azure AI Document Intelligence" and out["status"] == "mapped"
    assert out["wording_by"] == "ChatGPT" and out["question"] == "Which API?" and out["review"] == "accepted"
    assert out["cited_urls"] == "https://x.test/a https://x.test/b" and out["call_status"] == "ok"
    assert out["aliases_sha"] == "a"
    code = row("", study="code", call_id="c2", variant_id="j1-v01")
    out = result_row(code, classify(code, "x", table), {"output_text": "the whole reply"}, "", "")
    assert out["answer"] == "the whole reply" and out["wording_by"] == "Claude"
