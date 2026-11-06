import json
import sys
from types import SimpleNamespace

from whichtool.run import keep_console_log, record_outcome


def test_console_output_is_kept_in_the_run_folder(tmp_path):
    out, err = sys.stdout, sys.stderr
    try:
        keep_console_log(tmp_path)
        print("plan line")
        print("Stopping something", file=sys.stderr)
    finally:
        sys.stdout, sys.stderr = out, err
    keep_console_log(tmp_path)  # a second session appends instead of overwriting
    sys.stdout, sys.stderr = out, err
    log = (tmp_path / "console.log").read_text()
    assert "plan line" in log and "Stopping something" in log
    assert log.count("===== ") == 2


def test_each_session_records_how_it_ended(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({"sessions": [{"started_at": "a"}, {"started_at": "b"}]}))
    writer = SimpleNamespace(n_written=6, n_errors=1, n_skipped=0, spent_usd=0.8,
                             broken={("headline", "gpt-5.6-sol", "medium"): "APIError: no credits"})
    record_outcome(tmp_path, writer, left=1)
    sessions = json.loads((tmp_path / "run.json").read_text())["sessions"]
    assert "finished_at" not in sessions[0]  # earlier sessions are left as they were
    assert sessions[1]["calls_failed"] == 1 and sessions[1]["calls_without_success_left"] == 1
    assert sessions[1]["stopped_configurations"] == {"headline gpt-5.6-sol medium": "APIError: no credits"}
