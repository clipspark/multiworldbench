import json
from pathlib import Path
import pytest
from judge_jobs import parse, prepare
from controller.multiworldbench_v3.metrics import id_consistency
from controller.multiworldbench_v3.tools.score_cross_view_translation20_direct import refresh_judgment

def test_json_fence_uses_schema():
    assert parse('text ```json\n{"gt_same_building":1,"generated_same_building":0}\n```', id_consistency.validate_output)["generated_same_building"] == 0
    with pytest.raises(ValueError):
        parse("Both buildings are the same.", id_consistency.validate_output)
    with pytest.raises(ValueError):
        parse('{"gt_same_building":true,"generated_same_building":1}', id_consistency.validate_output)

def test_job_checks_and_input_hash(tmp_path):
    paths=[]
    for i in range(4):
        path=tmp_path/f"{i}.png"
        path.write_bytes(bytes([i]))
        paths.append(path.name)
    cases=[{"case_id":"x"}]
    job={"job_id":"x-view","case_id":"x","images":paths}
    a=prepare("revisit",cases,[job],tmp_path)[0][-1]
    (tmp_path/paths[0]).write_bytes(b"changed")
    b=prepare("revisit",cases,[job],tmp_path)[0][-1]
    assert a!=b
    with pytest.raises(ValueError,match="duplicate"):
        prepare("revisit",cases,[job,job],tmp_path)
    with pytest.raises(ValueError,match="safe"):
        prepare("revisit",cases,[dict(job,job_id="../escape")],tmp_path)
    with pytest.raises(ValueError,match="ordered images"):
        prepare("revisit",cases,[dict(job,images=paths[:3])],tmp_path)

def test_missing_response_is_format_error():
    row=refresh_judgment({"response":None,"expected_direction":"left"})
    assert row["score"] is None
    assert row["direction_parse_status"]=="format_error"
