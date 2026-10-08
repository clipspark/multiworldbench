import pytest
from score_shared_world import score_evidence
from controller.multiworldbench_v3.metrics import race_condition_consistency_v4 as race

def test_visibility_gate_before_majority():
    # Ungated question votes would wrongly pass: two invisible checks say yes.
    case = {"case_id":"c","operation":"destroy"}
    values = [
      {"cross_center_visible":0,"action_success":1,"same_block_count":1,"same_target_location":1},
      {"cross_center_visible":0,"action_success":1,"same_block_count":1,"same_target_location":1},
      {"cross_center_visible":1,"action_success":1,"same_block_count":1,"same_target_location":1}]
    row = {"case_id":"c","checks":[{"frame":f,"output":v} for f,v in zip((60,70,80),values)]}
    result = score_evidence("synchronization",[case],[row])
    assert result["summary"]["score_0_100"] == 0
    assert result["cases"][0]["gated_checkpoints"][0]["action_success"] == 0

def test_missing_case_is_task_error():
    with pytest.raises(ValueError, match="case mismatch"):
        score_evidence("persistence",[{"case_id":"c"}],[])

def test_wrong_checkpoint_is_task_error():
    with pytest.raises(ValueError, match="checkpoints"):
        score_evidence("race",[{"case_id":"c"}],[{"case_id":"c","checks":[]}])

def test_empty_race_agrees_but_scores_zero():
    value = {f"{r}_{k}":v for r in ("alpha","bravo")
             for k,v in (("visible",1),("count",0),("material","empty"))}
    result = race.score_case([value]*3)
    assert result["state_agreement"] == 1
    assert result["score"] == 0

def test_independent_style_and_pc_min(tmp_path):
    import numpy as np
    from PIL import Image
    from score_revisit import score_evidence as revisit
    a=np.full((64,64,3),(230,20,20),dtype=np.uint8)
    b=np.full((64,64,3),(20,230,20),dtype=np.uint8)
    Image.fromarray(a).save(tmp_path/"a.png"); Image.fromarray(b).save(tmp_path/"b.png")
    cases=[{"case_id":f"g{k}","independence_group":"g"} for k in ("P","C")]
    evidence=[]
    for k in ("P","C"):
        evidence.append({"case_id":f"g{k}","style_pairs":[{
          "gt_initial":"a.png","gt_return":"a.png","generated_initial":"a.png",
          "generated_return":"a.png" if k=="P" else "b.png"}],
          "id_outputs":[{"gt_same_building":1,"generated_same_building":int(k=="P")}]})
    out=revisit(cases,evidence,tmp_path)
    assert out["cases"][0]["style"]["score"] == 100
    assert out["summary"]["style_score_0_100"] == 0
    assert out["summary"]["id_score_0_100"] == 0
