"""V4: jointly judge target states; separate agreement from valid execution."""
from collections import Counter
from collections.abc import Mapping

OUTPUT_KEYS = tuple(f"{role}_{field}" for role in ("alpha","bravo") for field in ("visible","count","material"))
MATERIALS = ("candidate_a","candidate_b","candidate_a+candidate_a","candidate_a+candidate_b","candidate_b+candidate_b","other","empty","uncertain")
PROMPT = """Examine six Minecraft images: 1/2 are initial A/B views, 3/4 are post-action A/B views at the SAME timestamp, 5/6 are the intended block references for A/B. Locate the contested cell at the center of the colored ground cross. Examine ONLY that cell and immediately touching new placed blocks, not held items, players or buildings. Both agents attempt one placement at that cell. Either candidate may win; no GT winner is supplied. For EACH view report visible (1 only if the target state is clearly observable, else 0), count (0,1,2,3; 3 means three or more new blocks), material (candidate_a, candidate_b, candidate_a+candidate_a, candidate_a+candidate_b, candidate_b+candidate_b, other, empty, uncertain). For two blocks use an unordered material combination; A+B and B+A both use candidate_a+candidate_b. Recognize material under perspective and lighting changes; references may show flat textures. A clearly observable empty target has count 0/material empty; a hidden target must use visible 0/count null/material uncertain. One or two blocks are BOTH valid outcomes, including an adjacent second block. Three or more use count 3/material other. Do not infer a hidden target from the other view. Return ONLY the six flat JSON fields alpha_visible, alpha_count, alpha_material, bravo_visible, bravo_count, bravo_material."""
OUTPUT_SCHEMA={"type":"object","additionalProperties":False,"required":list(OUTPUT_KEYS),"properties":{
 **{f"{r}_visible":{"type":"integer","enum":[0,1]} for r in ("alpha","bravo")},
 **{f"{r}_count":{"enum":[None,0,1,2,3]} for r in ("alpha","bravo")},
 **{f"{r}_material":{"enum":list(MATERIALS)} for r in ("alpha","bravo")}}}

def validate_output(value):
 if not isinstance(value,Mapping) or set(value)!=set(OUTPUT_KEYS): raise ValueError("expected exactly six target-state fields")
 value=dict(value)
 for r in ("alpha","bravo"):
  v,c,m=(value[f"{r}_{f}"] for f in ("visible","count","material"))
  if type(v) is not int or v not in (0,1): raise ValueError("visible must be 0/1 integer")
  if c is not None and (type(c) is not int or c not in (0,1,2,3)): raise ValueError("invalid count")
  if m not in MATERIALS: raise ValueError("invalid material")
  if not v and (c is not None or m!="uncertain"): raise ValueError("hidden state must be unknown")
  if v and c is None: raise ValueError("visible state needs count")
  if v and ((c==0 and m!="empty") or (c!=0 and m=="empty")): raise ValueError("count/material contradiction")
  if c==1 and "+" in m: raise ValueError("single block cannot have two materials")
  if c==2 and m in ("candidate_a","candidate_b"): raise ValueError("two blocks require material combination")
  if c==3 and m not in ("other","uncertain"): raise ValueError("three-plus blocks outside protocol")
 return value

def frame_score(value):
 v=validate_output(value)
 a,b=v["alpha_count"],v["bravo_count"]
 am,bm=v["alpha_material"],v["bravo_material"]
 visible=v["alpha_visible"]==v["bravo_visible"]==1
 agreement=int(visible and a==b and (a==0 or (am==bm and am!="uncertain")))
 valid=int(visible and a==b and a in (1,2) and am==bm and am in MATERIALS[:5])
 reason=("target_hidden" if not visible else "both_empty" if a==b==0 else
         "too_many_blocks" if 3 in (a,b) else "count_mismatch" if a!=b else
         "material_uncertain" if "uncertain" in (am,bm) else
         "material_mismatch" if am!=bm else "wrong_material" if am=="other" else "pass")
 return {"state_agreement":agreement,"valid_placement":valid,"joint_pass":int(agreement and valid),"reason":reason}

def score_case(frame_measurements):
 frames=[validate_output(v) for v in frame_measurements]
 if len(frames)!=3: raise ValueError("V4 requires exactly three scheduled checkpoints; missing judge output is a task error, not zero")
 checks=[frame_score(v) for v in frames]
 vote=lambda key:int(sum(c[key] for c in checks)>=2)
 return {"metric_id":"race_condition_consistency","protocol":"race.v4",
 "score":vote("joint_pass"),"score_0_100":100*vote("joint_pass"),"status":"binary",
 "state_agreement":vote("state_agreement"),"valid_placement":vote("valid_placement"),
 "questions":{"state_agreement":vote("state_agreement"),"valid_placement":vote("valid_placement")},
 "frames":frames,"frame_checks":checks,"failure_reasons":dict(Counter(c["reason"] for c in checks if c["reason"]!="pass")),
 "aggregation":"joint pass per checkpoint, then at least 2 of 3 checkpoints"}

def aggregate_cases(rows):
 rows=list(rows)
 for row in rows:
  if row["score"] not in (0,1): raise ValueError("invalid binary score")
 mean=lambda key:sum(r[key] for r in rows)/len(rows) if rows else None
 return {"metric_id":"race_condition_consistency","protocol":"race.v4","score":mean("score"),
 "score_0_100":100*mean("score") if rows else None,"state_agreement":mean("state_agreement"),
 "valid_placement":mean("valid_placement"),"eligible_cases":len(rows),"total_cases":len(rows)}
