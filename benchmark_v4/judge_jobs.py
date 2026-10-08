#!/usr/bin/env python3
"""Judge explicitly ordered image jobs using the released prompts and validators."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from PIL import Image
from controller.multiworldbench_v3.metrics import (
    shared_state_synchronization as sync, shared_state_persistence as persistence,
    building_reveal as building, race_condition_consistency_v4 as race,
    id_consistency as identity, revisit_handoff_prompt,
)
from score_shared_world import SUITES

COUNTS = {"synchronization":4, "persistence":2, "building":4, "race":6, "revisit":4}

def protocol(suite, case):
    if suite == "synchronization":
        return sync.build_prompt(case), lambda obj:sync.validate_output(obj,case["operation"])
    if suite == "persistence":
        return persistence.build_prompt(case), persistence.validate_output
    if suite == "building":
        return building.PROMPT, building.validate_output
    if suite == "race":
        return race.PROMPT, race.validate_output
    return revisit_handoff_prompt.PROMPT, identity.validate_output

def parse(raw, validate):
    # Decode objects; accept a surrounding code fence, never infer labels from prose.
    decoder = json.JSONDecoder()
    for i, ch in enumerate(raw):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(raw[i:])
            return validate(obj)
        except (ValueError, TypeError, KeyError):
            continue
    raise ValueError("no JSON object satisfies the metric schema")

def dump(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")
    tmp.replace(path)

def prepare(suite, cases, jobs, base):
    by_id={c["case_id"]:c for c in cases}
    prepared=[]
    seen=set()
    for job in jobs:
        jid=job["job_id"]
        if not isinstance(jid,str) or not jid or any(x not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for x in jid):
            raise ValueError("job_id must be a unique safe filename")
        if jid in seen:
            raise ValueError(f"duplicate job_id: {jid}")
        seen.add(jid)
        case=by_id[job["case_id"]]
        if suite != "revisit" and job["frame"] not in SUITES[suite][1]:
            raise ValueError(f"{jid}: invalid checkpoint")
        prompt,validate=protocol(suite,case)
        paths=[Path(x) if Path(x).is_absolute() else base/Path(x) for x in job["images"]]
        if len(paths)!=COUNTS[suite]:
            raise ValueError(f"{jid}: expected {COUNTS[suite]} ordered images")
        digests=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        payload={"suite":suite,"job":job,"prompt":prompt,"image_sha256":digests}
        digest=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
        prepared.append((job,paths,prompt,validate,digest))
    return prepared

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite",choices=COUNTS,required=True)
    p.add_argument("--cases",type=Path,required=True)
    p.add_argument("--jobs",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,help="Local Qwen2.5-VL checkpoint")
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--max-attempts",type=int,choices=(1,2),default=2)
    p.add_argument("--max-new-tokens",type=int,default=512)
    p.add_argument("--max-pixels",type=int,default=384*28*28)
    p.add_argument("--cached-only",action="store_true")
    a=p.parse_args()
    cases=json.loads(a.cases.read_text())["cases"]
    prepared=prepare(a.suite,cases,json.loads(a.jobs.read_text())["jobs"],a.jobs.parent)
    # Reject incomplete job lists before loading a GPU checkpoint.
    if a.suite != "revisit":
        for case in cases:
            frames=sorted(job["frame"] for job, *_ in prepared if job["case_id"]==case["case_id"])
            if tuple(frames)!=SUITES[a.suite][1]:
                p.error(f'{case["case_id"]}: jobs must contain every checkpoint exactly once')
    generation={"do_sample":False,"seed":12345,
        "max_new_tokens":a.max_new_tokens,"max_pixels":a.max_pixels}
    checkpoint=str(a.checkpoint.resolve()) if a.checkpoint else None
    pending=[];records=[]
    for item in prepared:
        job,paths,prompt,validate,digest=item
        path=a.output_dir/"judgments"/(job["job_id"]+".json")
        if path.exists():
            saved=json.loads(path.read_text())
            if saved.get("input_sha256")!=digest:
                p.error(f"{path}: cached input/prompt hash mismatch")
            if checkpoint and saved.get("checkpoint")!=checkpoint:
                p.error(f"{path}: cached checkpoint mismatch")
            if saved.get("generation")!=generation:
                p.error(f"{path}: cached generation settings mismatch")
            records.append(saved)
        else:
            pending.append(item)
    if pending:
        if a.cached_only:
            p.error(f"{len(pending)} judgments missing")
        if a.checkpoint is None or not (a.checkpoint/"config.json").is_file():
            p.error("--checkpoint must be a local Qwen2.5-VL checkpoint")
        import torch
        from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
        from qwen_vl_utils import process_vision_info
        torch.manual_seed(12345);torch.set_num_threads(4)
        model=Qwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint,torch_dtype=torch.bfloat16,device_map={"":a.device},attn_implementation="sdpa").eval()
        processor=AutoProcessor.from_pretrained(checkpoint,max_pixels=a.max_pixels)
        for job,paths,prompt,validate,digest in pending:
            images=[]
            for path in paths:
                with Image.open(path) as im:
                    images.append(im.convert("RGB"))
            attempts=[];output=None
            for attempt in range(a.max_attempts):
                suffix="" if attempt==0 else "\nThe previous response did not match the JSON schema. Return only the requested JSON fields and allowed values."
                messages=[{"role":"user","content":[{"type":"text","text":prompt+suffix}]+[
                    {"type":"image","image":im} for im in images]}]
                text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                image_inputs,video_inputs=process_vision_info(messages)
                inputs=processor(text=[text],images=image_inputs,videos=video_inputs,padding=True,return_tensors="pt").to(model.device)
                with torch.inference_mode():
                    ids=model.generate(**inputs,max_new_tokens=a.max_new_tokens,do_sample=False)
                raw=processor.batch_decode(ids[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0]
                try:
                    output=parse(raw,validate);error=None
                except ValueError as exc:
                    error=str(exc)
                attempts.append({"raw":raw,"format_error":error})
                if output is not None:
                    break
            saved={"job_id":job["job_id"],"case_id":job["case_id"],
                "frame":job.get("frame"),"suite":a.suite,"input_sha256":digest,
                "checkpoint":checkpoint,"prompt":prompt,"images":[str(x) for x in paths],
                "output":output,"status":"valid" if output is not None else "format_error",
                "attempts":attempts,"generation":generation}
            dump(a.output_dir/"judgments"/(job["job_id"]+".json"),saved)
            records.append(saved)
            print(job["job_id"],saved["status"],flush=True)
    records.sort(key=lambda x:x["job_id"])
    # Validate caches again: cached-only does not bypass schema checks.
    for record in records:
        case=next(c for c in cases if c["case_id"]==record["case_id"])
        _,validate=protocol(a.suite,case)
        if record["status"]=="valid":
            record["output"]=validate(record["output"])
    bad=sum(r["status"]!="valid" for r in records)
    dump(a.output_dir/"judgments.json",{"suite":a.suite,"jobs":records,
        "format_errors":bad,"note":"A format failure is missing evidence, never a model zero."})
    if a.suite != "revisit" and not bad:
        evidence=[]
        for case in cases:
            rows=[r for r in records if r["case_id"]==case["case_id"]]
            rows.sort(key=lambda r:r["frame"])
            if tuple(r["frame"] for r in rows)!=SUITES[a.suite][1]:
                p.error(f'{case["case_id"]}: jobs must contain every checkpoint exactly once')
            evidence.append({"case_id":case["case_id"],"checks":[
                {"frame":r["frame"],"output":r["output"]} for r in rows]})
        dump(a.output_dir/"evidence.json",{"cases":evidence})
    if bad:
        sys.exit(1)

if __name__=="__main__":
    main()
