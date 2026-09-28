"""Prepare V3 concurrent-placement inference from conditioning stills, without GT."""
from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument("repo",type=Path)
parser.add_argument("capture",type=Path)
parser.add_argument("output",type=Path)
args=parser.parse_args()
assert not args.output.exists(), f"refusing overwrite: {args.output}"
suite=json.loads((args.repo/"MultiWorldBench/v3/cases/shared_world/race_condition_consistency.json").read_text())
manifest=json.loads((args.capture/"capture_manifest.json").read_text())
assert manifest["gt_collected"] is False and manifest["native_interaction_executed"] is False
by_id={row["case_id"]:row for row in manifest["captures"]}
keys=["forward","back","left","right","jump","sprint","sneak","attack","use","mount","dismount","place_block","place_entity","mine"]+[f"hotbar.{i}" for i in range(1,10)]

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n")

def pair_first(alpha: Path, bravo: Path, output: Path):
    output.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run([
        "ffmpeg","-nostdin","-hide_banner","-loglevel","error","-y",
        "-i",str(alpha),"-i",str(bravo),"-filter_complex","[0:v][1:v]hstack=inputs=2[v]",
        "-map","[v]","-frames:v","1",str(output),
    ],check=True)

def normalize_first(source: Path, output: Path):
    subprocess.run([
        "ffmpeg","-nostdin","-hide_banner","-loglevel","error","-y",
        "-i",str(source),"-vf","scale=640:360:flags=lanczos,crop=640:352:0:4",
        "-frames:v","1",str(output),
    ],check=True)

def gamma_action(rows, frames=129):
    keyboard=[];camera=[]
    for frame in range(frames):
        values=[0]*23
        if frame < len(rows) and rows[frame]["action"]["attack"]: values[19]=1
        if frame < len(rows) and rows[frame]["action"]["use"]: values[20]=1
        keyboard.append(values);camera.append([0.0,0.0])
    return {"keyboard":keyboard,"camera":camera}

cases=[]
for spec in suite["cases"]:
    cid=spec["case_id"]; captured=by_id[cid]
    dest=args.output/"inputs/cases"/cid;dest.mkdir(parents=True)
    asset=args.output/"assets/race"/cid;asset.mkdir(parents=True)
    role_rows={}
    for role in ("alpha","bravo"):
        source=Path(captured["images"][role]["image"])
        assert hashlib.sha256(source.read_bytes()).hexdigest()==captured["images"][role]["sha256"]
        normalize_first(source,dest/f"{role}_first_frame.png")
        reference=args.repo/"MultiWorldBench/v3"/spec["block_reference_images"][role]
        assert reference.is_file(), reference
        shutil.copyfile(reference,dest/f"{role}_block_reference.png")
        planned=[];canon=[]
        start,end=spec["simultaneous_action"]["active_frame_range"]
        for frame in range(spec["num_frames"]):
            action={key:False for key in keys};action["camera"]=[0.0,0.0]
            action["use"]=start<=frame<end
            planned.append({"frame_index":frame,"epoch_ms":frame*50,"control_source":"synthetic_planned_control","primitive":"USE_HELD_BLOCK" if action["use"] else "IDLE","action":action})
            canon.append({"renderTime":frame*50,"action":action})
        assert sum(row["action"]["use"] for row in planned)==end-start
        write(dest/f"{role}_actions.json",planned)
        write(dest/f"{role}_solaris_actions.json",canon)
        write(asset/f"{role}_actions.json",planned)
        write(asset/("action_left.json" if role=="alpha" else "action_right.json"),gamma_action(planned))
        role_rows[role]=planned
    pair_first(dest/"alpha_first_frame.png",dest/"bravo_first_frame.png",asset/"first_frame.png")
    prompt=("Two Minecraft players simultaneously attempt once to place different blocks "
            "at the same marked target cell.\n")
    (asset/"prompt.txt").write_text(prompt,encoding="utf-8")
    metadata={
        "schema_version":"mwb.v3.asset.race_condition.v1","case_id":cid,
        "group":"race","self_contained":True,"has_native_gt":False,
        "scene_id":spec["scene_id"],"block_pair_id":spec["block_pair_id"],
        "visibility_class":spec["visibility_class"],"target_cell":spec["world"]["target_cell"],
        "intended_materials":spec["intended_materials"],"gamma_frames":129,"solaris_frames":101,
    }
    write(asset/"metadata.json",metadata)
    cases.append({
        "case_id":cid,"num_frames":spec["num_frames"],"fps":spec["fps"],
        "scene_id":spec["scene_id"],"block_pair_id":spec["block_pair_id"],
        "visibility_class":spec["visibility_class"],"intended_materials":spec["intended_materials"],
        "counter_case_of":spec.get("counter_case_of"),
        "block_reference_images":{
            role:str((dest/f"{role}_block_reference.png").resolve())
            for role in ("alpha","bravo")
        },
        "target_cell":spec["world"]["target_cell"],"check_frames":suite["check_frames"],
        "gt_collected":False,
    })
write(args.output/"inputs/inference_manifest.json",{
    "metric_id":"race_condition_consistency","gt_collected":False,
    "native_interaction_executed":False,"case_count":len(cases),"cases":cases,
    "protocol":"conditioning frame; idle f0-20; one simultaneous planned native use at f21; idle f22-100",
})
config=args.output/"config";config.mkdir(parents=True)
(config/"inference_race_condition.yaml").write_text("""defaults:\n - shared\n - device: gpu\n - model: solaris\n - runner: inference\n - dataset@eval_datasets.eval_race_condition: eval_solid_sym_r2\n - _self_\neval_datasets:\n  eval_race_condition:\n    name: eval_race_condition\n""")
write(args.output/"validation.json",{"status":"prepared","gt_collected":False,"case_count":len(cases),"frames_per_case":101,"simultaneous_use_frames":[21]})
print(args.output)
