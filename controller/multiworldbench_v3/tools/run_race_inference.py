from __future__ import annotations
import fcntl,json,os,shutil,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);gpu=sys.argv[2];name="race_condition_v3";frames=101
sol=Path("/data2/xuzhangbo/multihuman-animate/solaris");py="/data2/xuzhangbo/.conda/envs/solaris/bin/python"
lock=(root/"inference.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
cases=json.loads((root/"inputs/inference_manifest.json").read_text())["cases"]
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+"\n")
def run(cmd,**kwargs):return subprocess.run(list(map(str,cmd)),check=True,**kwargs)
data=root/"dataset/test";data.mkdir(parents=True,exist_ok=True);(root/"logs").mkdir(exist_ok=True)
for index,case in enumerate(cases):
    for role,title in (("alpha","Alpha"),("bravo","Bravo")):
        source=root/"inputs/cases"/case["case_id"]
        video=data/f"{index:02}_{title}_instance_0_camera.mp4"
        run(["ffmpeg","-nostdin","-v","error","-y","-loop","1","-framerate","20","-i",source/f"{role}_first_frame.png","-frames:v",frames+1,"-c:v","libx264","-threads","4","-crf","18","-pix_fmt","yuv420p",video])
        shutil.copyfile(source/f"{role}_solaris_actions.json",data/f"{index:02}_{title}_instance_0.json")
write(sol/"src/data/eval_ids/eval_ids_eval_race_condition.json",[[i,0,frames,0,frames] for i in range(len(cases))])
env={**os.environ,"CUDA_VISIBLE_DEVICES":gpu,"OMP_NUM_THREADS":"4","MKL_NUM_THREADS":"4","OPENBLAS_NUM_THREADS":"4","PYTHONPATH":str(sol)}
cmd=[py,"src/inference.py","--config-dir",str(root/"config"),"--config-name","inference_race_condition",f"experiment_name={name}",f"device.eval_data_dir={root}/dataset",f"device.output_dir={root}/solaris_raw",f"device.pretrained_model_dir={sol}/pretrained",f"device.eval_num_samples={len(cases)}",f"runner.num_frames_eval={frames}","runner.params.eval_dataloader_config.params.batch_size=1","runner.params.eval_dataloader_config.params.num_workers=0","enable_jax_cache=true","device.jax_cache_dir=/data2/xuzhangbo/.cache/solaris-jax","shuffle_bots=false",f"hydra.run.dir={root}/hydra"]
write(root/"logs/command.json",cmd)
raw_dir=root/f"solaris_raw/{name}/eval_race_condition"
expected_raw=[raw_dir/f"video_{index}_side_by_side.mp4" for index in range(len(cases))]
if not all(path.is_file() and path.stat().st_size > 0 for path in expected_raw):
    with (root/"logs/inference.log").open("w") as log: result=subprocess.run(cmd,cwd=sol,env=env,stdout=log,stderr=subprocess.STDOUT)
    (root/"logs/inference.exitcode").write_text(str(result.returncode)+"\n");result.check_returncode()
checks=[]
for index,case in enumerate(cases):
    raw=raw_dir/f"video_{index}_side_by_side.mp4"
    dest=root/"solaris"/case["case_id"];dest.mkdir(parents=True,exist_ok=True)
    for role,y in (("alpha",0),("bravo",352)):
        out=dest/f"{role}_generated.mp4"
        run(["ffmpeg","-nostdin","-v","error","-y","-i",raw,"-vf",f"crop=640:352:640:{y}","-frames:v",frames,"-c:v","libx264","-threads","4","-crf","18","-movflags","+faststart",out])
        info=json.loads(subprocess.check_output(["ffprobe","-v","error","-count_frames","-select_streams","v:0","-show_entries","stream=width,height,r_frame_rate,nb_read_frames","-of","json",out]))["streams"][0]
        assert (info["width"],info["height"],info["r_frame_rate"],int(info["nb_read_frames"]))==(640,352,"20/1",frames)
        checks.append({"case_id":case["case_id"],"role":role,**info})
write(root/"inference_complete.json",{"status":"complete","gpu":gpu,"gt_collected":False,"checks":checks})
