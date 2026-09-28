'use strict';
// Capture case-defined shared-state persistence trajectories with native Minecraft input.
// RCON is used only for scene initialization and exact post-action validation.
const fs=require('fs'), path=require('path'), crypto=require('crypto');
const {spawn,execFileSync}=require('child_process');
const readline=require('readline');
const {Rcon}=require('rcon-client');
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const nowMs=()=>Date.now();
const degToRad=d=>d*Math.PI/180;
const wrapDeg=d=>((d+540)%360)-180;
const clamp=(x,lo,hi)=>Math.max(lo,Math.min(hi,x));

function minecraftWindow(display){
  const out=execFileSync('/usr/bin/xwininfo',['-root','-tree'],{env:{...process.env,DISPLAY:display},encoding:'utf8'});
  const m=out.match(/\n\s+(0x[0-9a-f]+)\s+"Minecraft/); if(!m)throw new Error(`Minecraft window not found on ${display}`); return m[1];
}
function startX11Sender(repo,display,windowId){
  const child=spawn('python3',[path.join(repo,'controller/style_x11_native_action_server.py'),display,windowId],{stdio:['pipe','pipe','inherit']});
  const lines=readline.createInterface({input:child.stdout}), pending=[];
  lines.on('line',line=>{const item=pending.shift();if(!item)return;try{item.resolve(JSON.parse(line));}catch(e){item.reject(e);}});
  const send=item=>new Promise((resolve,reject)=>{pending.push({resolve,reject});child.stdin.write(JSON.stringify(item)+'\n');});
  return {ready:send({focus:true}),focus:()=>send({focus:true}),release:()=>send({release_all:true}),send:(dx,dy,keys=[],buttons=[])=>send({dx,dy,held_keys:keys,held_buttons:buttons}),close:()=>{child.stdin.end();child.kill('SIGTERM');}};
}
function startVideo(display,output,frames){
  const child=spawn('ffmpeg',['-nostdin','-hide_banner','-loglevel','warning','-y','-use_wallclock_as_timestamps','1','-f','x11grab','-video_size','1280x720','-framerate','20','-i',display,'-copyts','-vsync','0','-enc_time_base','1:1000','-frames:v',String(frames),'-c:v','libx264','-threads','2','-preset','ultrafast','-crf','18','-bf','0','-pix_fmt','yuv420p',output],{stdio:['ignore','ignore','pipe']});
  let stderr=''; child.stderr.on('data',c=>stderr+=c.toString()); const startedEpochMs=nowMs();
  const done=new Promise((resolve,reject)=>child.once('close',code=>code===0?resolve():reject(new Error(`ffmpeg ${display} exited ${code}: ${stderr}`))));
  return {startedEpochMs,done};
}
function normalizeVideo(input,output,frames){
  execFileSync('ffmpeg',['-nostdin','-hide_banner','-loglevel','error','-y','-i',input,'-vf',`scale=640:360:flags=lanczos,crop=640:352:0:4,fps=20,trim=end_frame=${frames},setpts=PTS-STARTPTS`,'-frames:v',String(frames),'-an','-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p',output]);
}
function pairFirst(a,b,out){execFileSync('ffmpeg',['-nostdin','-hide_banner','-loglevel','error','-y','-i',a,'-i',b,'-filter_complex','[0:v]select=eq(n\\,0)[a];[1:v]select=eq(n\\,0)[b];[a][b]hstack=inputs=2[v]','-map','[v]','-frames:v','1',out]);}
function sha(p){return crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');}
function parseRotation(response){const m=String(response).match(/\[\s*([-+0-9.eE]+)f,\s*([-+0-9.eE]+)f\s*\]/);if(!m)throw new Error(`rotation parse failed: ${response}`);return {yaw:Number(m[1]),pitch:Number(m[2])};}
function bearingError(yaw,position,target){const dx=target[0]+.5-position[0],dz=target[2]+.5-position[2];const desired=Math.atan2(-dx,dz)*180/Math.PI;return Math.abs(wrapDeg(yaw-desired));}
function mouseDx(current,target){return Math.round(clamp(wrapDeg(target-current)*6.7,-100,100));}
function phaseAt(spec,frame){const p=spec.timeline.find(x=>frame>=x.range[0]&&frame<x.range[1]);return p?p.phase:'unknown';}
function gammaAction(rows,frames){const keyboard=[],camera=[];for(let i=0;i<frames+12;i++){const row=rows[Math.min(i,rows.length-1)],a=i<rows.length?row.action:{};const v=Array(23).fill(0);if(a.forward)v[0]=1;if(a.back)v[1]=1;if(a.left)v[2]=1;if(a.right)v[3]=1;if(a.jump)v[4]=1;if(a.sprint)v[5]=1;if(a.sneak)v[6]=1;if(a.attack)v[19]=1;if(a.use)v[20]=1;keyboard.push(v);camera.push(a.camera||[0,0]);}return {keyboard,camera};}
function complete(caseDir,assetDir){try{const d=JSON.parse(fs.readFileSync(path.join(caseDir,'case_manifest.json')));return d.validation.valid&&['alpha_gt.mp4','bravo_gt.mp4','alpha_actions.json','bravo_actions.json'].every(f=>fs.existsSync(path.join(caseDir,f)))&&['first_frame.png','action_left.json','action_right.json','alpha_actions.json','bravo_actions.json','prompt.txt','metadata.json'].every(f=>fs.existsSync(path.join(assetDir,f)));}catch{return false;}}

async function main(){
  if(!process.env.RCON_PASSWORD)throw new Error('RCON_PASSWORD must be set');
  const repo=path.resolve(process.argv[2]||'.'),run=path.resolve(process.argv[3]||''),caseFile=path.resolve(process.argv[4]||'');
  if(!run||!caseFile)throw new Error('usage: node capture_shared_state_persistence.js REPO RUN CASE_FILE');
  const suite=JSON.parse(fs.readFileSync(caseFile)),frames=181;
  const gtRoot=path.join(run,'models/gt/interaction'),assetRoot=path.join(run,'assets/interaction'); fs.mkdirSync(gtRoot,{recursive:true});fs.mkdirSync(assetRoot,{recursive:true});
  const rcon=await Rcon.connect({host:'127.0.0.1',port:25721,password:process.env.RCON_PASSWORD,timeout:30000});
  const windows={alpha:minecraftWindow(':130'),bravo:minecraftWindow(':131')};
  const senders={alpha:startX11Sender(repo,':130',windows.alpha),bravo:startX11Sender(repo,':131',windows.bravo)}; await Promise.all([senders.alpha.ready,senders.bravo.ready]);
  const manifest={schema_version:'mwb.v3.shared_state_persistence.native_gt.v1',case_file:caseFile,protocol:'native Minecraft X11/XTest actions; RCON initialization and validation only',fps:20,frames_per_case:frames,cases:[]};
  try{
    const online=await rcon.send('list');if(!online.includes('CameraAlpha')||!online.includes('CameraBravo'))throw new Error(`camera clients unavailable: ${online}`);
    for(const spec of suite.cases){
      const cid=spec.case_id,caseDir=path.join(gtRoot,cid),assetDir=path.join(assetRoot,cid);
      if(complete(caseDir,assetDir)){const cm=JSON.parse(fs.readFileSync(path.join(caseDir,'case_manifest.json')));manifest.cases.push(cm);console.log(`${cid} skip valid`);continue;}
      for(const d of [caseDir,assetDir])if(fs.existsSync(d))execFileSync('/bin/rm',['-rf','--',d]);fs.mkdirSync(caseDir,{recursive:true});fs.mkdirSync(assetDir,{recursive:true});
      const initLog=path.join(caseDir,'initialization_rcon.jsonl'),validLog=path.join(caseDir,'validation_rcon.jsonl');
      const sendInit=async command=>{const response=await rcon.send(command);fs.appendFileSync(initLog,JSON.stringify({epoch_ms:nowMs(),command,response})+'\n');if(/unknown or incomplete|incorrect argument|not a valid|no player was found|position is not loaded|too many blocks/i.test(String(response)))throw new Error(`${command} -> ${response}`);return response;};
      const sendValidate=async command=>{const response=await rcon.send(command);fs.appendFileSync(validLog,JSON.stringify({epoch_ms:nowMs(),command,response})+'\n');return response;};
      const [tx,ty,tz]=spec.scene_spec.origin;
      await sendInit(`forceload add ${tx-12} ${tz-12} ${tx+12} ${tz+12}`);await sendInit('difficulty peaceful');await sendInit('gamerule doDaylightCycle false');await sendInit('gamerule doWeatherCycle false');await sendInit('weather clear');await sendInit('time set noon');
      await rcon.send('team add mwb3persist');await sendInit('team modify mwb3persist collisionRule never');await sendInit('team modify mwb3persist nametagVisibility never');await sendInit('team join mwb3persist CameraAlpha');await sendInit('team join mwb3persist CameraBravo');
      await sendInit(`fill ${tx-10} ${ty-1} ${tz-8} ${tx+10} ${ty-1} ${tz+10} ${spec.scene_spec.floor_material}`);await sendInit(`fill ${tx-10} ${ty} ${tz-8} ${tx+10} ${ty+7} ${tz+10} minecraft:air`);await sendInit(`execute positioned ${tx} ${ty} ${tz} run kill @e[type=minecraft:item,distance=..32]`);
      for(const g of spec.scene_spec.geometry_relative){const [x1,y1,z1,x2,y2,z2,mat]=g;await sendInit(`fill ${tx+x1} ${ty+y1} ${tz+z1} ${tx+x2} ${ty+y2} ${tz+z2} ${mat}`);}
      await sendInit(`fill ${tx-4} ${ty-1} ${tz} ${tx+4} ${ty-1} ${tz} ${spec.scene_spec.cross_material}`);await sendInit(`fill ${tx} ${ty-1} ${tz-4} ${tx} ${ty-1} ${tz+4} ${spec.scene_spec.cross_material}`);await sendInit(`setblock ${tx} ${ty-1} ${tz} ${spec.scene_spec.marker_material}`);
      const initial=spec.operation==='destroy'?'coarse_dirt':'air';await sendInit(`setblock ${tx} ${ty} ${tz} minecraft:${initial}`);
      for(const [role,name] of [['alpha','CameraAlpha'],['bravo','CameraBravo']]){const pose=spec.initial_conditions[role].pose;await sendInit(`gamemode survival ${name}`);await sendInit(`clear ${name}`);if(role==='bravo'&&spec.operation==='place')await sendInit(`item replace entity ${name} weapon.mainhand with minecraft:coarse_dirt 1`);if(role==='bravo'&&spec.operation==='destroy')await sendInit(`item replace entity ${name} weapon.mainhand with minecraft:diamond_shovel 1`);await sendInit(`tp ${name} ${pose.position.join(' ')} ${pose.native_yaw_deg} ${pose.native_pitch_deg}`);}
      await Promise.all([senders.alpha.focus(),senders.bravo.focus()]);await sleep(250);await Promise.all([senders.alpha.release(),senders.bravo.release()]);await sleep(900);
      const rawA=path.join(caseDir,'alpha_raw.mp4'),rawB=path.join(caseDir,'bravo_raw.mp4');const videoA=startVideo(':130.0',rawA,205),videoB=startVideo(':131.0',rawB,205);
      const actionStartEpochMs=nowMs()+250,actions={alpha:[],bravo:[]},rotationEvidence=[];
      const initYaw={alpha:spec.initial_conditions.alpha.pose.native_yaw_deg,bravo:spec.initial_conditions.bravo.pose.native_yaw_deg};
      let current={alpha:{yaw:initYaw.alpha,pitch:spec.initial_conditions.alpha.pose.native_pitch_deg},bravo:{yaw:initYaw.bravo,pitch:spec.initial_conditions.bravo.pose.native_pitch_deg}};
      const sign=spec.variant===0?1:-1;
      for(let frame=0;frame<frames;frame++){
        const wait=actionStartEpochMs+frame*50-nowMs();if(wait>0)await sleep(wait);
        // Minecraft's configured sensitivity maps about 6.7 mouse pixels to one yaw degree.
        // Use an open-loop 110-degree sweep over 20 frames. RCON rotation reads trail the
        // native input by roughly one tick, so feedback control overshoots at segment ends.
        let adx=0,bdx=0;if(frame>=20&&frame<40)adx=37*sign;else if(frame>=130&&frame<150)adx=-37*sign;if(frame>=90&&frame<110)bdx=-37*sign;
        const bButtons=spec.operation==='place'&&frame===60?[3]:spec.operation==='destroy'&&frame>=60&&frame<68?[1]:[];
        const [ar,br]=await Promise.all([senders.alpha.send(adx,0,[],[]),senders.bravo.send(bdx,0,[],bButtons)]);
        const [aRot,bRot]=await Promise.all([rcon.send('data get entity CameraAlpha Rotation'),rcon.send('data get entity CameraBravo Rotation')]);const next={alpha:parseRotation(aRot),bravo:parseRotation(bRot)};
        const phase=phaseAt(spec,frame),epoch=nowMs();
        for(const role of ['alpha','bravo']){const delta=wrapDeg(next[role].yaw-current[role].yaw),buttons=role==='bravo'?bButtons:[];actions[role].push({frame_index:frame,epochTime:epoch/1000,epoch_ms:epoch,phase,primitive:buttons.includes(1)?'ATTACK_HOLD':buttons.includes(3)?'USE_ONCE':(role==='alpha'?adx:bdx)?'CAMERA_YAW':'-',control_source:'native_x11_xtest',action:{forward:false,back:false,left:false,right:false,jump:false,sprint:false,sneak:false,attack:buttons.includes(1),use:buttons.includes(3),mine:buttons.includes(1),place_block:buttons.includes(3),camera:[degToRad(delta),0]},native_input:{device:'X11_XTest',dx_px:role==='alpha'?adx:bdx,dy_px:0,held_buttons:buttons,executed_epoch_ms:(role==='alpha'?ar:br).sent_epoch_ms},measured_rotation_deg:{yaw:next[role].yaw,pitch:next[role].pitch}});}
        if([10,50,80,120,145,160,170,180].includes(frame))rotationEvidence.push({frame,alpha:{...next.alpha,bearing_error_deg:bearingError(next.alpha.yaw,spec.initial_conditions.alpha.pose.position,spec.target_cell)},bravo:{...next.bravo,bearing_error_deg:bearingError(next.bravo.yaw,spec.initial_conditions.bravo.pose.position,spec.target_cell)}});
        current=next;
      }
      await Promise.all([senders.alpha.send(0,0,[],[]),senders.bravo.send(0,0,[],[])]);await Promise.all([videoA.done,videoB.done]);
      const finalExpected=spec.operation==='place'?'coarse_dirt':'air';const tr=await sendValidate(`execute if block ${tx} ${ty} ${tz} minecraft:${finalExpected} run time query gametime`);const sr=await sendValidate(`execute if block ${tx} ${ty-1} ${tz} ${spec.scene_spec.marker_material} run time query gametime`);
      const byFrame=Object.fromEntries(rotationEvidence.map(x=>[x.frame,x])),stateValid=Boolean(String(tr).trim()),supportValid=Boolean(String(sr).trim());
      const visibilityValid=byFrame[50].alpha.bearing_error_deg>=80&&byFrame[120].alpha.bearing_error_deg>=80&&byFrame[120].bravo.bearing_error_deg>=80&&byFrame[160].alpha.bearing_error_deg<=12;
      const valid=stateValid&&supportValid&&visibilityValid;if(!valid)throw new Error(`${cid} validation failed state=${stateValid} support=${supportValid} visibility=${visibilityValid} evidence=${JSON.stringify(rotationEvidence)}`);
      for(const role of ['alpha','bravo'])fs.writeFileSync(path.join(caseDir,`${role}_actions.json`),JSON.stringify(actions[role],null,2)+'\n');normalizeVideo(rawA,path.join(caseDir,'alpha_gt.mp4'),frames);normalizeVideo(rawB,path.join(caseDir,'bravo_gt.mp4'),frames);fs.unlinkSync(rawA);fs.unlinkSync(rawB);
      for(const role of ['alpha','bravo'])fs.copyFileSync(path.join(caseDir,`${role}_actions.json`),path.join(assetDir,`${role}_actions.json`));pairFirst(path.join(caseDir,'alpha_gt.mp4'),path.join(caseDir,'bravo_gt.mp4'),path.join(assetDir,'first_frame.png'));fs.writeFileSync(path.join(assetDir,'action_left.json'),JSON.stringify(gammaAction(actions.alpha,189),null,2)+'\n');fs.writeFileSync(path.join(assetDir,'action_right.json'),JSON.stringify(gammaAction(actions.bravo,189),null,2)+'\n');fs.writeFileSync(path.join(assetDir,'prompt.txt'),`Minecraft shared-state persistence: both players observe the marked cross center; Alpha turns away; Bravo ${spec.operation}s exactly one block; Bravo turns away; Alpha returns and verifies the persistent change.\n`);
      const files=['first_frame.png','action_left.json','action_right.json','alpha_actions.json','bravo_actions.json','prompt.txt'];const metadata={schema_version:'mwb.v3.asset.persistence.v1',case_id:cid,group:'interaction',self_contained:true,has_native_gt:true,scene:spec.scene,operation:spec.operation,expected_change_count:1,target_cell:spec.target_cell,files_sha256:Object.fromEntries(files.map(f=>[f,sha(path.join(assetDir,f))]))};fs.writeFileSync(path.join(assetDir,'metadata.json'),JSON.stringify(metadata,null,2)+'\n');
      const cm={case_id:cid,operation:spec.operation,target_cell:spec.target_cell,native_action:spec.native_action_protocol,recording:{fps:20,frames,action_start_epoch_ms:actionStartEpochMs,alpha_video_start_epoch_ms:videoA.startedEpochMs,bravo_video_start_epoch_ms:videoB.startedEpochMs},rotation_evidence:rotationEvidence,validation:{target_expected:`minecraft:${finalExpected}`,target_valid:stateValid,support_valid:supportValid,visibility_valid:visibilityValid,valid}};fs.writeFileSync(path.join(caseDir,'case_manifest.json'),JSON.stringify(cm,null,2)+'\n');manifest.cases.push(cm);fs.writeFileSync(path.join(gtRoot,'manifest.json'),JSON.stringify(manifest,null,2)+'\n');console.log(`${cid} ${spec.operation} valid=${valid}`);
    }
    fs.writeFileSync(path.join(gtRoot,'manifest.json'),JSON.stringify(manifest,null,2)+'\n');fs.writeFileSync(path.join(run,'.gt_capture_complete'),JSON.stringify({case_count:manifest.cases.length,all_valid:manifest.cases.length===suite.cases.length&&manifest.cases.every(c=>c.validation.valid)},null,2)+'\n');
  }finally{senders.alpha.close();senders.bravo.close();await rcon.end().catch(()=>{});}
}
main().catch(e=>{console.error(e.stack||e.message);process.exit(1);});
