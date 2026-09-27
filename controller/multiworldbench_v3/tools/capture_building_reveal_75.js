'use strict';
// Record native GT for the self-contained 75-case V3 building-reveal suite.
// RCON initializes and validates state. The formal reveal action is native X11/XTest input.
const fs=require('fs'), path=require('path'), crypto=require('crypto');
const {spawn,execFileSync}=require('child_process');
const readline=require('readline');
const {Rcon}=require('rcon-client');
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
const nowMs=()=>Date.now();
function sha(p){return crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');}
function minecraftWindow(display){const out=execFileSync('/usr/bin/xwininfo',['-root','-tree'],{env:{...process.env,DISPLAY:display},encoding:'utf8'});const m=out.match(/\n\s+(0x[0-9a-f]+)\s+"Minecraft/);if(!m)throw new Error(`Minecraft window not found on ${display}`);return m[1];}
function startSender(repo,display,windowId){const child=spawn('python3',[path.join(repo,'controller/style_x11_native_action_server.py'),display,windowId],{stdio:['pipe','pipe','inherit']});const rl=readline.createInterface({input:child.stdout});const pending=[];rl.on('line',line=>{const p=pending.shift();if(!p)return;try{p.resolve(JSON.parse(line));}catch(e){p.reject(e);}});const raw=item=>new Promise((resolve,reject)=>{pending.push({resolve,reject});child.stdin.write(JSON.stringify(item)+'\n');});return{ready:raw({focus:true}),focus:()=>raw({focus:true}),release:()=>raw({release_all:true}),send:buttons=>raw({dx:0,dy:0,held_keys:[],held_buttons:buttons}),close:()=>{child.stdin.end();child.kill('SIGTERM');}};}
function startVideo(display,out){const child=spawn('ffmpeg',['-nostdin','-hide_banner','-loglevel','warning','-y','-use_wallclock_as_timestamps','1','-f','x11grab','-video_size','1280x720','-framerate','20','-i',display,'-copyts','-vsync','0','-enc_time_base','1:1000','-frames:v','160','-c:v','libx264','-threads','2','-preset','ultrafast','-crf','18','-bf','0','-pix_fmt','yuv420p',out],{stdio:['ignore','ignore','pipe']});let err='';child.stderr.on('data',c=>err+=c);return{started_epoch_ms:nowMs(),done:new Promise((resolve,reject)=>child.once('close',code=>code===0?resolve():reject(new Error(`ffmpeg ${display} ${code}: ${err}`))))};}
function normalize(input,out){execFileSync('ffmpeg',['-nostdin','-hide_banner','-loglevel','error','-y','-i',input,'-vf','scale=640:360:flags=lanczos,crop=640:352:0:4,fps=20,trim=end_frame=129,setpts=PTS-STARTPTS','-frames:v','129','-an','-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p',out]);}
function firstPair(a,b,out){execFileSync('ffmpeg',['-nostdin','-hide_banner','-loglevel','error','-y','-i',a,'-i',b,'-filter_complex','[0:v]select=eq(n\\,0)[a];[1:v]select=eq(n\\,0)[b];[a][b]hstack=inputs=2[v]','-map','[v]','-frames:v','1',out]);}
function pairVideo(a,b,out){execFileSync('ffmpeg',['-nostdin','-hide_banner','-loglevel','error','-y','-i',a,'-i',b,'-filter_complex','[0:v][1:v]hstack=inputs=2[v]','-map','[v]','-an','-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p',out]);}
function actionRow(frame,epoch,result,active){return{frame_index:frame,epochTime:epoch/1000,epoch_ms:epoch,phase:frame<21?'PRE_ACTION':(active?'ACTION':'POST_ACTION'),primitive:active?'ATTACK_HOLD':'-',control_source:'native_x11_xtest',action:{forward:false,back:false,left:false,right:false,jump:false,sprint:false,sneak:false,attack:active,use:false,mine:active,place_block:false,place_entity:false,mount:false,dismount:false,'hotbar.1':false,'hotbar.2':false,'hotbar.3':false,'hotbar.4':false,'hotbar.5':false,'hotbar.6':false,'hotbar.7':false,'hotbar.8':false,'hotbar.9':false,camera:[0,0]},native_input:{device:'X11_XTest',held_buttons:result.held_buttons||[],sent_epoch_ms:result.sent_epoch_ms}};}
function gammaAction(rows){return{keyboard:rows.map(row=>{const v=Array(23).fill(0);if(row.action.attack)v[19]=1;return v;}),camera:rows.map(()=>[0,0])};}
function validExisting(caseDir,assetDir){try{const m=JSON.parse(fs.readFileSync(path.join(caseDir,'case_manifest.json')));return m.validation.valid&&['alpha_gt.mp4','bravo_gt.mp4','gt_pair.mp4'].every(f=>fs.existsSync(path.join(caseDir,f)))&&['first_frame.png','action_left.json','action_right.json','alpha_actions.json','bravo_actions.json','prompt.txt','metadata.json'].every(f=>fs.existsSync(path.join(assetDir,f)));}catch{return false;}}
async function main(){
  if(!process.env.RCON_PASSWORD)throw new Error('RCON_PASSWORD must be set');
  const repo=path.resolve(process.argv[2]||'.'),run=path.resolve(process.argv[3]||''),caseFile=path.resolve(process.argv[4]||'');
  const limitArg=process.argv.indexOf('--limit'),limit=limitArg>=0?Number(process.argv[limitArg+1]):null;
  if(!run||!caseFile)throw new Error('usage: node capture_building_reveal_75.js REPO RUN CASE_FILE [--limit N]');
  const suite=JSON.parse(fs.readFileSync(caseFile));const selected=limit?suite.cases.slice(0,limit):suite.cases;
  const gtRoot=path.join(run,'models/gt/reveal'),assetRoot=path.join(run,'assets/reveal');fs.mkdirSync(gtRoot,{recursive:true});fs.mkdirSync(assetRoot,{recursive:true});
  const rcon=await Rcon.connect({host:'127.0.0.1',port:25721,password:process.env.RCON_PASSWORD,timeout:30000});
  const senders={alpha:startSender(repo,':130',minecraftWindow(':130')),bravo:startSender(repo,':131',minecraftWindow(':131'))};await Promise.all([senders.alpha.ready,senders.bravo.ready]);
  const manifest={schema_version:'mwb.v3.building_reveal.native_gt.v1',case_file:caseFile,protocol:'native Minecraft X11/XTest reveal; RCON initialization and validation only',fps:20,frames_per_case:129,cases:[]};
  try{
    const online=await rcon.send('list');if(!online.includes('CameraAlpha')||!online.includes('CameraBravo'))throw new Error(`camera clients unavailable: ${online}`);
    for(const spec of selected){
      const cid=spec.case_id,caseDir=path.join(gtRoot,cid),assetDir=path.join(assetRoot,cid);
      if(validExisting(caseDir,assetDir)){const cm=JSON.parse(fs.readFileSync(path.join(caseDir,'case_manifest.json')));manifest.cases.push(cm);console.log(`${cid} skip valid`);continue;}
      for(const d of [caseDir,assetDir])if(fs.existsSync(d))execFileSync('/bin/rm',['-rf','--',d]);fs.mkdirSync(caseDir,{recursive:true});fs.mkdirSync(assetDir,{recursive:true});
      const initLog=path.join(caseDir,'initialization_rcon.jsonl'),validationLog=path.join(caseDir,'validation_rcon.jsonl');
      const sendInit=async command=>{const response=await rcon.send(command);fs.appendFileSync(initLog,JSON.stringify({epoch_ms:nowMs(),command,response:String(response)})+'\n');if(/unknown or incomplete|incorrect argument|expected integer|not a valid|position is not loaded|too many blocks|no player was found/i.test(String(response)))throw new Error(`${command} -> ${response}`);return response;};
      const validate=async command=>{const response=await rcon.send(command);fs.appendFileSync(validationLog,JSON.stringify({epoch_ms:nowMs(),command,response:String(response)})+'\n');return Boolean(String(response).trim());};
      const cx=spec.world_x_offset;
      await rcon.send('forceload remove all');await sendInit(`forceload add ${cx-36} -31 ${cx+36} 31`);
      for(const command of ['difficulty peaceful','gamerule doDaylightCycle false','gamerule doWeatherCycle false','weather clear','time set noon','kill @e[type=!player]'])await sendInit(command);
      await rcon.send('team add mwb3reveal');await sendInit('team modify mwb3reveal collisionRule never');await sendInit('team modify mwb3reveal nametagVisibility never');await sendInit('team join mwb3reveal CameraAlpha');await sendInit('team join mwb3reveal CameraBravo');
      for(const command of spec.setup_commands)await sendInit(command);
      for(const [role,name] of [['alpha','CameraAlpha'],['bravo','CameraBravo']]){const pose=spec.initial_poses[role];await sendInit(`gamemode survival ${name}`);await sendInit(`clear ${name}`);await sendInit(`tp ${name} ${pose.position.join(' ')} ${pose.native_yaw_deg} ${pose.native_pitch_deg}`);}
      await Promise.all([senders.alpha.focus(),senders.bravo.focus()]);await sleep(150);await Promise.all([senders.alpha.release(),senders.bravo.release()]);
      await sendInit(`setblock ${spec.reveal.occluder.position.join(' ')} minecraft:dirt`);
      for(const [role,name] of [['alpha','CameraAlpha'],['bravo','CameraBravo']]){const pose=spec.initial_poses[role];await sendInit(`clear ${name}`);await sendInit(`tp ${name} ${pose.position.join(' ')} ${pose.native_yaw_deg} ${pose.native_pitch_deg}`);}
      await Promise.all([senders.alpha.send([]),senders.bravo.send([])]);await sleep(1200);
      const rawA=path.join(caseDir,'alpha_raw.mp4'),rawB=path.join(caseDir,'bravo_raw.mp4');const videoA=startVideo(':130.0',rawA),videoB=startVideo(':131.0',rawB);
      const start=nowMs()+250,actions={alpha:[],bravo:[]};
      for(let frame=0;frame<129;frame++){const remaining=start+frame*50-nowMs();if(remaining>0)await sleep(remaining);const active=frame>=21&&frame<=60;const epoch=nowMs();const [a,b]=await Promise.all([senders.alpha.send(active?[1]:[]),senders.bravo.send([])]);actions.alpha.push(actionRow(frame,epoch,a,active));actions.bravo.push(actionRow(frame,epoch,b,false));}
      await Promise.all([senders.alpha.send([]),senders.bravo.send([])]);await Promise.all([videoA.done,videoB.done]);
      const [ox,oy,oz]=spec.reveal.occluder.position;
      const occluderRemoved=await validate(`execute if block ${ox} ${oy} ${oz} minecraft:air run time query gametime`);
      const apertureIntact=await validate(`execute if block ${ox} ${oy-1} ${oz} minecraft:polished_deepslate run time query gametime`)&&await validate(`execute if block ${ox} ${oy+1} ${oz} minecraft:polished_deepslate run time query gametime`);
      const poses={};for(const [role,name] of [['alpha','CameraAlpha'],['bravo','CameraBravo']]){const p=spec.initial_poses[role].position;poses[role]=await validate(`execute positioned ${p.join(' ')} if entity @a[name=${name},distance=..0.6] run time query gametime`);}
      const valid=occluderRemoved&&apertureIntact&&poses.alpha&&poses.bravo;if(!valid)throw new Error(`${cid} validation failed ${JSON.stringify({occluderRemoved,apertureIntact,poses})}`);
      fs.writeFileSync(path.join(caseDir,'alpha_actions.json'),JSON.stringify(actions.alpha,null,2)+'\n');fs.writeFileSync(path.join(caseDir,'bravo_actions.json'),JSON.stringify(actions.bravo,null,2)+'\n');
      normalize(rawA,path.join(caseDir,'alpha_gt.mp4'));normalize(rawB,path.join(caseDir,'bravo_gt.mp4'));fs.unlinkSync(rawA);fs.unlinkSync(rawB);pairVideo(path.join(caseDir,'alpha_gt.mp4'),path.join(caseDir,'bravo_gt.mp4'),path.join(caseDir,'gt_pair.mp4'));
      for(const role of ['alpha','bravo'])fs.copyFileSync(path.join(caseDir,`${role}_actions.json`),path.join(assetDir,`${role}_actions.json`));
      firstPair(path.join(caseDir,'alpha_gt.mp4'),path.join(caseDir,'bravo_gt.mp4'),path.join(assetDir,'first_frame.png'));
      fs.writeFileSync(path.join(assetDir,'action_left.json'),JSON.stringify(gammaAction(actions.alpha),null,2)+'\n');fs.writeFileSync(path.join(assetDir,'action_right.json'),JSON.stringify(gammaAction(actions.bravo),null,2)+'\n');
      fs.writeFileSync(path.join(assetDir,'prompt.txt'),`Minecraft shared-world reveal: player A destroys one dirt occluder and reveals the ${spec.building_name}; stationary player B sees player A and the same building throughout.\n`);
      const files=['first_frame.png','action_left.json','action_right.json','alpha_actions.json','bravo_actions.json','prompt.txt'];
      const metadata={schema_version:'mwb.v3.asset.v1',case_id:cid,group:'reveal',self_contained:true,has_native_gt:true,scene_id:spec.scene_id,building_id:spec.building_id,relative_agent_angle_deg:spec.relative_agent_angle_deg,frames:129,check_frames:[80,100,120],files_sha256:Object.fromEntries(files.map(f=>[f,sha(path.join(assetDir,f))]))};fs.writeFileSync(path.join(assetDir,'metadata.json'),JSON.stringify(metadata,null,2)+'\n');
      const cm={case_id:cid,scene_id:spec.scene_id,building_id:spec.building_id,relative_agent_angle_deg:spec.relative_agent_angle_deg,native_action:spec.reveal.required_native_action,recording:{fps:20,frames:129,action_start_epoch_ms:start,alpha_video_start_epoch_ms:videoA.started_epoch_ms,bravo_video_start_epoch_ms:videoB.started_epoch_ms},validation:{occluder_removed:occluderRemoved,aperture_intact:apertureIntact,poses_stationary:poses,valid}};
      fs.writeFileSync(path.join(caseDir,'case_manifest.json'),JSON.stringify(cm,null,2)+'\n');manifest.cases.push(cm);fs.writeFileSync(path.join(gtRoot,'manifest.json'),JSON.stringify(manifest,null,2)+'\n');console.log(`${cid} valid=${valid}`);
    }
    fs.writeFileSync(path.join(gtRoot,'manifest.json'),JSON.stringify(manifest,null,2)+'\n');fs.writeFileSync(path.join(run,limit?'.gt_smoke_complete':'.gt_capture_complete'),JSON.stringify({selected:selected.length,captured:manifest.cases.length,all_valid:manifest.cases.length===selected.length&&manifest.cases.every(c=>c.validation.valid)},null,2)+'\n');
  }finally{senders.alpha.close();senders.bravo.close();await rcon.end().catch(()=>{});}
}
main().catch(e=>{console.error(e.stack||e.message);process.exit(1);});
