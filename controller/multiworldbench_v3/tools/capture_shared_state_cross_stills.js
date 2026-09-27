'use strict';
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { execFileSync } = require('child_process');
const { Rcon } = require('rcon-client');

const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
const sha256 = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');

async function main() {
  if (!process.env.RCON_PASSWORD) throw new Error('RCON_PASSWORD must be set');
  const repo = path.resolve(process.argv[2] || '.');
  const logDir = path.resolve(process.argv[3] || '/tmp/mwb3-shared-state-cross-capture');
  const suite = JSON.parse(fs.readFileSync(path.join(repo, 'MultiWorldBench/v3/cases/shared_world/shared_state_synchronization.json')));
  const assetsRoot = path.join(repo, 'MultiWorldBench/v3/assets/interaction');
  fs.mkdirSync(logDir, {recursive:true});
  const logPath = path.join(logDir, 'initialization.jsonl');
  fs.writeFileSync(logPath, '');
  const rcon = await Rcon.connect({host:'127.0.0.1',port:25721,password:process.env.RCON_PASSWORD,timeout:10000});
  const send = async command => {
    const response = await rcon.send(command);
    fs.appendFileSync(logPath, JSON.stringify({epoch_ms:Date.now(),command,response})+'\n');
    if (/unknown or incomplete|incorrect argument|not a valid|no player was found|position is not loaded|too many blocks/i.test(String(response))) {
      throw new Error(`${command} -> ${response}`);
    }
    return response;
  };
  try {
    const online = await rcon.send('list');
    if (!online.includes('CameraAlpha') || !online.includes('CameraBravo')) throw new Error(`camera clients unavailable: ${online}`);
    await send('forceload add 610 -8 634 10');
    await send('difficulty peaceful');
    await send('gamerule doDaylightCycle false');
    await send('gamerule doWeatherCycle false');
    await send('weather clear');
    await send('time set noon');
    await rcon.send('team add mwb3sync');
    await send('team modify mwb3sync collisionRule never');
    await send('team modify mwb3sync nametagVisibility never');
    await send('team join mwb3sync CameraAlpha');
    await send('team join mwb3sync CameraBravo');
    const captures=[];
    for (const spec of suite.cases) {
      const cid=spec.case_id;
      const caseDir=path.join(assetsRoot,cid);
      const [tx,ty,tz]=spec.target_cell;
      await send(`fill ${tx-8} ${ty-1} ${tz-7} ${tx+8} ${ty-1} ${tz+9} minecraft:polished_andesite`);
      await send(`fill ${tx-8} ${ty} ${tz-7} ${tx+8} ${ty+7} ${tz+9} minecraft:air`);
      await send(`fill ${tx-4} ${ty-1} ${tz} ${tx+4} ${ty-1} ${tz} minecraft:white_concrete`);
      await send(`fill ${tx} ${ty-1} ${tz-4} ${tx} ${ty-1} ${tz+4} minecraft:white_concrete`);
      await send(`setblock ${tx} ${ty-1} ${tz} minecraft:red_concrete`);
      await send(`fill ${tx-8} ${ty-1} ${tz-7} ${tx+8} ${ty-1} ${tz-7} minecraft:deepslate_tiles`);
      await send(`fill ${tx-8} ${ty-1} ${tz+9} ${tx+8} ${ty-1} ${tz+9} minecraft:smooth_sandstone`);
      await send(`fill ${tx-8} ${ty-1} ${tz-7} ${tx-8} ${ty-1} ${tz+9} minecraft:oxidized_copper`);
      await send(`fill ${tx+8} ${ty-1} ${tz-7} ${tx+8} ${ty-1} ${tz+9} minecraft:bricks`);
      for (const [x,z,block] of [[tx-7,tz-6,'yellow_concrete'],[tx+7,tz-6,'cyan_concrete'],[tx-7,tz+8,'lime_concrete'],[tx+7,tz+8,'magenta_concrete']]) {
        await send(`fill ${x} ${ty} ${z} ${x} ${ty+2} ${z} minecraft:${block}`);
        await send(`setblock ${x} ${ty+3} ${z} minecraft:sea_lantern`);
      }
      await send(`setblock ${tx} ${ty} ${tz} minecraft:${spec.operation==='destroy'?'coarse_dirt':'air'}`);
      const actual={};
      for (const [role,name] of [['alpha','CameraAlpha'],['bravo','CameraBravo']]) {
        const pose=spec.initial_conditions[role].pose;
        await send(`gamemode survival ${name}`);
        await send(`clear ${name}`);
        if (spec.operation==='place' && pose.active) await send(`item replace entity ${name} weapon.mainhand with minecraft:coarse_dirt 64`);
        else if (spec.operation==='destroy' && pose.active) await send(`item replace entity ${name} weapon.mainhand with minecraft:diamond_pickaxe 1`);
        await send(`tp ${name} ${pose.position.join(' ')} ${pose.native_yaw_deg} ${pose.native_pitch_deg}`);
        actual[role]={position:await send(`data get entity ${name} Pos`),rotation:await send(`data get entity ${name} Rotation`),selected_item:await send(`data get entity ${name} SelectedItem`)};
      }
      await wait(1800);
      const rawA=path.join(logDir,`${cid}_alpha_raw.png`);
      const rawB=path.join(logDir,`${cid}_bravo_raw.png`);
      execFileSync('ffmpeg',['-nostdin','-v','error','-y','-f','x11grab','-video_size','1280x720','-i',':130.0','-frames:v','1',rawA]);
      execFileSync('ffmpeg',['-nostdin','-v','error','-y','-f','x11grab','-video_size','1280x720','-i',':131.0','-frames:v','1',rawB]);
      const first=path.join(caseDir,'first_frame.png');
      execFileSync('ffmpeg',['-nostdin','-v','error','-y','-i',rawA,'-i',rawB,'-filter_complex','[0:v]scale=640:360:flags=lanczos,crop=640:352:0:4[a];[1:v]scale=640:360:flags=lanczos,crop=640:352:0:4[b];[a][b]hstack=inputs=2[v]','-map','[v]','-frames:v','1',first]);
      fs.writeFileSync(path.join(caseDir,'prompt.txt'),'Minecraft gameplay. A and B observe a marked courtyard. Perform the planned single shared-world block interaction at the red center of the white ground cross.\n');
      const metadataPath=path.join(caseDir,'metadata.json');
      const metadata=JSON.parse(fs.readFileSync(metadataPath));
      metadata.scene='marked_cross_courtyard';
      metadata.expected_change_count=1;
      metadata.target_cell=spec.target_cell;
      metadata.capture={native_clients:true,resolution:[1280,352],cross_center:[tx,ty-1,tz]};
      for (const name of ['action_left.json','action_right.json','alpha_actions.json','bravo_actions.json','first_frame.png','prompt.txt']) metadata.files_sha256[name]=sha256(path.join(caseDir,name));
      fs.writeFileSync(metadataPath,JSON.stringify(metadata,null,2)+'\n');
      captures.push({case_id:cid,operation:spec.operation,active_roles:spec.active_roles,target_cell:spec.target_cell,expected_change_count:1,first_frame:first,first_frame_sha256:sha256(first),actual});
      console.log(`${cid} ${spec.operation} ${sha256(first).slice(0,12)}`);
    }
    const manifestPath=path.join(repo,'MultiWorldBench/v3/assets/manifest.json');
    const manifest=JSON.parse(fs.readFileSync(manifestPath));
    for (const entry of manifest.cases) if (entry.case_id.startsWith('MWB3-S-SS')) entry.metadata_sha256=sha256(path.join(repo,'MultiWorldBench/v3',entry.directory,'metadata.json'));
    fs.writeFileSync(manifestPath,JSON.stringify(manifest,null,2)+'\n');
    fs.writeFileSync(path.join(logDir,'capture_manifest.json'),JSON.stringify({protocol:'V3 shared-state marked cross conditioning stills',gt_collected:false,native_interaction_executed:false,case_count:captures.length,captures},null,2)+'\n');
  } finally { await rcon.end().catch(()=>{}); }
}
main().catch(error=>{console.error(error.stack||error.message);process.exit(1);});
