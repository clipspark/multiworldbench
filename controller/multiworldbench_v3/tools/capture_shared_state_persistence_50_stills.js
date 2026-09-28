'use strict';
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const {execFileSync} = require('child_process');
const {Rcon} = require('rcon-client');

const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
const sha256 = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const deg = radians => radians * 180 / Math.PI;
function lookAt(position, aim) {
  const dx = aim[0] - position[0];
  const dy = aim[1] - (position[1] + 1.62);
  const dz = aim[2] - position[2];
  return {yaw: deg(Math.atan2(-dx, dz)), pitch: -deg(Math.atan2(dy, Math.hypot(dx, dz)))};
}
function shot(display, output) {
  execFileSync('ffmpeg',['-nostdin','-v','error','-y','-f','x11grab','-video_size','1280x720','-i',display,'-frames:v','1',output]);
}
function pair(a,b,out) {
  execFileSync('ffmpeg',['-nostdin','-v','error','-y','-i',a,'-i',b,'-filter_complex','[0:v]scale=640:360:flags=lanczos,crop=640:352:0:4[a];[1:v]scale=640:360:flags=lanczos,crop=640:352:0:4[b];[a][b]hstack=inputs=2[v]','-map','[v]','-frames:v','1',out]);
}

const blockTypes = [
  {id:'minecraft:coarse_dirt',name_zh:'粗泥土',tool:'minecraft:diamond_shovel'},
  {id:'minecraft:oak_planks',name_zh:'橡木木板',tool:'minecraft:diamond_axe'},
  {id:'minecraft:red_wool',name_zh:'红色羊毛',tool:'minecraft:shears'},
  {id:'minecraft:gold_block',name_zh:'金块',tool:'minecraft:diamond_pickaxe'},
  {id:'minecraft:diamond_block',name_zh:'钻石块',tool:'minecraft:diamond_pickaxe'},
];
const scenes = [
  {id:'grass_oak',name_zh:'草地木构',floor:'minecraft:grass_block',cross:'minecraft:white_concrete',center:'minecraft:red_concrete',sample:blockTypes[4],geometry:[[-8,0,-7,-8,4,-7,'minecraft:oak_log'],[8,0,-7,8,4,-7,'minecraft:oak_log'],[-8,5,-7,8,5,-7,'minecraft:oak_log'],[-7,0,-7,-3,1,-7,'minecraft:mossy_cobblestone'],[3,0,-7,7,2,-7,'minecraft:oak_planks'],[-10,0,-2,-9,3,-2,'minecraft:birch_log'],[9,0,-3,10,2,-3,'minecraft:moss_block']]},
  {id:'sandstone_court',name_zh:'砂岩庭院',floor:'minecraft:smooth_sandstone',cross:'minecraft:cyan_concrete',center:'minecraft:black_concrete',sample:blockTypes[2],geometry:[[-6,0,-7,-6,5,-7,'minecraft:cut_sandstone'],[6,0,-7,6,5,-7,'minecraft:cut_sandstone'],[-6,6,-7,6,6,-7,'minecraft:smooth_sandstone'],[-10,0,-4,-8,2,-4,'minecraft:chiseled_sandstone'],[8,0,-4,10,3,-4,'minecraft:sandstone'],[-3,0,-9,3,1,-9,'minecraft:smooth_red_sandstone']]},
  {id:'deepslate_ruin',name_zh:'深板岩遗迹',floor:'minecraft:polished_deepslate',cross:'minecraft:lime_concrete',center:'minecraft:yellow_concrete',sample:blockTypes[3],geometry:[[-8,0,-7,-8,5,-7,'minecraft:polished_basalt'],[8,0,-7,8,5,-7,'minecraft:polished_basalt'],[-8,6,-7,8,6,-7,'minecraft:deepslate_bricks'],[-5,0,-8,-2,2,-8,'minecraft:deepslate_tiles'],[2,0,-8,5,3,-8,'minecraft:cracked_deepslate_bricks'],[-10,0,-2,-9,3,-2,'minecraft:amethyst_block'],[9,0,-2,10,2,-2,'minecraft:calcite']]},
  {id:'tuff_copper',name_zh:'凝灰铜庭',floor:'minecraft:tuff',cross:'minecraft:orange_concrete',center:'minecraft:light_blue_concrete',sample:blockTypes[1],geometry:[[-8,0,-7,-8,5,-7,'minecraft:oxidized_copper'],[8,0,-7,8,5,-7,'minecraft:cut_copper'],[-8,6,-7,8,6,-7,'minecraft:tuff_bricks'],[-6,0,-8,-3,2,-8,'minecraft:exposed_copper'],[3,0,-8,6,3,-8,'minecraft:weathered_copper'],[-10,0,-2,-9,3,-2,'minecraft:chiseled_tuff'],[9,0,-3,10,2,-3,'minecraft:copper_grate']]},
  {id:'snow_spruce',name_zh:'冰雪云杉庭',floor:'minecraft:snow_block',cross:'minecraft:magenta_concrete',center:'minecraft:lime_concrete',sample:blockTypes[0],geometry:[[-8,0,-7,-8,5,-7,'minecraft:packed_ice'],[8,0,-7,8,5,-7,'minecraft:blue_ice'],[-8,6,-7,8,6,-7,'minecraft:spruce_log'],[-6,0,-8,-3,2,-8,'minecraft:spruce_planks'],[3,0,-8,6,3,-8,'minecraft:packed_ice'],[-10,0,-2,-9,3,-2,'minecraft:blue_ice'],[9,0,-3,10,2,-3,'minecraft:spruce_log']]},
];

async function main() {
  if (!process.env.RCON_PASSWORD) throw new Error('RCON_PASSWORD must be set');
  const out = path.resolve(process.argv[2]);
  fs.mkdirSync(out,{recursive:true});
  const rcon = await Rcon.connect({host:'127.0.0.1',port:25721,password:process.env.RCON_PASSWORD,timeout:30000});
  const logs=[];
  const send=async command=>{
    const response=await rcon.send(command); logs.push({epoch_ms:Date.now(),command,response});
    if (/unknown or incomplete|incorrect argument|not a valid|no player was found|position is not loaded|too many blocks/i.test(String(response))) throw new Error(`${command} -> ${response}`);
    return response;
  };
  const records=[];
  try {
    const online=await rcon.send('list');
    if (!online.includes('CameraAlpha')||!online.includes('CameraBravo')) throw new Error(`camera clients unavailable: ${online}`);
    await send('difficulty peaceful'); await send('gamerule doDaylightCycle false'); await send('gamerule doWeatherCycle false'); await send('weather clear'); await send('time set noon');
    await rcon.send('team add mwb3persist50'); await send('team modify mwb3persist50 collisionRule never'); await send('team modify mwb3persist50 nametagVisibility never'); await send('team join mwb3persist50 CameraAlpha'); await send('team join mwb3persist50 CameraBravo');
    for (let index=0; index<scenes.length; index++) {
      const scene=scenes[index],tx=12000+index*48,ty=91,tz=0,aim=[tx+.5,ty+.1,tz+.5];
      await send(`forceload add ${tx-14} ${tz-12} ${tx+14} ${tz+12}`);
      await send(`fill ${tx-12} ${ty-1} ${tz-10} ${tx+12} ${ty-1} ${tz+11} ${scene.floor}`);
      await send(`fill ${tx-12} ${ty} ${tz-10} ${tx+12} ${ty+8} ${tz+11} minecraft:air`);
      await send(`execute positioned ${tx} ${ty} ${tz} run kill @e[type=minecraft:item,distance=..40]`);
      for (const [x1,y1,z1,x2,y2,z2,material] of scene.geometry) await send(`fill ${tx+x1} ${ty+y1} ${tz+z1} ${tx+x2} ${ty+y2} ${tz+z2} ${material}`);
      await send(`fill ${tx-4} ${ty-1} ${tz} ${tx+4} ${ty-1} ${tz} ${scene.cross}`); await send(`fill ${tx} ${ty-1} ${tz-4} ${tx} ${ty-1} ${tz+4} ${scene.cross}`); await send(`setblock ${tx} ${ty-1} ${tz} ${scene.center}`); await send(`setblock ${tx} ${ty} ${tz} minecraft:air`);
      const positions={alpha:[tx-2.6,ty,tz+6.6],bravo:[tx+2.6,ty,tz+6.6]};
      const actual={};
      for (const [role,player] of [['alpha','CameraAlpha'],['bravo','CameraBravo']]) {
        const rotation=lookAt(positions[role],aim); await send(`gamemode survival ${player}`); await send(`clear ${player}`);
        if (role==='bravo') await send(`item replace entity ${player} weapon.mainhand with ${scene.sample.id} 1`);
        await send(`tp ${player} ${positions[role].join(' ')} ${rotation.yaw} ${rotation.pitch}`);
        actual[role]={position:positions[role],yaw:rotation.yaw,pitch:rotation.pitch};
      }
      await wait(5000);
      const a0=path.join(out,`${index+1}_${scene.id}_alpha_first.png`),b0=path.join(out,`${index+1}_${scene.id}_bravo_first.png`),first=path.join(out,`${index+1}_${scene.id}_first.png`);
      shot(':130.0',a0); shot(':131.0',b0); pair(a0,b0,first);
      await send(`setblock ${tx} ${ty} ${tz} ${scene.sample.id}`); await wait(700);
      const a1=path.join(out,`${index+1}_${scene.id}_alpha_after.png`),b1=path.join(out,`${index+1}_${scene.id}_bravo_after.png`),after=path.join(out,`${index+1}_${scene.id}_after.png`);
      shot(':130.0',a1); shot(':131.0',b1); pair(a1,b1,after);
      records.push({scene_id:scene.id,scene_name_zh:scene.name_zh,floor:scene.floor,cross:scene.cross,center:scene.center,sample_block:scene.sample,origin:[tx,ty,tz],camera_poses:actual,first_frame:path.basename(first),first_frame_sha256:sha256(first),placed_preview:path.basename(after),placed_preview_sha256:sha256(after)});
      console.log(scene.id,first,after);
    }
  } finally {
    fs.writeFileSync(path.join(out,'initialization.jsonl'),logs.map(x=>JSON.stringify(x)).join('\n')+'\n');
    fs.writeFileSync(path.join(out,'design.json'),JSON.stringify({schema_version:'mwb.v3.shared_state_persistence.50_case_design.v1',candidate_only:true,gt_collected:false,inference_run:false,factorization:{scenes:5,block_types:5,cases_per_combination:2,total:50},case_pair_interpretation:['place','destroy'],block_types:blockTypes,scenes:records},null,2)+'\n');
    await rcon.end().catch(()=>{});
  }
}
main().catch(error=>{console.error(error.stack||error.message);process.exit(1);});
