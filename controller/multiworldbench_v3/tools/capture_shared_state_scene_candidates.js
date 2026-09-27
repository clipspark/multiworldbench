'use strict';
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');
const { Rcon } = require('rcon-client');

const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
const deg = radians => radians * 180 / Math.PI;
function lookAt(position, aim) {
  const dx = aim[0] - position[0];
  const dy = aim[1] - (position[1] + 1.62);
  const dz = aim[2] - position[2];
  return {yaw: deg(Math.atan2(-dx, dz)), pitch: -deg(Math.atan2(dy, Math.hypot(dx, dz)))};
}

const tx = 760, ty = 91, tz = 0;
const aim = [tx + 0.5, ty, tz + 0.5];
const poses = {
  mutual: {alpha:[tx-2,ty,tz+4], bravo:[tx+2,ty,tz+4]},
  alpha_sees_bravo: {alpha:[tx-1.5,ty,tz+6], bravo:[tx+0.5,ty,tz+3.5]},
  bravo_sees_alpha: {alpha:[tx-0.5,ty,tz+3.5], bravo:[tx+1.5,ty,tz+6]},
  none: {alpha:[tx-4,ty,tz+4], bravo:[tx+4,ty,tz+4]},
};
const scenes = [
  ['MWB3-SS-SC01','苔石草地','mutual','A、B 相互可见','minecraft:grass_block','minecraft:white_concrete','minecraft:red_concrete',[
    'fill 753 91 -6 753 93 -6 minecraft:oak_log','fill 767 91 -6 767 93 -6 minecraft:oak_log','fill 753 94 -6 767 94 -6 minecraft:oak_log','fill 754 91 -6 758 91 -6 minecraft:mossy_cobblestone','fill 762 91 -6 766 91 -6 minecraft:mossy_cobblestone']],
  ['MWB3-SS-SC02','砂岩拱庭','mutual','A、B 相互可见','minecraft:smooth_sandstone','minecraft:yellow_concrete','minecraft:blue_concrete',[
    'fill 756 91 -6 756 94 -6 minecraft:cut_sandstone','fill 764 91 -6 764 94 -6 minecraft:cut_sandstone','fill 756 95 -6 764 95 -6 minecraft:smooth_sandstone','fill 751 91 -4 753 92 -4 minecraft:sandstone','fill 767 91 -4 769 92 -4 minecraft:sandstone']],
  ['MWB3-SS-SC03','石砖广场','mutual','A、B 相互可见','minecraft:stone_bricks','minecraft:cyan_concrete','minecraft:orange_concrete',[
    'fill 752 91 -6 752 94 -6 minecraft:polished_andesite','fill 768 91 -6 768 94 -6 minecraft:polished_andesite','fill 755 91 -7 755 92 -7 minecraft:chiseled_stone_bricks','fill 765 91 -7 765 93 -7 minecraft:chiseled_stone_bricks','fill 757 91 -6 763 91 -6 minecraft:mossy_stone_bricks']],
  ['MWB3-SS-SC04','云杉作业场','alpha_sees_bravo','仅 A 看见 B','minecraft:podzol','minecraft:lime_concrete','minecraft:purple_concrete',[
    'fill 753 91 -6 753 94 -6 minecraft:spruce_log','fill 767 91 -6 767 94 -6 minecraft:spruce_log','fill 753 95 -6 767 95 -6 minecraft:spruce_log','fill 754 91 -6 758 92 -6 minecraft:spruce_planks','fill 762 91 -6 766 92 -6 minecraft:spruce_planks']],
  ['MWB3-SS-SC05','凝灰铜庭','alpha_sees_bravo','仅 A 看见 B','minecraft:tuff','minecraft:orange_concrete','minecraft:light_blue_concrete',[
    'fill 753 91 -6 753 94 -6 minecraft:oxidized_copper','fill 767 91 -6 767 94 -6 minecraft:cut_copper','fill 753 95 -6 767 95 -6 minecraft:tuff_bricks','fill 756 91 -7 758 92 -7 minecraft:exposed_copper','fill 762 91 -7 764 92 -7 minecraft:weathered_copper']],
  ['MWB3-SS-SC06','红砂遗迹','bravo_sees_alpha','仅 B 看见 A（单向反例）','minecraft:red_sandstone','minecraft:light_blue_concrete','minecraft:yellow_concrete',[
    'fill 755 91 -6 755 94 -6 minecraft:chiseled_red_sandstone','fill 765 91 -6 765 94 -6 minecraft:chiseled_red_sandstone','fill 755 95 -6 765 95 -6 minecraft:smooth_red_sandstone','fill 751 91 -3 753 92 -3 minecraft:cut_red_sandstone','fill 767 91 -3 769 92 -3 minecraft:cut_red_sandstone']],
  ['MWB3-SS-SC07','冰雪庭院','bravo_sees_alpha','仅 B 看见 A（单向反例）','minecraft:snow_block','minecraft:magenta_concrete','minecraft:lime_concrete',[
    'fill 753 91 -6 753 94 -6 minecraft:packed_ice','fill 767 91 -6 767 94 -6 minecraft:blue_ice','fill 753 95 -6 767 95 -6 minecraft:spruce_log','fill 756 91 -7 758 92 -7 minecraft:packed_ice','fill 762 91 -7 764 92 -7 minecraft:blue_ice']],
  ['MWB3-SS-SC08','深板岩分廊','none','A、B 相互不可见','minecraft:polished_deepslate','minecraft:red_concrete','minecraft:white_concrete',[
    'fill 760 91 4 760 93 4 minecraft:polished_basalt','fill 752 91 -6 752 94 -6 minecraft:polished_basalt','fill 768 91 -6 768 94 -6 minecraft:polished_basalt','fill 752 95 -6 768 95 -6 minecraft:deepslate_bricks','fill 756 91 -7 758 92 -7 minecraft:deepslate_tiles','fill 762 91 -7 764 92 -7 minecraft:deepslate_tiles']],
  ['MWB3-SS-SC09','石英分庭','none','A、B 相互不可见','minecraft:moss_block','minecraft:blue_concrete','minecraft:orange_concrete',[
    'fill 760 91 4 760 93 4 minecraft:quartz_pillar','fill 754 91 -6 754 94 -6 minecraft:quartz_bricks','fill 766 91 -6 766 94 -6 minecraft:quartz_bricks','fill 754 95 -6 766 95 -6 minecraft:smooth_quartz','fill 756 91 -7 758 91 -7 minecraft:flowering_azalea_leaves','fill 762 91 -7 764 91 -7 minecraft:flowering_azalea_leaves']],
  ['MWB3-SS-SC10','泥砖分厅','none','A、B 相互不可见','minecraft:mud_bricks','minecraft:pink_concrete','minecraft:black_concrete',[
    'fill 760 91 4 760 93 4 minecraft:bricks','fill 752 91 -6 752 94 -6 minecraft:dark_oak_log','fill 768 91 -6 768 94 -6 minecraft:dark_oak_log','fill 752 95 -6 768 95 -6 minecraft:dark_oak_planks','fill 755 91 -7 758 92 -7 minecraft:mud_bricks','fill 762 91 -7 765 92 -7 minecraft:packed_mud']],
].map(([id,name,visibility,visibility_zh,floor,cross,center,geometry]) => ({id,name,visibility,visibility_zh,floor,cross,center,geometry,poses:poses[visibility]}));

async function main() {
  if (!process.env.RCON_PASSWORD) throw new Error('RCON_PASSWORD must be set');
  const out = path.resolve(process.argv[2] || '/tmp/mwb3-shared-state-scene-candidates');
  fs.mkdirSync(out,{recursive:true});
  const rcon = await Rcon.connect({host:'127.0.0.1',port:25721,password:process.env.RCON_PASSWORD,timeout:10000});
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
    await send('forceload add 748 -8 772 10'); await send('difficulty peaceful');
    await send('gamerule doDaylightCycle false'); await send('gamerule doWeatherCycle false'); await send('weather clear'); await send('time set noon');
    await rcon.send('team add mwb3scene'); await send('team modify mwb3scene collisionRule never'); await send('team modify mwb3scene nametagVisibility never');
    await send('team join mwb3scene CameraAlpha'); await send('team join mwb3scene CameraBravo');
    for (const scene of scenes) {
      await send(`fill ${tx-10} ${ty-1} ${tz-8} ${tx+10} ${ty-1} ${tz+10} ${scene.floor}`);
      await send(`fill ${tx-10} ${ty} ${tz-8} ${tx+10} ${ty+7} ${tz+10} minecraft:air`);
      await send(`execute positioned ${tx} ${ty} ${tz} run kill @e[type=minecraft:item,distance=..32]`);
      for (const command of scene.geometry) await send(command);
      await send(`fill ${tx-4} ${ty-1} ${tz} ${tx+4} ${ty-1} ${tz} ${scene.cross}`);
      await send(`fill ${tx} ${ty-1} ${tz-4} ${tx} ${ty-1} ${tz+4} ${scene.cross}`);
      await send(`setblock ${tx} ${ty-1} ${tz} ${scene.center}`); await send(`setblock ${tx} ${ty} ${tz} minecraft:air`);
      const actual={};
      for (const [role,player] of [['alpha','CameraAlpha'],['bravo','CameraBravo']]) {
        const position=scene.poses[role], rotation=lookAt(position,aim);
        await send(`gamemode survival ${player}`); await send(`clear ${player}`); await send(`tp ${player} ${position.join(' ')} ${rotation.yaw} ${rotation.pitch}`);
        actual[role]={position,yaw:rotation.yaw,pitch:rotation.pitch,server_position:await send(`data get entity ${player} Pos`),server_rotation:await send(`data get entity ${player} Rotation`)};
      }
      await wait(1500);
      const alpha=path.join(out,`${scene.id}_alpha.png`), bravo=path.join(out,`${scene.id}_bravo.png`), pair=path.join(out,`${scene.id}.png`);
      execFileSync('ffmpeg',['-nostdin','-v','error','-y','-f','x11grab','-video_size','1280x720','-i',':130.0','-frames:v','1',alpha]);
      execFileSync('ffmpeg',['-nostdin','-v','error','-y','-f','x11grab','-video_size','1280x720','-i',':131.0','-frames:v','1',bravo]);
      execFileSync('ffmpeg',['-nostdin','-v','error','-y','-i',alpha,'-i',bravo,'-filter_complex','[0:v]scale=640:360:flags=lanczos,crop=640:352:0:4[a];[1:v]scale=640:360:flags=lanczos,crop=640:352:0:4[b];[a][b]hstack=inputs=2[v]','-map','[v]','-frames:v','1',pair]);
      records.push({...scene,aim,actual,files:{alpha,bravo,pair}}); console.log(`${scene.id} ${scene.visibility} ${pair}`);
    }
  } finally {
    fs.writeFileSync(path.join(out,'initialization.jsonl'),logs.map(x=>JSON.stringify(x)).join('\n')+'\n');
    fs.writeFileSync(path.join(out,'design.json'),JSON.stringify({schema_version:'mwb.v3.shared_state.scene_candidates.v1',candidate_only:true,gt_collected:false,inference_run:false,target_cell:[tx,ty,tz],candidates:records},null,2)+'\n');
    await rcon.end().catch(()=>{});
  }
}
main().catch(error=>{console.error(error.stack||error.message);process.exit(1);});
