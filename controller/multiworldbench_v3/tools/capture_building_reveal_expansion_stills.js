'use strict';

// Build five real Minecraft scene/building combinations and capture the first
// native RGB frame from CameraAlpha and CameraBravo. This script changes only
// case initialization through RCON; it does not patch Minecraft or model code.
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { execFileSync } = require('child_process');
const { Rcon } = require('rcon-client');

const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
const sha256 = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');

const candidates = [
  { id:'MWB3-RV-C01', scene_id:'MWB3-RV-SC01', scene:'云杉峡谷工场', building_id:'MWB3-RV-BL01', building:'偏置水轮磨坊', angle_deg:20, ground:'grass_block', sceneBuild:buildSpruceCanyon, buildingBuild:buildMill },
  { id:'MWB3-RV-C02', scene_id:'MWB3-RV-SC02', scene:'砂岩峡谷绿洲', building_id:'MWB3-RV-BL02', building:'高低双塔门楼', angle_deg:25, ground:'sand', sceneBuild:buildOasis, buildingBuild:buildGatehouse },
  { id:'MWB3-RV-C03', scene_id:'MWB3-RV-SC03', scene:'雪原玄武岩采石场', building_id:'MWB3-RV-BL03', building:'阶梯星象台', angle_deg:30, ground:'snow_block', sceneBuild:buildSnowQuarry, buildingBuild:buildObservatory },
  { id:'MWB3-RV-C04', scene_id:'MWB3-RV-SC04', scene:'红树林潮汐码头', building_id:'MWB3-RV-BL04', building:'高脚信号屋', angle_deg:35, ground:'mud', sceneBuild:buildMangroveDock, buildingBuild:buildSignalHouse },
  { id:'MWB3-RV-C05', scene_id:'MWB3-RV-SC05', scene:'恶地化石发掘营', building_id:'MWB3-RV-BL05', building:'锯齿顶铸造厂', angle_deg:40, ground:'red_sand', sceneBuild:buildBadlandsDig, buildingBuild:buildFoundry },
];

function aimPose(origin, focus) {
  const dx=focus[0]-origin[0], dy=focus[1]-(origin[1]+1.62), dz=focus[2]-origin[2];
  return { yaw:Math.atan2(-dx,dz)*180/Math.PI, pitch:-Math.atan2(dy,Math.hypot(dx,dz))*180/Math.PI };
}

function builder(send,cx) {
  const fill=(x1,y1,z1,x2,y2,z2,block,mode='')=>send(`fill ${cx+x1} ${y1} ${z1} ${cx+x2} ${y2} ${z2} minecraft:${block}${mode?` ${mode}`:''}`);
  const set=(x,y,z,block)=>send(`setblock ${cx+x} ${y} ${z} minecraft:${block}`);
  const column=(x,y,z,h,block)=>fill(x,y,z,x,y+h-1,z,block);
  const beamX=(x1,x2,y,z,block)=>fill(x1,y,z,x2,y,z,block);
  const beamZ=(x,y,z1,z2,block)=>fill(x,y,z1,x,y,z2,block);
  const hollow=async(x1,y1,z1,x2,y2,z2,block)=>{
    await fill(x1,y1,z1,x2,y2,z1,block); await fill(x1,y1,z2,x2,y2,z2,block);
    await fill(x1,y1,z1,x1,y2,z2,block); await fill(x2,y1,z1,x2,y2,z2,block);
  };
  return {fill,set,column,beamX,beamZ,hollow};
}

async function base(b,ground) {
  await b.fill(-34,60,-30,34,62,30,'dirt');
  await b.fill(-34,63,-30,34,63,30,ground);
  for (const [x1,z1,x2,z2] of [[-34,-30,0,0],[-34,1,0,30],[1,-30,34,0],[1,1,34,30]]) {
    await b.fill(x1,64,z1,x2,91,z2,'air');
    await b.fill(x1,92,z1,x2,120,z2,'air');
  }
}

async function tree(b,x,z,h,log='spruce_log',leaf='spruce_leaves') {
  await b.column(x,64,z,h,log);
  await b.fill(x-2,64+h-3,z-2,x+2,64+h-1,z+2,leaf);
  await b.fill(x-1,64+h,z-1,x+1,64+h+1,z+1,leaf);
}

async function buildSpruceCanyon(b) {
  await b.fill(-33,64,15,-19,70,29,'stone'); await b.fill(20,64,13,33,73,29,'andesite');
  await b.fill(-31,64,-24,-22,68,3,'mossy_cobblestone'); await b.fill(23,64,-24,32,69,2,'stone');
  await b.fill(-18,63,-27,-14,63,29,'water');
  await b.fill(-21,65,-4,-11,66,0,'spruce_planks');
  for (const x of [-20,-13]) for (const z of [-4,0]) await b.column(x,64,z,3,'spruce_log');
  for (const [x,z,h] of [[-29,-18,8],[-24,8,10],[28,-17,9],[25,7,12],[18,24,8]]) await tree(b,x,z,h);
  for (const [x,z] of [[-10,18],[13,18],[19,-10],[-27,12]]) { await b.column(x,64,z,4,'cobblestone_wall'); await b.set(x,68,z,'lantern'); }
  await b.fill(-9,64,20,-2,65,23,'stripped_spruce_log'); await b.fill(12,64,21,18,66,23,'spruce_planks');
}

async function buildOasis(b) {
  for (let i=0;i<4;i++) { await b.fill(-34+i*3,64+i*2,13+i*2,-22+i*2,65+i*2,30,'cut_sandstone'); await b.fill(22-i*2,64+i*2,15+i*2,34,65+i*2,30,'red_sandstone'); }
  await b.fill(-30,64,-28,-22,70,0,'smooth_sandstone'); await b.fill(23,64,-28,32,72,1,'sandstone');
  await b.fill(-17,63,-20,-9,63,-11,'water'); await b.fill(-18,63,-21,-8,63,-10,'sand');
  for (const [x,z,h] of [[-20,8,7],[-14,19,5],[18,18,8],[27,7,6]]) { await b.column(x,64,z,h,'chiseled_sandstone'); await b.set(x,64+h,z,'gold_block'); }
  for (const [x,z] of [[-18,-16],[-10,-14]]) { await b.column(x,64,z,5,'jungle_log'); await b.fill(x-3,69,z-3,x+3,70,z+3,'acacia_leaves'); }
  await b.fill(11,68,-20,21,68,-13,'red_wool');
  for (const x of [11,21]) for (const z of [-20,-13]) await b.column(x,64,z,5,'oak_fence');
  await b.fill(-7,64,22,8,66,26,'red_sandstone'); await b.fill(-3,64,22,4,65,22,'air');
}

async function buildSnowQuarry(b) {
  for (let i=0;i<5;i++) await b.fill(-33+i*3,64+i*2,10+i*2,-21+i*2,65+i*2,29,'stone');
  for (let i=0;i<4;i++) await b.fill(22-i*2,64+i*2,12+i*3,34,65+i*2,30,i%2?'polished_basalt':'basalt');
  await b.fill(-28,64,-24,-21,71,-5,'packed_ice'); await b.fill(23,64,-23,31,70,-3,'snow_block');
  for (const [x,z,h] of [[-18,20,10],[-9,24,7],[16,21,12],[27,10,9]]) await b.column(x,64,z,h,'polished_basalt');
  await b.beamX(-25,-7,73,-10,'dark_oak_log'); await b.column(-25,64,-10,10,'dark_oak_log'); await b.column(-7,64,-10,10,'dark_oak_log');
  await b.beamZ(-16,72,-10,1,'dark_oak_log'); await b.set(-16,70,1,'chain'); await b.set(-16,69,1,'lantern');
  await b.fill(8,64,-24,10,64,23,'iron_block');
  for (let z=-22;z<=22;z+=5) await b.beamX(6,12,64,z,'spruce_planks');
  for (const [x,z] of [[-30,4],[-20,-16],[29,-13],[20,4]]) { await b.column(x,64,z,3,'spruce_fence'); await b.set(x,67,z,'lantern'); }
}

async function buildMangroveDock(b) {
  await b.fill(-32,63,-26,-19,63,28,'water'); await b.fill(19,63,-27,33,63,29,'water');
  await b.fill(-13,63,16,14,63,28,'water');
  await b.fill(-30,64,-3,30,64,0,'mangrove_planks'); await b.fill(-2,64,-26,2,64,28,'mangrove_planks');
  for (let x=-30;x<=30;x+=6) { await b.column(x,63,-3,3,'mangrove_log'); await b.column(x,63,0,3,'mangrove_log'); }
  for (let z=-24;z<=27;z+=6) { await b.column(-2,63,z,3,'mangrove_log'); await b.column(2,63,z,3,'mangrove_log'); }
  for (const [x,z,h] of [[-25,-18,7],[-24,15,9],[25,-17,8],[27,15,10]]) { await b.column(x,64,z,h,'mangrove_log'); await b.fill(x-3,64+h-2,z-3,x+3,64+h+1,z+3,'mangrove_leaves'); }
  for (const [x,z] of [[-14,-12],[13,-11],[-12,10],[15,12]]) { await b.column(x,64,z,4,'dark_oak_fence'); await b.set(x,68,z,'lantern'); }
  await b.fill(-30,64,20,-20,66,25,'packed_mud'); await b.fill(21,64,19,31,65,26,'mud_bricks');
}

async function buildBadlandsDig(b) {
  const layers=['orange_terracotta','red_terracotta','yellow_terracotta','terracotta'];
  for (let i=0;i<5;i++) { await b.fill(-34+i*3,64+i*2,10+i*2,-20+i*2,65+i*2,30,layers[i%4]); await b.fill(21-i*2,64+i*2,13+i*2,34,65+i*2,30,layers[(i+1)%4]); }
  await b.fill(-31,64,-27,-23,73,1,'red_sandstone'); await b.fill(25,64,-27,33,70,2,'orange_terracotta');
  for (let i=0;i<6;i++) { const x=-24+i*6; await b.column(x,64,20,4+Math.abs(3-i),'bone_block'); await b.beamZ(x,68,16,24,'bone_block'); }
  await b.beamX(-28,-7,72,-11,'scaffolding'); await b.column(-28,64,-11,9,'scaffolding'); await b.column(-7,64,-11,9,'scaffolding');
  await b.fill(11,64,-23,13,64,23,'iron_block'); for (let z=-21;z<=21;z+=6) await b.beamX(9,15,64,z,'oak_planks');
  for (const [x,z,c] of [[-18,-18,'blue_wool'],[20,-14,'white_wool'],[27,8,'lime_wool']]) { await b.column(x,64,z,6,'oak_fence'); await b.fill(x,68,z,x+3,70,z,c); }
  await b.fill(-6,64,22,3,67,27,'barrel'); await b.fill(14,64,20,20,66,25,'oak_planks');
}

async function roof(b,x1,x2,y,z1,z2,block) {
  let inset=0;
  while (x1+inset<=x2-inset && z1+inset<=z2-inset) {
    await b.fill(x1+inset,y+inset,z1+inset,x2-inset,y+inset,z1+inset,block);
    await b.fill(x1+inset,y+inset,z2-inset,x2-inset,y+inset,z2-inset,block);
    inset++;
  }
}

async function buildMill(b) {
  await b.hollow(-4,64,-2,4,68,5,'cobblestone'); await b.fill(-3,69,-1,3,72,4,'spruce_planks');
  await b.fill(-1,64,-2,1,67,-2,'air'); await b.fill(-3,66,-2,-2,67,-2,'glass'); await b.fill(2,66,-2,3,67,-2,'glass');
  await roof(b,-5,5,73,-3,6,'dark_oak_planks');
  await b.hollow(-2,73,-1,0,75,2,'red_terracotta'); await b.set(-1,74,-1,'glass');
  await b.fill(5,64,-1,5,69,3,'dark_oak_log');
  for (const [y,z] of [[65,-1],[65,3],[69,-1],[69,3]]) await b.set(6,y,z,'dark_oak_planks');
  await b.column(6,65,1,5,'dark_oak_log'); await b.beamZ(6,67,-1,3,'dark_oak_log');
  await b.fill(4,64,5,10,64,7,'oak_planks');
}

async function buildGatehouse(b) {
  await b.hollow(-6,64,-2,-2,74,4,'cut_sandstone'); await b.hollow(2,64,-2,6,71,4,'smooth_sandstone');
  await b.fill(-2,64,-2,2,69,4,'sandstone'); await b.fill(-1,64,-2,1,68,-2,'air');
  await b.fill(-1,64,-1,1,68,4,'air'); await b.fill(-1,69,-2,1,69,-2,'chiseled_sandstone');
  await b.fill(-5,66,-2,-5,72,-2,'blue_concrete');
  for (const x of [-6,-4,-2]) await b.set(x,75,-1,'sandstone'); for (const x of [2,4,6]) await b.set(x,72,-1,'sandstone');
  await b.beamZ(7,67,-2,4,'dark_oak_log'); await b.beamX(5,9,70,4,'dark_oak_log');
}

async function buildObservatory(b) {
  await b.fill(-5,64,-3,5,65,7,'stone_bricks'); await b.fill(-4,66,-2,4,67,6,'polished_andesite'); await b.fill(-3,68,-1,3,70,5,'quartz_bricks');
  await b.fill(-1,64,-3,1,68,-3,'air'); await b.fill(-1,64,-2,1,68,1,'air');
  await b.fill(-2,71,0,2,73,4,'oxidized_copper'); await b.fill(-1,74,1,1,75,3,'weathered_copper'); await b.set(0,76,2,'lightning_rod');
  await b.beamZ(1,75,-3,5,'waxed_copper_block'); await b.set(1,75,-4,'amethyst_block');
  for (let i=0;i<5;i++) await b.fill(-7+i,64+i,-5,-5+i,64+i,-3,'stone_bricks');
  await b.fill(-1,66,-3,1,67,-3,'purple_concrete');
}

async function buildSignalHouse(b) {
  for (const [x,z] of [[-4,-2],[3,-2],[-4,5],[3,5]]) await b.column(x,64,z,5,'mangrove_log');
  await b.fill(-5,69,-3,4,69,6,'mangrove_planks'); await b.hollow(-4,70,-2,3,74,5,'stripped_mangrove_wood');
  await b.fill(-1,70,-2,1,73,-2,'air'); await b.fill(-3,71,-2,-2,72,-2,'glass'); await b.fill(2,71,-2,3,72,-2,'glass');
  for (let i=0;i<5;i++) await b.fill(-5,75+i,-3+i,4,75+i,-3+i,'dark_prismarine');
  await b.fill(4,69,1,7,69,6,'mangrove_planks'); await b.fill(7,70,1,7,72,6,'mangrove_fence');
  await b.column(-3,75,4,8,'spruce_fence'); await b.set(-3,83,4,'redstone_lamp');
  for (const [x,z] of [[-4,-2],[3,-2],[3,5]]) await b.set(x,68,z,'lantern');
}

async function buildFoundry(b) {
  await b.hollow(-6,64,-3,6,70,5,'bricks'); await b.fill(-2,64,-3,1,68,-3,'air');
  await b.fill(-5,66,-3,-3,68,-3,'iron_bars'); await b.fill(3,66,-3,5,68,-3,'iron_bars');
  for (let bay=0;bay<3;bay++) { const x=-6+bay*4; await b.fill(x,71,-3,x+3,71,5,'deepslate_tiles'); await b.fill(x+1,72,-3,x+3,72,5,'deepslate_tiles'); await b.fill(x+2,73,-3,x+3,73,5,'deepslate_tiles'); }
  await b.column(-5,71,4,9,'bricks'); await b.column(-4,71,4,9,'white_concrete'); await b.set(-5,80,4,'black_concrete');
  await b.column(8,64,3,8,'oxidized_copper'); await b.beamX(6,12,72,3,'waxed_copper_block'); await b.set(12,71,3,'chain'); await b.set(12,70,3,'lantern');
  await b.beamZ(7,66,-2,5,'exposed_copper');
}

async function main() {
  if (!process.env.RCON_PASSWORD) throw new Error('RCON_PASSWORD must be set');
  const output=path.resolve(process.argv[2]||'');
  if (!process.argv[2]) throw new Error('usage: node capture_building_reveal_expansion_stills.js OUTPUT_DIR');
  if (fs.existsSync(output)) throw new Error(`Refusing to overwrite ${output}`);
  fs.mkdirSync(output,{recursive:true});
  const rcon=await Rcon.connect({host:'127.0.0.1',port:25721,password:process.env.RCON_PASSWORD,timeout:30000});
  const initLog=path.join(output,'initialization.jsonl');
  const send=async command=>{
    const response=await rcon.send(command);
    fs.appendFileSync(initLog,JSON.stringify({epoch_ms:Date.now(),command,response:String(response)})+'\n');
    if (/unknown or incomplete|incorrect argument|expected integer|not a valid|position is not loaded|too many blocks|no player was found/i.test(String(response))) throw new Error(`${command} -> ${response}`);
    return response;
  };
  const records=[];
  try {
    const online=await send('list');
    if (!online.includes('CameraAlpha')||!online.includes('CameraBravo')) throw new Error(`native clients unavailable: ${online}`);
    for (const command of ['difficulty peaceful','gamerule doDaylightCycle false','gamerule doWeatherCycle false','weather clear','time set noon','kill @e[type=!player]','gamemode survival CameraAlpha','gamemode survival CameraBravo','clear CameraAlpha','clear CameraBravo']) await send(command);
    await rcon.send('team add mwb3reveal'); await send('team modify mwb3reveal collisionRule never'); await send('team modify mwb3reveal nametagVisibility never'); await send('team join mwb3reveal CameraAlpha'); await send('team join mwb3reveal CameraBravo');
    for (let i=0;i<candidates.length;i++) {
      const spec=candidates[i],cx=6200+i*96,caseDir=path.join(output,spec.id); fs.mkdirSync(caseDir);
      await send(`forceload add ${cx-35} -31 ${cx+35} 31`);
      const b=builder(send,cx); await base(b,spec.ground); await spec.sceneBuild(b); await spec.buildingBuild(b);
      const focus=[cx+0.5,68,2.5];
      const alpha=[cx+0.5,64,-10.05];
      const rad=spec.angle_deg*Math.PI/180,bravo=[cx+0.5+18*Math.sin(rad),64,2.5-18*Math.cos(rad)];
      // B is the stationary observer. Aim between the building center and A's
      // eye position so both remain comfortably inside B's initial frame.
      const bravoFocus=[cx+0.5,66.81,-3.775];
      // A single dirt block seals a one-block observation aperture. The frame
      // prevents tall buildings from leaking around the occluder; breaking the
      // dirt is still the only reveal action required by the case.
      await b.fill(-2,64,-10,2,69,-10,'polished_deepslate');
      const occluder=[cx,65,-10]; await send(`setblock ${occluder.join(' ')} minecraft:dirt`);
      await send('clear CameraAlpha'); await send('clear CameraBravo');
      const actual={};
      for (const [role,name,position] of [['alpha','CameraAlpha',alpha],['bravo','CameraBravo',bravo]]) {
        const roleFocus=role==='bravo'?bravoFocus:focus;
        const rotation=aimPose(position,roleFocus); await send(`tp ${name} ${position.join(' ')} ${rotation.yaw} ${rotation.pitch}`); await wait(800);
        actual[role]={requested:{position,yaw:rotation.yaw,pitch:rotation.pitch},server_position:await send(`data get entity ${name} Pos`),server_rotation:await send(`data get entity ${name} Rotation`),on_ground:await send(`data get entity ${name} OnGround`)};
      }
      await wait(3500);
      const alphaFile=path.join(caseDir,'alpha_first_frame.png'),bravoFile=path.join(caseDir,'bravo_first_frame.png'),pairFile=path.join(caseDir,'first_frame_pair.png');
      execFileSync('ffmpeg',['-nostdin','-v','error','-y','-f','x11grab','-video_size','1280x720','-i',':130.0','-frames:v','1',alphaFile]);
      execFileSync('ffmpeg',['-nostdin','-v','error','-y','-f','x11grab','-video_size','1280x720','-i',':131.0','-frames:v','1',bravoFile]);
      execFileSync('ffmpeg',['-nostdin','-v','error','-y','-i',alphaFile,'-i',bravoFile,'-filter_complex','[0:v]scale=640:360:flags=lanczos[a];[1:v]scale=640:360:flags=lanczos[b];[a][b]hstack=inputs=2[v]','-map','[v]','-frames:v','1',pairFile]);
      const files={alpha:path.relative(output,alphaFile),bravo:path.relative(output,bravoFile),pair:path.relative(output,pairFile),sha256:{alpha:sha256(alphaFile),bravo:sha256(bravoFile),pair:sha256(pairFile)}};
      const record={case_id:spec.id,scene_id:spec.scene_id,scene_name:spec.scene,building_id:spec.building_id,building_name:spec.building,relative_angle_deg:spec.angle_deg,focus,bravo_focus:bravoFocus,occluder:{block:'minecraft:dirt',position:occluder,aperture_frame:'minecraft:polished_deepslate',required_break_count:1},precondition:{alpha_building_visible:false,bravo_building_visible:true,bravo_sees_alpha:true},actual,files,gt_collected:false,inference_run:false};
      fs.writeFileSync(path.join(caseDir,'capture.json'),JSON.stringify(record,null,2)+'\n'); records.push(record); console.log(`Captured ${spec.id}: ${spec.scene} / ${spec.building} / ${spec.angle_deg}°`);
    }
    fs.writeFileSync(path.join(output,'capture_manifest.json'),JSON.stringify({schema_version:'mwb.v3.building_reveal.first_frames.v1',source:'native Minecraft clients after RCON-only case initialization',candidate_only:true,game_modified:false,gt_collected:false,inference_run:false,resolution_each:[1280,720],pair_resolution:[1280,360],records},null,2)+'\n');
  } finally { await rcon.end().catch(()=>{}); }
}

if (require.main===module) main().catch(error=>{console.error(error.stack||error.message);process.exit(1);});
module.exports={candidates,aimPose,builder};
