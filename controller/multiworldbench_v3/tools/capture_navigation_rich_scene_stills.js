'use strict';

// Build ten deterministic, geometry-rich Minecraft scenes and capture the
// actual first RGB frame from both native camera clients. Scene setup is RCON
// initialization only; this tool records no navigation actions or GT video.
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { execFileSync } = require('child_process');
const { Rcon } = require('rcon-client');

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const sha256 = (file) => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');

const scenes = [
  { id: 'MWB3-N-SC01', name: '分层采石场庭院', slug: 'layered_quarry_court', build: buildQuarry,
    alpha: [-24, 64, -18], bravo: [22, 64, -15], focus: [0, 66, 5] },
  { id: 'MWB3-N-SC02', name: '残破引水渠广场', slug: 'broken_aqueduct_plaza', build: buildAqueduct,
    alpha: [-24, 64, -17], bravo: [23, 64, -17], focus: [0, 68, 7] },
  { id: 'MWB3-N-SC03', name: '折返崖壁工坊', slug: 'switchback_cliff_workshop', build: buildWorkshop,
    alpha: [-25, 64, -16], bravo: [23, 64, -18], focus: [1, 69, 8] },
  { id: 'MWB3-N-SC04', name: '下沉神庙发掘区', slug: 'sunken_temple_excavation', build: buildTemple,
    alpha: [-23, 64, -19], bravo: [22, 64, -18], focus: [0, 67, 5] },
  { id: 'MWB3-N-SC05', name: '铸造厂高架庭院', slug: 'foundry_catwalk_yard', build: buildFoundry,
    alpha: [-25, 64, -17], bravo: [24, 64, -16], focus: [0, 69, 7] },
  { id: 'MWB3-N-SC06', name: '双桥峡谷节点', slug: 'twin_bridge_canyon_hub', build: buildCanyon,
    alpha: [-25, 64, -18], bravo: [24, 64, -16], focus: [0, 68, 7] },
  { id: 'MWB3-N-SC07', name: '非对称集市堡院', slug: 'asymmetric_market_citadel', build: buildMarket,
    alpha: [-23, 64, -19], bravo: [22, 64, -18], focus: [0, 68, 6] },
  { id: 'MWB3-N-SC08', name: '岩林遗迹神社', slug: 'rocky_woodland_shrine', build: buildShrine,
    alpha: [-24, 64, -17], bravo: [23, 64, -17], focus: [1, 67, 8] },
  { id: 'MWB3-N-SC09', name: '干船坞造船场', slug: 'dry_dock_shipyard', build: buildDock,
    alpha: [-25, 64, -17], bravo: [24, 64, -16], focus: [0, 67, 8] },
  { id: 'MWB3-N-SC10', name: '玄武岩晶簇洞厅', slug: 'basalt_crystal_cavern', build: buildCavern,
    alpha: [-23, 64, -18], bravo: [22, 64, -17], focus: [0, 68, 7] },
];

function aimPose(origin, focus) {
  const eye = [origin[0], origin[1] + 1.62, origin[2]];
  const d = [focus[0] - eye[0], focus[1] - eye[1], focus[2] - eye[2]];
  return {
    yaw: Math.atan2(-d[0], d[2]) * 180 / Math.PI,
    pitch: -Math.atan2(d[1], Math.hypot(d[0], d[2])) * 180 / Math.PI,
  };
}

function builder(send, cx) {
  const abs = (x, z) => [cx + x, z];
  const fill = async (x1, y1, z1, x2, y2, z2, block, mode = '') => {
    const [ax1, az1] = abs(x1, z1), [ax2, az2] = abs(x2, z2);
    return send(`fill ${ax1} ${y1} ${az1} ${ax2} ${y2} ${az2} minecraft:${block}${mode ? ` ${mode}` : ''}`);
  };
  const set = async (x, y, z, block) => {
    const [ax, az] = abs(x, z);
    return send(`setblock ${ax} ${y} ${az} minecraft:${block}`);
  };
  const column = (x, y, z, h, block) => fill(x, y, z, x, y + h - 1, z, block);
  const beamX = (x1, x2, y, z, block) => fill(x1, y, z, x2, y, z, block);
  const beamZ = (x, y, z1, z2, block) => fill(x, y, z1, x, y, z2, block);
  const hollow = async (x1, y1, z1, x2, y2, z2, block) => {
    await fill(x1,y1,z1,x2,y2,z1,block); await fill(x1,y1,z2,x2,y2,z2,block);
    await fill(x1,y1,z1,x1,y2,z2,block); await fill(x2,y1,z1,x2,y2,z2,block);
  };
  const stairs = async (x, y, z, count, dx, dz, width, block) => {
    for (let i = 0; i < count; i++) {
      const px = x + dx*i, pz = z + dz*i;
      if (dx) await fill(px, y+i, pz, px, y+i, pz+width-1, block);
      else await fill(px, y+i, pz, px+width-1, y+i, pz, block);
    }
  };
  return { fill, set, column, beamX, beamZ, hollow, stairs };
}

async function commonBase(b, ground = 'stone') {
  await b.fill(-31, 61, -27, 31, 62, 27, 'dirt');
  await b.fill(-31, 63, -27, 31, 63, 27, ground);
  // Four chunks keep every /fill below Minecraft's 32,768 block limit.
  await b.fill(-31, 64, -27, 0, 90, -1, 'air');
  await b.fill(-31, 64, 0, 0, 90, 27, 'air');
  await b.fill(1, 64, -27, 31, 90, -1, 'air');
  await b.fill(1, 64, 0, 31, 90, 27, 'air');
  // Broken patterned boundary gives far-range corners without enclosing view.
  for (let x = -30; x <= 30; x += 6) {
    const h = 2 + ((x + 30) / 6) % 3;
    await b.fill(x, 64, 25, x + 2, 63 + h, 26, x % 12 ? 'cobblestone' : 'andesite');
  }
}

async function buildQuarry(b) {
  await commonBase(b, 'andesite');
  await b.fill(-10,60,-1,12,62,18,'tuff'); await b.fill(-7,59,2,9,59,15,'deepslate');
  for (let i=0;i<4;i++) await b.fill(-25+i*3,64+i,-3,-18+i*3,64+i,20,'stone');
  await b.stairs(-12,60,3,7,1,0,5,'cobblestone');
  await b.hollow(15,64,5,24,72,13,'cracked_stone_bricks');
  await b.fill(17,66,5,19,69,5,'air'); await b.fill(21,65,5,23,68,5,'air');
  for (let i=0;i<7;i++) await b.set(-5+i*2,60,13-i,'iron_ore');
  await b.column(3,64,-2,11,'cut_copper'); await b.beamX(3,18,74,-2,'waxed_copper_block');
  for (let i=0;i<4;i++) await b.column(18+i*2,64,18-i*3,2+i,'cobblestone');
}

async function buildAqueduct(b) {
  await commonBase(b, 'stone_bricks');
  const xs=[-24,-13,1,17,27];
  for (let i=0;i<xs.length;i++) await b.fill(xs[i],64,10,xs[i]+2,73+(i%2),12,i===2?'mossy_stone_bricks':'cracked_stone_bricks');
  for (let i=0;i<xs.length-1;i++) await b.fill(xs[i],73+(i%2),10,xs[i+1]+2,75+(i%2),12,i===1?'bricks':'stone_bricks');
  await b.fill(-8,63,-1,8,63,13,'polished_andesite');
  await b.hollow(-8,64,-1,8,65,13,'granite'); await b.fill(-5,64,2,5,65,10,'air');
  await b.fill(-27,64,18,-8,67,22,'stone_bricks'); await b.fill(-22,64,18,-18,66,18,'air');
  await b.column(24,64,18,14,'deepslate_tiles'); await b.column(20,64,21,8,'bricks');
  for (const [x,z,h] of [[-19,-2,3],[-13,3,5],[13,1,4],[19,-3,6]]) await b.column(x,64,z,h,'mossy_cobblestone');
}

async function buildWorkshop(b) {
  await commonBase(b, 'gravel');
  for (let i=0;i<6;i++) await b.fill(-29+i*4,64+i*2,2,-24+i*4,65+i*2,25,'stone');
  await b.stairs(-22,64,-2,10,1,0,4,'cobblestone');
  await b.stairs(8,69,1,9,-1,0,4,'stone_bricks');
  await b.hollow(7,64,8,19,71,20,'bricks'); await b.fill(11,64,8,15,68,8,'air');
  await b.fill(6,72,7,20,73,21,'weathered_copper');
  await b.hollow(20,64,-2,28,69,8,'spruce_planks'); await b.fill(22,64,-2,25,67,-2,'air');
  await b.column(24,70,3,9,'bricks');
  for (const x of [-16,-8,3,15]) { await b.column(x,64,-8,6+(x+16)%4,'spruce_log'); await b.beamZ(x,69,-8,-2,'spruce_log'); }
}

async function buildTemple(b) {
  await commonBase(b, 'sandstone');
  await b.fill(-16,60,0,16,62,22,'tuff');
  for (let i=0;i<4;i++) await b.fill(-15+i*3,60+i,-1+i*2,15-i*3,60+i,21-i*2,i===2?'red_sandstone':'cut_sandstone');
  await b.fill(-10,64,5,10,65,18,'smooth_sandstone');
  for (const [x,z,h] of [[-9,6,9],[-9,17,7],[9,17,10]]) await b.column(x,66,z,h,'chiseled_sandstone');
  await b.beamX(-10,10,75,17,'lapis_block'); await b.beamX(-10,10,74,17,'cut_sandstone');
  await b.column(21,64,9,14,'red_sandstone'); await b.set(21,78,9,'gold_block');
  await b.beamX(-24,22,68,-1,'dark_oak_planks');
  for (const x of [-24,-8,8,22]) await b.column(x,64,-1,5,'dark_oak_log');
  for (const [x,z] of [[-23,15],[-18,20],[18,22],[25,18]]) await b.column(x,64,z,2+(x+z+40)%4,'tuff');
}

async function buildFoundry(b) {
  await commonBase(b, 'polished_andesite');
  await b.hollow(-23,64,5,-11,76,17,'deepslate_bricks'); await b.fill(-21,67,5,-13,73,5,'blackstone');
  await b.hollow(14,64,10,22,72,18,'bricks'); await b.fill(16,66,10,20,69,10,'blackstone');
  await b.column(-17,77,11,8,'bricks'); await b.column(18,73,14,12,'deepslate_tiles');
  await b.fill(-20,72,-2,21,73,1,'iron_block');
  for (const x of [-20,-7,6,21]) await b.column(x,64,0,9,'polished_blackstone_bricks');
  await b.stairs(-7,64,-7,9,1,0,3,'cut_copper');
  await b.fill(-2,61,8,10,62,19,'blackstone'); await b.hollow(-2,63,8,10,64,19,'yellow_concrete');
  await b.beamX(-17,25,78,22,'oxidized_copper'); await b.column(25,64,22,15,'oxidized_copper');
}

async function buildCanyon(b) {
  await commonBase(b, 'stone');
  for (let i=0;i<5;i++) {
    await b.fill(-30,64+i*2,-2+i*2,-15+i*2,65+i*2,25,'cobblestone');
    await b.fill(15-i*2,64+i*2,1+i*2,30,65+i*2,25,i===3?'calcite':'deepslate');
  }
  await b.fill(-26,71,7,25,72,11,'stone_bricks');
  for (const x of [-26,-10,8,25]) await b.column(x,64,9,8,'stone_bricks');
  await b.fill(-17,67,18,18,67,21,'oak_planks');
  for (let x=-17;x<=18;x+=5) { await b.column(x,64,18,4,'oak_log'); await b.column(x,64,21,4,'oak_log'); await b.beamZ(x,70,18,21,'oak_log'); }
  await b.stairs(-4,64,-8,8,0,1,5,'mossy_cobblestone');
  await b.hollow(20,72,17,28,79,24,'stone'); await b.fill(22,72,17,26,76,17,'air');
}

async function buildMarket(b) {
  await commonBase(b, 'cobblestone');
  const houses=[[-27,5,-15,15,7,'oak_planks'],[-9,10,2,20,9,'bricks'],[11,8,23,21,8,'cyan_terracotta'],[-25,-13,-11,-4,6,'spruce_planks'],[10,-12,26,-3,7,'stone_bricks']];
  for (const [x1,z1,x2,z2,h,block] of houses) {
    await b.hollow(x1,64,z1,x2,63+h,z2,block); await b.fill(x1+2,64,z1,x1+4,67,z1,'air');
    await b.fill(x1-1,64+h,z1-1,x2+1,65+h,z2+1,block==='cyan_terracotta'?'dark_prismarine':'dark_oak_planks');
  }
  await b.column(22,64,17,16,'stone_bricks'); await b.set(22,80,17,'gold_block');
  for (const [x,z,c] of [[-12,-4,'yellow_wool'],[0,-2,'cyan_wool'],[-3,10,'red_wool']]) {
    for (const dx of [0,6]) for (const dz of [0,4]) await b.column(x+dx,64,z+dz,4,'oak_fence');
    await b.fill(x,68,z,x+6,68,z+4,c);
  }
  await b.fill(-18,71,2,8,72,4,'spruce_planks');
}

async function buildShrine(b) {
  await commonBase(b, 'moss_block');
  for (let i=0;i<9;i++) {
    const x=-24+i*6, z=-8+Math.round(7*Math.sin(i));
    await b.fill(x,63,z-2,x+5,63,z+2,i%2?'coarse_dirt':'gravel');
  }
  await b.column(-10,64,3,8,'dark_oak_log'); await b.column(-2,64,3,5,'dark_oak_log'); await b.beamX(-11,-1,71,3,'red_terracotta');
  await b.hollow(12,64,10,26,70,22,'stone_bricks'); await b.fill(16,64,10,21,68,10,'air');
  await b.fill(10,71,8,28,73,24,'dark_oak_planks'); await b.fill(14,74,12,24,74,20,'red_terracotta');
  for (const [x,z,h] of [[-22,15,5],[-14,20,7],[2,17,4]]) { await b.column(x,64,z,h,'mossy_cobblestone'); await b.set(x,64+h,z,'shroomlight'); }
  for (const [x,z,h] of [[-25,-2,8],[0,-10,11],[7,2,7],[24,-5,12]]) { await b.column(x,64,z,h,'spruce_log'); await b.fill(x-2,63+h,z-2,x+2,66+h,z+2,'spruce_leaves'); }
  await b.fill(-1,64,11,9,67,18,'mossy_cobblestone'); await b.fill(2,65,14,7,69,19,'tuff');
}

async function buildDock(b) {
  await commonBase(b, 'gravel');
  await b.fill(-17,60,-1,17,62,23,'deepslate_tiles');
  await b.fill(-22,63,-3,-18,72,25,'stone_bricks'); await b.fill(18,63,-3,22,76,25,'stone_bricks');
  await b.stairs(-17,60,-5,7,0,1,35,'polished_andesite');
  // Uneven ship ribs with one deliberate missing station.
  for (const z of [3,7,12,19]) {
    const half=4+Math.floor(z/7);
    await b.column(-half,63,z,5,'stripped_spruce_log'); await b.column(half,63,z,5,'stripped_spruce_log');
    await b.beamX(-half,half,67,z,'oak_log');
  }
  await b.beamZ(0,68,1,23,'oak_log');
  await b.column(24,64,0,16,'cut_copper'); await b.beamX(3,25,78,0,'waxed_copper_block'); await b.column(4,64,0,8,'iron_block');
  await b.hollow(-30,64,10,-23,72,23,'spruce_planks'); await b.fill(-28,64,10,-25,68,10,'air');
  await b.hollow(23,64,15,30,76,25,'stone_bricks'); await b.fill(25,64,15,28,70,15,'air');
}

async function buildCavern(b) {
  await commonBase(b, 'basalt');
  await b.fill(-30,64,19,30,81,27,'blackstone');
  await b.fill(-30,64,-4,-26,82,27,'tuff'); await b.fill(26,64,-4,30,82,27,'deepslate');
  const pillars=[[-16,7,5,15,'basalt'],[0,13,7,18,'tuff'],[16,4,4,13,'polished_basalt']];
  for (const [x,z,r,h,block] of pillars) {
    await b.fill(x-r,64,z-r,x+r,63+h,z+r,block);
    await b.fill(x-r+2,64,z-r+2,x+r-2,63+h,z+r-2,'air');
  }
  await b.fill(-24,69,20,22,72,23,'calcite');
  await b.stairs(-20,64,-4,10,1,1,4,'cobbled_deepslate');
  await b.hollow(18,66,19,27,75,26,'blackstone'); await b.fill(20,66,19,25,72,19,'air');
  for (const [x,y,z] of [[-9,65,2],[-4,66,8],[8,65,1],[12,68,14],[21,67,8]]) {
    await b.set(x,y,z,'amethyst_block'); await b.set(x,y+1,z,'amethyst_cluster');
  }
  for (const [x,z,h] of [[-22,15,6],[-7,20,9],[9,21,5],[23,15,8]]) await b.column(x,64,z,h,'pointed_dripstone');
  for (const [x,z] of [[-20,0],[-10,15],[2,4],[14,17],[23,1]]) await b.set(x,66,z,'shroomlight');
}

async function main() {
  const output = path.resolve(process.argv[2] || '');
  if (!process.argv[2]) throw new Error('usage: node capture_navigation_rich_scene_stills.js OUTPUT_DIR');
  if (fs.existsSync(output)) throw new Error(`Refusing to overwrite ${output}`);
  fs.mkdirSync(output, { recursive: true });
  const rcon = await Rcon.connect({ host: '127.0.0.1', port: 25721, password: 'research', timeout: 30000 });
  const initLog = path.join(output, 'initialization.jsonl');
  const send = async (command) => {
    const response = await rcon.send(command);
    fs.appendFileSync(initLog, JSON.stringify({ epoch_ms: Date.now(), command, response: String(response) }) + '\n');
    if (/unknown or incomplete|incorrect argument|expected integer|not a valid|position is not loaded|too many blocks|no player was found/i.test(String(response))) throw new Error(`${command} -> ${response}`);
    return response;
  };
  const captures=[];
  try {
    const online = await send('list');
    if (!online.includes('CameraAlpha') || !online.includes('CameraBravo')) throw new Error(`native clients unavailable: ${online}`);
    for (const command of ['difficulty peaceful','gamerule doDaylightCycle false','gamerule doWeatherCycle false','weather clear','time set noon','kill @e[type=!player]','gamemode survival CameraAlpha','gamemode survival CameraBravo','clear CameraAlpha','clear CameraBravo']) await send(command);
    await send('team add mwb3nav'); await send('team modify mwb3nav collisionRule never'); await send('team modify mwb3nav nametagVisibility never');
    await send('team join mwb3nav CameraAlpha'); await send('team join mwb3nav CameraBravo');
    for (let index=0; index<scenes.length; index++) {
      const spec=scenes[index], cx=4000+index*96;
      const caseDir=path.join(output,spec.id); fs.mkdirSync(caseDir);
      await send(`forceload add ${cx-32} -28 ${cx+32} 28`);
      const b=builder(send,cx); await spec.build(b);
      const poses={};
      for (const [role,name,offset] of [['alpha','CameraAlpha',spec.alpha],['bravo','CameraBravo',spec.bravo]]) {
        const origin=[cx+offset[0],offset[1],offset[2]], focus=[cx+spec.focus[0],spec.focus[1],spec.focus[2]];
        const rotation=aimPose(origin,focus);
        await send(`tp ${name} ${origin.join(' ')} ${rotation.yaw} ${rotation.pitch}`);
        poses[role]={position:origin,focus,native_yaw_deg:rotation.yaw,native_pitch_deg:rotation.pitch};
      }
      await wait(4500);
      const images={};
      for (const [role,name,display] of [['alpha','CameraAlpha',':130.0'],['bravo','CameraBravo',':131.0']]) {
        const file=path.join(caseDir,`${role}_first_frame.png`);
        execFileSync('ffmpeg',['-nostdin','-v','error','-f','x11grab','-video_size','1280x720','-i',display,'-frames:v','1',file]);
        images[role]={file:path.relative(output,file),sha256:sha256(file),actual_position:await send(`data get entity ${name} Pos`),actual_rotation:await send(`data get entity ${name} Rotation`),on_ground:await send(`data get entity ${name} OnGround`)};
      }
      const row={case_id:spec.id,name:spec.name,slug:spec.slug,center_x:cx,poses,images,trajectory_collected:false};
      fs.writeFileSync(path.join(caseDir,'capture.json'),JSON.stringify(row,null,2)+'\n'); captures.push(row);
      console.log(`Captured ${spec.id} ${spec.name}`);
    }
    fs.writeFileSync(path.join(output,'capture_manifest.json'),JSON.stringify({protocol:'mwb3_navigation_rich_scene_stills_v1',source:'native Minecraft initialization and native client X11 frames',resolution:[1280,720],scene_count:captures.length,trajectory_collected:false,captures},null,2)+'\n');
  } finally { await rcon.end().catch(()=>{}); }
}

if (require.main === module) {
  main().catch((error)=>{ console.error(error.stack||error.message); process.exit(1); });
}

module.exports = { scenes, aimPose, builder };
