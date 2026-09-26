const { stopAll } = require("../../primitives/movement");

const DEFAULT_MOVE_TICKS = 60;
const DEFAULT_LOOK_TICKS = 40;
const MOVE_BRAKING_MARGIN = 0.3;
const MOVE_LOOK_BRAKING_MARGIN = 0.35;

function degreesToRadians(value) {
  return (value * Math.PI) / 180;
}

function radiansToDegrees(value) {
  return (value * 180) / Math.PI;
}

function normalizeDegrees(value) {
  let result = value % 360;
  if (result < 0) result += 360;
  return result;
}

function round(value, digits = 3) {
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function mineflayerYawToSpec(yawRadians) {
  return normalizeDegrees(radiansToDegrees(yawRadians + Math.PI));
}

function poseFromEntity(entity) {
  if (!entity) return null;
  return {
    position: [
      round(entity.position.x),
      round(entity.position.y),
      round(entity.position.z),
    ],
    yaw_deg: round(mineflayerYawToSpec(entity.yaw)),
    pitch_deg: round(radiansToDegrees(entity.pitch)),
  };
}

function clonePose(pose) {
  return {
    position: [...pose.position],
    yaw_deg: pose.yaw_deg,
    pitch_deg: pose.pitch_deg || 0,
  };
}

function createExpectedAgents({ scene, caseSpec, origin, roleNames }) {
  const preset = scene.pose_presets[caseSpec.setup.pose_preset];
  if (!preset) {
    throw new Error(
      `Unknown pose preset for ${caseSpec.id}: ${caseSpec.setup.pose_preset}`,
    );
  }
  const overrides = caseSpec.setup.overrides || {};
  return Object.fromEntries(
    ["A", "B"].map((role) => {
      const pose = clonePose({ ...preset[role], ...(overrides[role] || {}) });
      pose.position[0] += origin.x;
      pose.position[1] += origin.y;
      pose.position[2] += origin.z;
      return [role, { name: roleNames[role], ...pose }];
    }),
  );
}

function expectedAfterAction(startPose, action) {
  const expected = clonePose(startPose);
  if (action.type === "LOOK" || action.type === "MOVE_LOOK") {
    expected.yaw_deg = normalizeDegrees(
      expected.yaw_deg + Number(action.yaw_delta_deg || 0),
    );
    expected.pitch_deg = Math.max(
      -90,
      Math.min(90, expected.pitch_deg + Number(action.pitch_delta_deg || 0)),
    );
  }
  return expected;
}

function horizontalDistance(start, end) {
  return Math.hypot(end.x - start.x, end.z - start.z);
}

function trajectoryDistance(trajectory) {
  let distance = 0;
  for (let index = 1; index < trajectory.length; index += 1) {
    const previous = trajectory[index - 1].position;
    const current = trajectory[index].position;
    distance += Math.hypot(
      current[0] - previous[0],
      current[2] - previous[2],
    );
  }
  return distance;
}

function clampPitch(pitch) {
  return Math.max(-Math.PI / 2, Math.min(Math.PI / 2, pitch));
}

function angularDelta(current, previous) {
  return ((current - previous + 3 * Math.PI) % (2 * Math.PI)) - Math.PI;
}

function setLinearLookStep(
  bot,
  startYaw,
  startPitch,
  yawDelta,
  pitchDelta,
  step,
  steps,
) {
  const fraction = step / steps;
  return bot.look(
    startYaw + yawDelta * fraction,
    clampPitch(startPitch + pitchDelta * fraction),
    true,
    360,
    360,
    false,
  );
}

async function executeLinearLook(
  bot,
  turnClock,
  startYaw,
  startPitch,
  yawDelta,
  pitchDelta,
  durationTicks,
  afterTick = null,
) {
  const startTick = turnClock.ticks;
  let lookError = null;
  const onPhysicsTick = () => {
    const elapsedTicks = turnClock.ticks - startTick;
    if (elapsedTicks < 1 || elapsedTicks > durationTicks) return;
    setLinearLookStep(
      bot,
      startYaw,
      startPitch,
      yawDelta,
      pitchDelta,
      elapsedTicks,
      durationTicks,
    ).catch((error) => {
      lookError ||= error;
    });
    if (afterTick) afterTick();
  };

  bot.on("physicsTick", onPhysicsTick);
  try {
    while (turnClock.ticks - startTick < durationTicks) {
      await bot.waitForTicks(1);
    }
    if (lookError) throw lookError;
  } finally {
    bot.removeListener("physicsTick", onPhysicsTick);
  }
}

async function waitUntilEpoch(epochMs) {
  while (Date.now() < epochMs) {
    const remaining = epochMs - Date.now();
    await new Promise((resolve) => setTimeout(resolve, Math.min(20, remaining)));
  }
}

async function padToTurnEnd(bot, turnClock, turnTicks) {
  stopAll(bot);
  while (turnClock.ticks < turnTicks) await bot.waitForTicks(1);
}

function nativeControls(bot) {
  return Object.fromEntries(
    ["forward", "back", "left", "right", "jump", "sprint", "sneak"].map(
      (key) => [key, Boolean(bot.controlState[key])],
    ),
  );
}

function sampleTrajectory(bot, turnClock, trajectory, previousPose, sampleKind) {
  const pose = poseFromEntity(bot.entity);
  const epochMs = Date.now();
  const sample = {
    sample_index: trajectory.length,
    sample_kind: sampleKind,
    tick_offset: turnClock.ticks,
    epoch_ms: epochMs,
    epochTime: epochMs / 1000,
    minecraft_tick: Number(bot.time?.age),
    action: {
      ...nativeControls(bot),
      camera: previousPose
        ? [
            angularDelta(bot.entity.yaw, previousPose.yaw),
            bot.entity.pitch - previousPose.pitch,
          ]
        : [0, 0],
    },
    ...pose,
  };
  trajectory.push(sample);
  return { yaw: bot.entity.yaw, pitch: bot.entity.pitch };
}

async function executeMove(bot, action, turnClock, turnTicks, trajectory) {
  const allowedControls = new Set(["forward", "back", "left", "right"]);
  if (!allowedControls.has(action.control)) {
    throw new Error(`Unsupported MOVE control: ${action.control}`);
  }
  const targetDistance = Number(action.distance || 3);
  const start = bot.entity.position.clone();
  stopAll(bot);
  bot.setControlState(action.control, true);
  while (turnClock.ticks < turnTicks - 2) {
    if (trajectoryDistance(trajectory) >= targetDistance - MOVE_BRAKING_MARGIN) {
      break;
    }
    await bot.waitForTicks(1);
  }
  stopAll(bot);
  const releaseDistance = trajectoryDistance(trajectory);
  await padToTurnEnd(bot, turnClock, turnTicks);
  const observedDistance = trajectoryDistance(trajectory);
  return {
    status:
      Math.abs(observedDistance - targetDistance) <= 0.35
        ? "completed"
        : "distance_error",
    requested_distance: targetDistance,
    observed_distance: round(observedDistance),
    release_distance: round(releaseDistance),
    displacement: round(horizontalDistance(start, bot.entity.position)),
  };
}

async function executeLook(bot, action, turnClock, turnTicks, trajectory) {
  const startYaw = bot.entity.yaw;
  const startPitch = bot.entity.pitch;
  const yawDelta = degreesToRadians(Number(action.yaw_delta_deg || 0));
  const pitchDelta = degreesToRadians(Number(action.pitch_delta_deg || 0));
  const steps = Math.max(1, turnTicks);
  await executeLinearLook(
    bot,
    turnClock,
    startYaw,
    startPitch,
    yawDelta,
    pitchDelta,
    steps,
  );
  await padToTurnEnd(bot, turnClock, turnTicks);
  return { status: "completed" };
}

async function executeMoveLook(bot, action, turnClock, turnTicks, trajectory) {
  const allowedControls = new Set(["forward", "back", "left", "right"]);
  if (!allowedControls.has(action.control)) {
    throw new Error(`Unsupported MOVE_LOOK control: ${action.control}`);
  }
  const start = bot.entity.position.clone();
  const startYaw = bot.entity.yaw;
  const startPitch = bot.entity.pitch;
  const targetDistance = Number(action.distance || 3);
  const yawDelta = degreesToRadians(Number(action.yaw_delta_deg || 0));
  const pitchDelta = degreesToRadians(Number(action.pitch_delta_deg || 0));

  stopAll(bot);
  bot.setControlState(action.control, true);
  const steps = Math.max(1, turnTicks);
  let releaseDistance = null;
  await executeLinearLook(
    bot,
    turnClock,
    startYaw,
    startPitch,
    yawDelta,
    pitchDelta,
    steps,
    () => {
      if (releaseDistance !== null) return;
      const distance = trajectoryDistance(trajectory);
      if (distance >= targetDistance - MOVE_LOOK_BRAKING_MARGIN) {
        stopAll(bot);
        releaseDistance = distance;
      }
    },
  );
  stopAll(bot);
  if (releaseDistance === null) {
    releaseDistance = trajectoryDistance(trajectory);
  }
  await padToTurnEnd(bot, turnClock, turnTicks);
  const observedDistance = trajectoryDistance(trajectory);
  return {
    status:
      Math.abs(observedDistance - targetDistance) <= 0.35
        ? "completed"
        : "distance_error",
    requested_distance: targetDistance,
    observed_distance: round(observedDistance),
    release_distance: round(releaseDistance),
    displacement: round(horizontalDistance(start, bot.entity.position)),
  };
}

async function executeNativeAction({
  bot,
  action,
  startTick,
  startEpochMs,
  turnTicks,
}) {
  await waitUntilEpoch(startEpochMs);
  const actualStartEpochMs = Date.now();
  const actualStartTick = bot.time.age;
  const turnClock = { ticks: 0 };
  const trajectory = [];
  let previousPose = sampleTrajectory(
    bot,
    turnClock,
    trajectory,
    null,
    "pre_action_anchor",
  );
  const onPhysicsTick = () => {
    turnClock.ticks += 1;
    previousPose = sampleTrajectory(
      bot,
      turnClock,
      trajectory,
      previousPose,
      "consumed_physics_tick",
    );
  };
  bot.on("physicsTick", onPhysicsTick);
  try {
    let result;
    if (action.type === "MOVE") {
      result = await executeMove(
        bot,
        action,
        turnClock,
        turnTicks || DEFAULT_MOVE_TICKS,
        trajectory,
      );
    } else if (action.type === "LOOK") {
      result = await executeLook(
        bot,
        action,
        turnClock,
        turnTicks || DEFAULT_LOOK_TICKS,
        trajectory,
      );
    } else if (action.type === "MOVE_LOOK") {
      result = await executeMoveLook(
        bot,
        action,
        turnClock,
        turnTicks || DEFAULT_MOVE_TICKS,
        trajectory,
      );
    } else if (action.type === "IDLE") {
      await padToTurnEnd(
        bot,
        turnClock,
        turnTicks || Number(action.duration_ticks || DEFAULT_LOOK_TICKS),
      );
      result = { status: "idle" };
    } else {
      throw new Error(`Unsupported V3 native action type: ${action.type}`);
    }
    return {
      ...result,
      requested_start_tick: startTick,
      requested_start_epoch_ms: startEpochMs,
      actual_start_tick: actualStartTick,
      actual_start_epoch_ms: actualStartEpochMs,
      actual_end_epoch_ms: Date.now(),
      elapsed_physics_ticks: turnClock.ticks,
      pose: poseFromEntity(bot.entity),
      trajectory,
    };
  } finally {
    stopAll(bot);
    bot.removeListener("physicsTick", onPhysicsTick);
  }
}

module.exports = {
  createExpectedAgents,
  executeNativeAction,
  expectedAfterAction,
  normalizeDegrees,
  poseFromEntity,
  round,
};
