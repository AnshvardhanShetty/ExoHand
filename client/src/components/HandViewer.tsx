/**
 * HandViewer v12 — Animation-clip driven with track rebinding.
 *
 * The GLB's "Open/Close" clip targets IK Ctrl bones. Three.js can't solve IK,
 * so we:
 *   1. Fix track binding paths (strip prefixes / handle Blender pipes)
 *   2. Sample Ctrl bone poses at open (t=0) and closed (t=duration)
 *   3. Map Ctrl bone deltas → corresponding deform bones
 *   4. Per-frame: slerp all deform bones in sync by x (0=open, 1=closed)
 *
 * CANONICAL:  0°=OPEN  110°=REST  180°=CLOSE
 *   x = angle/180
 */

import React, { useRef, useEffect, useState, useCallback, useMemo } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { useGLTF, OrbitControls } from "@react-three/drei";
import * as THREE from "three";

/* ================================================================
   Constants
   ================================================================ */

const REST_ANGLE = 110;
const X_REST = 1 - REST_ANGLE / 180; // 0.389 (GLB: x=0 closed, x=1 open)

export type DriverMode = "DEMO" | "LIVE";

export interface LiveState {
  stateCmd: string;
  motionLocked: boolean;
  lockRemainingMs: number;
  targetAngle: number;
}

export interface ExerciseInfo {
  id?: string;
  startAngle: number;
  targetAngle: number;
  holdSeconds: number;
  reps?: number;
  name: string;
  category?: "close" | "open" | "combined";
}

interface DebugLine { label: string; value: string; color?: string }

export interface HeroPose {
  rotX: number; rotY: number; rotZ: number;
  posX: number; posY: number; posZ: number;
  scale: number;
}

const DEFAULT_HERO: HeroPose = {
  rotX: 146, rotY: -57, rotZ: 51,
  posX: 0, posY: 0, posZ: 0,
  scale: 1,
};

/* ================================================================
   Demo drivers
   ================================================================ */

/** Convert canonical angle to slerp x (GLB: x=0 closed, x=1 open) */
function angleToX(angle: number): number {
  return 1 - angle / 180;
}

/** Generic demo: slow sine oscillation between closed and open, ~10s period */
function demoProgressGeneric(t: number): number {
  return Math.sin(t * 0.6) * 0.5 + 0.5;
}

/** Exercise-specific demo: starts at rest for 1.5s, then loops */
const DEMO_LEAD_IN = 1.5; // seconds at rest before first rep

function demoProgressExercise(t: number, exercise: ExerciseInfo): number {
  const startX = angleToX(exercise.startAngle);

  // Hold at rest for the lead-in period
  if (t < DEMO_LEAD_IN) return startX;
  const t2 = t - DEMO_LEAD_IN;

  // Combined: continuous open ↔ close without stopping at rest
  if (exercise.category === "combined") {
    const openX = angleToX(0);    // 1.0 = fully open
    const closeX = angleToX(180); // 0.0 = fully closed
    // First 1.5s: smooth transition from rest → closed (entry ramp)
    const rampDur = 1.5;
    if (t2 < rampDur) return startX + (closeX - startX) * (t2 / rampDur);
    const t3 = t2 - rampDur;
    // Then loop: 1.5s close→open + 0.5s hold + 1.5s open→close + 0.5s hold = 4s
    const period = 4;
    const p = t3 % period;
    if (p < 1.5) return closeX + (openX - closeX) * (p / 1.5);           // close → open
    if (p < 2.0) return openX;                                             // hold open
    if (p < 3.5) return openX + (closeX - openX) * ((p - 2.0) / 1.5);   // open → close
    return closeX;                                                          // hold closed
  }

  const targetX = angleToX(exercise.targetAngle);
  const hold = exercise.holdSeconds || 1.5;
  // 1.5s move + hold + 1.5s return + 1s rest
  const period = 1.5 + hold + 1.5 + 1;
  const phase = t2 % period;
  const moveEnd = 1.5;
  const holdEnd = moveEnd + hold;
  const returnEnd = holdEnd + 1.5;
  if (phase < moveEnd) return startX + (targetX - startX) * (phase / 1.5);                    // move to target
  if (phase < holdEnd) return targetX;                                                          // hold at target
  if (phase < returnEnd) return targetX + (startX - targetX) * ((phase - holdEnd) / 1.5);     // return to start
  return startX;                                                                                // rest
}

/** Map motor angle (110=open, 180=close) → canonical angle (0=open, 180=close) */
function liveTarget(state: LiveState): number {
  const MOTOR_OPEN = 110;
  const MOTOR_CLOSE = 180;
  const canonical = ((state.targetAngle - MOTOR_OPEN) / (MOTOR_CLOSE - MOTOR_OPEN)) * 180;
  return angleToX(canonical);
}

/* ================================================================
   Fix animation track names for Three.js PropertyBinding
   Sketchfab/Blender exports may use paths like:
     "Armature|bone_name.quaternion"
     "Armature/bone_name.quaternion"
   Three.js needs just "bone_name.quaternion" when mixer root = scene
   ================================================================ */

function fixTrackNames(clip: THREE.AnimationClip, root: THREE.Object3D): {
  fixed: THREE.AnimationClip;
  log: string[];
} {
  const log: string[] = [];
  const newTracks: THREE.KeyframeTrack[] = [];

  // Build a set of all object names in the scene
  const sceneNames = new Set<string>();
  root.traverse((obj) => sceneNames.add(obj.name));

  for (const track of clip.tracks) {
    const origName = track.name;

    // Split into object path and property: "path.property" or "path[index].property"
    // Three.js track format: "objectName.property" or "objectPath.property"
    const dotMatch = origName.match(/^(.+)\.(position|quaternion|scale|morphTargetInfluences.*)$/);
    if (!dotMatch) {
      log.push(`skip: ${origName} (no property match)`);
      continue;
    }

    let objPath = dotMatch[1];
    const property = dotMatch[2];

    // Strip Blender pipe separator: "Armature|bone" → "bone"
    if (objPath.includes("|")) {
      objPath = objPath.split("|").pop()!;
    }

    // Strip path separators: "Armature/bone" → "bone"
    if (objPath.includes("/")) {
      objPath = objPath.split("/").pop()!;
    }

    // Check if this object exists in the scene
    const newName = `${objPath}.${property}`;
    if (sceneNames.has(objPath)) {
      // Clone the track with the fixed name
      const newTrack = track.clone();
      newTrack.name = newName;
      newTracks.push(newTrack);
    } else {
      log.push(`miss: ${origName} → ${objPath} not found`);
    }
  }

  log.unshift(`Tracks: ${clip.tracks.length} original → ${newTracks.length} bound`);

  const fixed = new THREE.AnimationClip(clip.name, clip.duration, newTracks);
  return { fixed, log };
}

/* ================================================================
   Bone pose types for open/closed sampling
   ================================================================ */

interface BonePose {
  quaternion: THREE.Quaternion;
  position: THREE.Vector3;
}

/* ================================================================
   Module-level bind pose cache — survives component remounts.
   First mount captures true bind pose from the fresh GLB scene.
   Subsequent mounts reuse it (since useGLTF returns cached scene
   with stale bone positions).
   ================================================================ */

const bindPoseCache = new Map<string, BonePose>();

/* ================================================================
   FloorGrid — shader-based infinite grid beneath the hand
   ================================================================ */

const gridVertexShader = `
  varying vec3 vWorldPos;
  void main() {
    vec4 worldPos = modelMatrix * vec4(position, 1.0);
    vWorldPos = worldPos.xyz;
    gl_Position = projectionMatrix * viewMatrix * worldPos;
  }
`;

const gridFragmentShader = `
  varying vec3 vWorldPos;
  uniform float uGridSize;
  uniform float uSubGridSize;
  uniform vec3 uColor;
  uniform vec3 uSubColor;
  uniform float uFade;

  float gridLine(vec2 p, float size) {
    vec2 g = abs(fract(p / size - 0.5) - 0.5) / fwidth(p / size);
    return 1.0 - min(min(g.x, g.y), 1.0);
  }

  void main() {
    float dist = length(vWorldPos.xz);
    float fade = 1.0 - smoothstep(uFade * 0.3, uFade, dist);

    float majorLine = gridLine(vWorldPos.xz, uGridSize);
    float minorLine = gridLine(vWorldPos.xz, uSubGridSize);

    vec3 color = mix(uSubColor * minorLine * 0.4, uColor, majorLine);
    float alpha = max(majorLine * 0.5, minorLine * 0.18) * fade;

    gl_FragColor = vec4(color, alpha);
  }
`;

function FloorGrid() {
  const material = useMemo(() => {
    return new THREE.ShaderMaterial({
      vertexShader: gridVertexShader,
      fragmentShader: gridFragmentShader,
      uniforms: {
        uGridSize: { value: 1.0 },
        uSubGridSize: { value: 0.25 },
        uColor: { value: new THREE.Color(0.45, 0.52, 0.68) },
        uSubColor: { value: new THREE.Color(0.3, 0.36, 0.5) },
        uFade: { value: 8.0 },
      },
      transparent: true,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
  }, []);

  return (
    <mesh rotation-x={-Math.PI / 2} position-y={-1.8} material={material}>
      <planeGeometry args={[20, 20, 1, 1]} />
    </mesh>
  );
}

/* ================================================================
   HandScene — inner R3F component
   ================================================================ */

interface HandSceneProps {
  mode: DriverMode;
  liveState: LiveState | null;
  exercise: ExerciseInfo | null;
  heroPose: HeroPose;
  heroView: boolean;
  onDebugLines: (lines: DebugLine[]) => void;
  onTrackLog: (log: string[]) => void;
}

function HandScene({
  mode, liveState, exercise, heroPose, heroView,
  onDebugLines, onTrackLog,
}: HandSceneProps) {
  const gltf = useGLTF("/hand.glb");
  const { camera } = useThree();

  const rootRef = useRef<THREE.Object3D | null>(null);
  const wrapperRef = useRef<THREE.Group>(null);
  const cameraInitDone = useRef(false);
  const frameCount = useRef(0);
  const liveProgress = useRef(0);
  const exerciseDemoStart = useRef<number | null>(null);
  const cameraDist = useRef(4);

  // Open/closed pose maps for ALL bones (including Ctrl)
  const openPoseRef = useRef<Map<string, BonePose>>(new Map());
  const closedPoseRef = useRef<Map<string, BonePose>>(new Map());
  const bonesRef = useRef<THREE.Bone[]>([]);

  const debugInfo = useRef({
    clipName: "", tracks: 0, boundTracks: 0, duration: 0,
    skinnedMeshes: 0, boneCount: 0, diffBones: 0,
    method: "none",
  });

  /* ── INIT ── */
  useEffect(() => {
    // Always re-init: useGLTF caches the scene, so bones may be
    // in a stale slerp state from a previous mount.

    const root = gltf.scene;
    rootRef.current = root;

    // Material override
    let skinnedCount = 0;
    root.traverse((child) => {
      if ((child as THREE.Mesh).isMesh) {
        (child as THREE.Mesh).material = new THREE.MeshStandardMaterial({
          color: "#DDE3F2", roughness: 0.65, metalness: 0.05,
        });
        (child as THREE.Mesh).frustumCulled = false;
        if ((child as THREE.SkinnedMesh).isSkinnedMesh) skinnedCount++;
      }
    });

    let boneCount = 0;
    root.traverse((obj) => { if ((obj as THREE.Bone).isBone) boneCount++; });

    // Center + scale + camera — only once (bounding box depends on pose)
    if (!cameraInitDone.current) {
      cameraInitDone.current = true;
      root.updateMatrixWorld(true);
      const box = new THREE.Box3().setFromObject(root);
      const center = new THREE.Vector3();
      const size = new THREE.Vector3();
      box.getCenter(center);
      box.getSize(size);

      let maxDim = Math.max(size.x, size.y, size.z);
      if (maxDim < 1) {
        const sf = 2.5 / maxDim;
        root.scale.multiplyScalar(sf);
        root.updateMatrixWorld(true);
        box.setFromObject(root);
        box.getCenter(center);
        box.getSize(size);
        maxDim = Math.max(size.x, size.y, size.z);
      }

      root.position.sub(center);
      root.updateMatrixWorld(true);

      // Camera
      const cam = camera as THREE.PerspectiveCamera;
      const fovRad = (cam.fov * Math.PI) / 180;
      const dist = ((maxDim / 2) / Math.tan(fovRad / 2)) * 2.5;
      cameraDist.current = dist;
      cam.near = 0.001;
      cam.far = dist * 50;
      cam.updateProjectionMatrix();
    } // end cameraInitDone guard

    // Collect all bones
    const allBones: THREE.Bone[] = [];
    root.traverse((obj) => {
      if ((obj as THREE.Bone).isBone) allBones.push(obj as THREE.Bone);
    });
    bonesRef.current = allBones;

    // Log all bone names
    console.log("[HV] All bones:", allBones.map(b => b.name).join(", "));

    // Process animation — read keyframe data directly from tracks (bypass mixer)
    const clips = gltf.animations;
    if (clips && clips.length > 0) {
      const clip = clips[0];
      const trackLog: string[] = [];
      trackLog.push(`Clip "${clip.name}": ${clip.tracks.length} tracks, ${clip.duration.toFixed(2)}s`);

      // Build bone lookup by name
      const boneByName = new Map<string, THREE.Bone>();
      for (const bone of allBones) {
        boneByName.set(bone.name, bone);
      }

      // ── Bind pose: cache on first mount, reuse on subsequent mounts ──
      // useGLTF returns the same cached scene, so bone positions are
      // stale from the previous mount's slerping. We capture the true
      // bind pose once (first mount) and reuse it forever.
      const isFirstMount = bindPoseCache.size === 0;
      if (isFirstMount) {
        for (const bone of allBones) {
          bindPoseCache.set(bone.name, {
            quaternion: bone.quaternion.clone(),
            position: bone.position.clone(),
          });
        }
        console.log("[HV] Bind pose cached:", bindPoseCache.size, "bones");
      }

      // Build openPose from cache and reset bones to bind state
      const openPose = new Map<string, BonePose>();
      for (const bone of allBones) {
        const cached = bindPoseCache.get(bone.name);
        if (cached) {
          openPose.set(bone.name, {
            quaternion: cached.quaternion.clone(),
            position: cached.position.clone(),
          });
          // Reset actual bone to bind pose
          bone.quaternion.copy(cached.quaternion);
          bone.position.copy(cached.position);
        } else {
          openPose.set(bone.name, {
            quaternion: bone.quaternion.clone(),
            position: bone.position.clone(),
          });
        }
      }

      // closedPose starts as a copy of openPose
      const closedPose = new Map<string, BonePose>();
      for (const [name, pose] of openPose) {
        closedPose.set(name, { quaternion: pose.quaternion.clone(), position: pose.position.clone() });
      }

      // ── PASS 2: Build closedPose from last-keyframe values ──
      let tracksApplied = 0;
      let ctrlTracksApplied = 0;

      for (const track of clip.tracks) {
        const dotMatch = track.name.match(/^(.+)\.(position|quaternion|scale|morphTargetInfluences.*)$/);
        if (!dotMatch) continue;

        let objPath = dotMatch[1];
        const property = dotMatch[2];
        if (objPath.includes("|")) objPath = objPath.split("|").pop()!;
        if (objPath.includes("/")) objPath = objPath.split("/").pop()!;

        const valueSize = track.getValueSize();
        const numKeyframes = track.times.length;
        if (numKeyframes === 0) continue;

        const openValues = Array.from(track.values.slice(0, valueSize));
        const closedValues = Array.from(track.values.slice((numKeyframes - 1) * valueSize));

        let targetBoneName = objPath;
        let bone = boneByName.get(targetBoneName);

        const isCtrl = targetBoneName.toLowerCase().includes("ctrl");
        if (isCtrl) {
          const deformName = targetBoneName
            .replace(/^Ctrl_/i, "").replace(/^ctrl_/i, "")
            .replace(/_Ctrl$/i, "").replace(/_ctrl$/i, "");
          const deformBone = boneByName.get(deformName);
          if (deformBone) {
            bone = deformBone;
            targetBoneName = deformName;
            ctrlTracksApplied++;
          }
        }

        if (!bone) continue;

        if (property === "quaternion" && valueSize === 4) {
          const closedQ = new THREE.Quaternion(closedValues[0], closedValues[1], closedValues[2], closedValues[3]);
          const openQ = new THREE.Quaternion(openValues[0], openValues[1], openValues[2], openValues[3]);

          if (isCtrl) {
            const bindQ = openPose.get(targetBoneName)!.quaternion;
            const delta = openQ.clone().invert().multiply(closedQ);
            closedPose.set(targetBoneName, {
              ...closedPose.get(targetBoneName)!,
              quaternion: bindQ.clone().multiply(delta),
            });
          } else {
            closedPose.set(targetBoneName, {
              ...closedPose.get(targetBoneName)!,
              quaternion: closedQ,
            });
          }
          tracksApplied++;
        } else if (property === "position" && valueSize === 3) {
          const closedP = new THREE.Vector3(closedValues[0], closedValues[1], closedValues[2]);

          if (isCtrl) {
            const openP = new THREE.Vector3(openValues[0], openValues[1], openValues[2]);
            const bindP = openPose.get(targetBoneName)!.position;
            const delta = closedP.clone().sub(openP);
            closedPose.set(targetBoneName, {
              ...closedPose.get(targetBoneName)!,
              position: bindP.clone().add(delta),
            });
          } else {
            closedPose.set(targetBoneName, {
              ...closedPose.get(targetBoneName)!,
              position: closedP,
            });
          }
          tracksApplied++;
        }
      }

      // Count differing bones
      let diffCount = 0;
      const diffNames: string[] = [];
      for (const bone of allBones) {
        const op = openPose.get(bone.name);
        const cp = closedPose.get(bone.name);
        if (op && cp) {
          const quatDiff = op.quaternion.angleTo(cp.quaternion) > 0.01;
          const posDiff = op.position.distanceTo(cp.position) > 0.001;
          if (quatDiff || posDiff) {
            diffCount++;
            diffNames.push(bone.name);
          }
        }
      }

      trackLog.push(`Applied: ${tracksApplied} tracks (${ctrlTracksApplied} ctrl→deform)`);
      trackLog.push(`Diff bones: ${diffCount} — ${diffNames.slice(0, 10).join(", ")}${diffNames.length > 10 ? "..." : ""}`);
      console.log("[HV] Track processing:", trackLog.join(" | "));
      console.log("[HV] Diff bones:", diffNames);
      onTrackLog(trackLog);

      openPoseRef.current = openPose;
      closedPoseRef.current = closedPose;

      // Set bones to rest position immediately so there's no flash of closed hand
      for (const bone of allBones) {
        const op = openPose.get(bone.name);
        const cp = closedPose.get(bone.name);
        if (op && cp) {
          bone.quaternion.copy(op.quaternion).slerp(cp.quaternion, X_REST);
          bone.position.lerpVectors(op.position, cp.position, X_REST);
        }
      }

      debugInfo.current = {
        clipName: clip.name || "unnamed",
        tracks: clip.tracks.length,
        boundTracks: tracksApplied,
        duration: clip.duration,
        skinnedMeshes: skinnedCount,
        boneCount,
        diffBones: diffCount,
        method: ctrlTracksApplied > 0 ? "ctrl_to_deform" : "direct_keyframes",
      };
    } else {
      console.warn("[HV] No animation clips found!");
      onTrackLog(["No clips"]);
      debugInfo.current = {
        clipName: "NONE", tracks: 0, boundTracks: 0, duration: 0,
        skinnedMeshes: skinnedCount, boneCount, diffBones: 0, method: "none",
      };
    }
  }, [gltf, camera, onTrackLog]);

  /* ── Hero pose ── */
  useEffect(() => {
    const w = wrapperRef.current;
    if (!w) return;
    if (heroView) {
      w.rotation.set(
        THREE.MathUtils.degToRad(heroPose.rotX),
        THREE.MathUtils.degToRad(heroPose.rotY),
        THREE.MathUtils.degToRad(heroPose.rotZ),
      );
      w.position.set(heroPose.posX, heroPose.posY, heroPose.posZ);
      w.scale.setScalar(heroPose.scale);
    } else {
      w.rotation.set(0, 0, 0);
      w.position.set(0, 0, 0);
      w.scale.setScalar(1);
    }
  }, [heroView, heroPose]);

  /* ── Camera ── */
  useEffect(() => {
    const cam = camera as THREE.PerspectiveCamera;
    const d = cameraDist.current;
    if (heroView) {
      cam.position.set(0, d * 0.3, d * 0.95);
    } else {
      cam.position.set(0, 0, d);
    }
    cam.lookAt(0, 0, 0);
  }, [heroView, camera]);

  /* ── Per-frame ── */
  useFrame((state) => {
    const bones = bonesRef.current;
    if (bones.length === 0) return;
    frameCount.current++;

    // Version check — confirm browser has latest code
    if (frameCount.current === 1) {
      console.log("[HV] v12 | mode=" + mode + " exercise=" + (exercise?.name || "null"));
    }

    let x: number;
    if (mode === "DEMO") {
      if (exercise) {
        if (exerciseDemoStart.current === null) {
          exerciseDemoStart.current = state.clock.elapsedTime;
        }
        const t = state.clock.elapsedTime - exerciseDemoStart.current;
        x = demoProgressExercise(t, exercise);
      } else {
        exerciseDemoStart.current = null;
        x = demoProgressGeneric(state.clock.elapsedTime);
      }
    } else {
      exerciseDemoStart.current = null;
      const target = liveState ? liveTarget(liveState) : X_REST;
      liveProgress.current += (target - liveProgress.current) * 0.04;
      x = liveProgress.current;
    }
    x = Math.max(0, Math.min(1, x));

    // Slerp ALL bones between open and closed poses
    const openPose = openPoseRef.current;
    const closedPose = closedPoseRef.current;
    for (const bone of bones) {
      const op = openPose.get(bone.name);
      const cp = closedPose.get(bone.name);
      if (op && cp) {
        bone.quaternion.copy(op.quaternion).slerp(cp.quaternion, x);
        bone.position.lerpVectors(op.position, cp.position, x);
      }
    }

    // Debug
    if (frameCount.current % 15 === 0) {
      const d = debugInfo.current;
      const lines: DebugLine[] = [
        { label: "clip", value: d.clipName },
        { label: "tracks", value: `${d.boundTracks}/${d.tracks} bound` },
        { label: "bones", value: String(d.boneCount) },
        { label: "diffBones", value: String(d.diffBones), color: d.diffBones > 0 ? "text-success" : "text-danger" },
        { label: "method", value: d.method },
        { label: "mode", value: mode, color: mode === "DEMO" ? "text-warn" : "text-accent" },
        { label: "x", value: `${Math.round(x * 180)}° (${(x * 100).toFixed(0)}%)` },
      ];
      if (exercise) {
        lines.push({ label: "exercise", value: exercise.name, color: "text-accent" });
      }
      if (mode === "LIVE" && liveState) {
        lines.push({ label: "stateCmd", value: liveState.stateCmd });
        lines.push({ label: "target", value: `${liveState.targetAngle}°` });
      }
      onDebugLines(lines);
    }
  });

  return (
    <>
      <ambientLight intensity={0.4} />
      <directionalLight position={[5, 6, 4]} intensity={0.85} />
      <pointLight position={[-3, 1, -3]} intensity={0.15} color="#1D4ED8" />
      <FloorGrid />
      <group ref={wrapperRef}>
        <primitive object={gltf.scene} />
      </group>
      <OrbitControls
        enableZoom enablePan={false}
        target={[0, 0, 0]}
        enableRotate={!heroView}
        minPolarAngle={Math.PI / 6}
        maxPolarAngle={Math.PI * 0.85}
      />
    </>
  );
}

/* ================================================================
   Pose Tuner
   ================================================================ */

function PoseTuner({
  pose, onChange, onReset, onCopy,
}: {
  pose: HeroPose;
  onChange: (p: HeroPose) => void;
  onReset: () => void;
  onCopy: () => void;
}) {
  const slider = (label: string, key: keyof HeroPose, min: number, max: number, step: number) => (
    <div className="flex items-center gap-2">
      <label className="text-[10px] text-muted w-10 shrink-0">{label}</label>
      <input type="range" min={min} max={max} step={step} value={pose[key]}
        onChange={(e) => onChange({ ...pose, [key]: parseFloat(e.target.value) })}
        className="flex-1 h-1" />
      <span className="text-[10px] font-mono text-text w-12 text-right">{pose[key].toFixed(1)}</span>
    </div>
  );

  return (
    <div className="space-y-1.5">
      <div className="text-[11px] text-text font-semibold mb-1">Pose Tuner</div>
      {slider("rotX", "rotX", -180, 180, 1)}
      {slider("rotY", "rotY", -180, 180, 1)}
      {slider("rotZ", "rotZ", -180, 180, 1)}
      {slider("posX", "posX", -2, 2, 0.01)}
      {slider("posY", "posY", -2, 2, 0.01)}
      {slider("posZ", "posZ", -2, 2, 0.01)}
      {slider("scale", "scale", 0.2, 5, 0.1)}
      <div className="flex gap-1 pt-1 flex-wrap">
        <button onClick={onReset}
          className="px-2 py-0.5 rounded text-[10px] font-mono bg-white/[0.06] text-muted hover:text-text border border-border">
          Reset
        </button>
        <button onClick={onCopy}
          className="px-2 py-0.5 rounded text-[10px] font-mono bg-accent/20 text-text border border-border">
          Copy Hero Preset
        </button>
      </div>
    </div>
  );
}

/* ================================================================
   Public HandViewer
   ================================================================ */

interface HandViewerProps {
  mode?: DriverMode;
  liveState?: LiveState | null;
  className?: string;
  targetAngle?: number;
  exercise?: ExerciseInfo | null;
}

export function HandViewer({
  mode = "DEMO",
  liveState = null,
  className = "",
  targetAngle,
  exercise = null,
}: HandViewerProps) {
  const [debugLines, setDebugLines] = useState<DebugLine[]>([]);
  const [trackLog, setTrackLog] = useState<string[]>([]);
  const [showDebug, setShowDebug] = useState(false);
  const [heroView, setHeroView] = useState(true);
  const [heroPose, setHeroPose] = useState<HeroPose>({ ...DEFAULT_HERO });
  const [forceMode, setForceMode] = useState<DriverMode | null>(null);

  const effectiveMode = forceMode ?? mode;

  const effectiveLiveState: LiveState | null = useMemo(() => {
    if (targetAngle !== undefined) {
      return {
        stateCmd: "DEMO_TARGET",
        motionLocked: false,
        lockRemainingMs: 0,
        targetAngle,
      };
    }
    return liveState;
  }, [targetAngle, liveState]);

  const handleCopyPose = useCallback(() => {
    const code = `const DEFAULT_HERO: HeroPose = ${JSON.stringify(heroPose, null, 2)};`;
    console.log("[HV] COPY HERO PRESET:\n" + code);
    try { navigator.clipboard.writeText(code); } catch {}
  }, [heroPose]);

  const handleTrackLog = useCallback((log: string[]) => setTrackLog(log), []);

  return (
    <div className={`relative bg-bg overflow-hidden ${className}`}>
      {/* Top-right badges */}
      <div className="absolute top-2 right-2 z-20 flex gap-1">
        <button onClick={() => setHeroView(v => !v)}
          className={`px-2 py-0.5 rounded text-[10px] font-mono border border-border ${
            heroView ? "bg-accent/20 text-text" : "bg-white/[0.06] text-muted"}`}>
          {heroView ? "Hero" : "Free"}
        </button>
        <span className={`px-2 py-0.5 rounded text-[10px] font-mono font-medium ${
          effectiveMode === "DEMO" ? "bg-warn/15 text-warn" : "bg-success/15 text-success"}`}>
          {effectiveMode}
        </span>
      </div>

      {/* Top-left controls */}
      <div className="absolute top-2 left-2 z-20 flex gap-1 flex-wrap">
        <button onClick={() => setShowDebug(v => !v)}
          className="w-7 h-7 rounded bg-white/[0.08] text-muted hover:text-text text-[11px] font-mono flex items-center justify-center border border-border">
          ?
        </button>
        <button onClick={() => setForceMode(forceMode === "DEMO" ? null : "DEMO")}
          className={`px-2 py-0.5 rounded text-[10px] font-mono border border-border ${
            forceMode === "DEMO" ? "bg-warn/20 text-warn" : "bg-white/[0.06] text-muted hover:text-text"}`}>
          Force Demo
        </button>
      </div>

      {/* Debug overlay */}
      {showDebug && (
        <div className="absolute top-12 left-2 z-20 bg-bg/95 border border-border rounded-lg p-2.5 text-[10px] font-mono leading-[1.6] text-muted min-w-[290px] max-h-[85%] overflow-auto">
          <div className="text-text font-semibold mb-1 text-[11px]">Hand Debug</div>
          {debugLines.map((line, i) => (
            <div key={i}>
              <span className="text-muted">{line.label}: </span>
              <span className={line.color || "text-text"}>{line.value}</span>
            </div>
          ))}

          {trackLog.length > 0 && (
            <div className="mt-2 pt-2 border-t border-border">
              <div className="text-text font-semibold mb-1 text-[11px]">Track Binding</div>
              {trackLog.map((line, i) => (
                <div key={i} className="text-[9px] leading-[1.4] text-muted">{line}</div>
              ))}
            </div>
          )}

          {heroView && (
            <div className="mt-3 pt-2 border-t border-border">
              <PoseTuner
                pose={heroPose}
                onChange={setHeroPose}
                onReset={() => setHeroPose({ ...DEFAULT_HERO })}
                onCopy={handleCopyPose}
              />
            </div>
          )}
        </div>
      )}

      <Canvas
        frameloop="always"
        camera={{ fov: 40, near: 0.001, far: 1000 }}
        style={{ background: "transparent" }}
      >
        <HandScene
          mode={effectiveMode}
          liveState={effectiveLiveState}
          exercise={exercise}
          heroPose={heroPose}
          heroView={heroView}
          onDebugLines={setDebugLines}
          onTrackLog={handleTrackLog}
        />
      </Canvas>
    </div>
  );
}
