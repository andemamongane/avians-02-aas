# Tool and observation spec (schema_version 2)

The authority is `src/avians/schema.py`; this document explains it.

## Frame and units

The aerial-autonomy-stack's frame: metres **east** and **north** of the launch
point, and **altitude** above it (always positive). AAS's `set_reposition` and
`orbit_action` take exactly these, so tool arguments pass through unchanged.

PX4's local frame is NED (x north, y east, z down) from the EKF origin, which is
not at ground level. `backends/aas.py` measures from where the drone rests when
the episode starts (where AAS saves home at takeoff) and records that origin in
`meta.json` as `frame_origin_ned`, so the rosbag trace is converted identically.

## Tools

| Tool | Arguments | AAS | Returns when |
|---|---|---|---|
| `get_state` | none | none | immediately |
| `takeoff` | `altitude` | `takeoff_action` | within 1 m of the altitude and slow |
| `reposition` | `east`, `north`, `altitude`, optional `speed` | `set_speed`, `set_reposition` | within 1 m and slow, or timeout |
| `goto_region` | `region`, `altitude`, optional `speed` | `set_reposition` | same as reposition |
| `orbit` | `east`, `north`, `altitude`, `radius`, `duration` | `orbit_action`, then hold | duration elapsed |
| `hover` | `duration` | `set_reposition` (current position) | duration elapsed |
| `check_path` | `east`, `north` | none | immediately |
| `land` | none | `land_action` | disarmed, or timeout |
| `abort` | optional `reason` | `set_reposition` (current position) | immediately |

AAS accepts reposition, orbit and land only while hovering or orbiting, and
refuses any command while another is active. A refusal comes back as `failed`
with the reason.

## Status codes

`arrived` · `ok` · `timeout` · `rejected` · `blocked` · `aborted` · `failed`

- `rejected`: bad arguments, or outside a global limit (geofence, altitude, speed, durations)
- `blocked`: the vehicle state forbids it (e.g. reposition before takeoff)
- `failed`: the autopilot interface refused it, or transport error

Mission regions never cause a rejection.

## Observation

Position (east, north, alt), velocity, heading, mode, armed, battery (unknown
under AAS), target and distance, elapsed action and mission time, an interval
summary (path length, peak speed), the regions the drone is inside, and the
signed distance to every region's boundary (+ inside). `check_path` adds an
`info` line. `detections` is always empty in this milestone.

Two clocks are recorded on every observation: `t_sim` (PX4 timestamp) and
`t_wall` (host).

## Changing this spec

Bump `SCHEMA_VERSION`, re-record, and never mix versions in one dataset.
