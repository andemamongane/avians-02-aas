# Tool and observation spec (schema_version 3)

The authority is `src/avians/schema.py`; this document explains it.

## Frame and units

The aerial-autonomy-stack's frame: metres **east** and **north** of the launch
point, and **altitude** above it (always positive). AAS's `set_reposition` and
`orbit_action` take exactly these, so tool arguments pass through unchanged.

PX4's local frame is NED (x north, y east, z down) from the EKF origin, which is
not at ground level. `backends/aas.py` measures from where the drone rests when
the episode starts (where AAS saves home at takeoff) and records that origin in
`meta.json` as `frame_origin_ned`, so the rosbag trace is converted identically.

## Two control modes

A mission flies under one of two tool sets (`control:` in the mission, or
`avians fly --control`):

- **plan** (default): plan-ahead control. The pilot commits to whole flight
  plans that the drone flies continuously while the pilot keeps thinking.
  Every call returns at once. `src/avians/trajectory.py`, `executor.py`.
- **step**: the step-by-step baseline. Every flight tool blocks until the drone
  has arrived, so the drone hovers whenever the model thinks.

## Plan tools

| Tool | Arguments | Returns |
|---|---|---|
| `get_state` | none | immediately |
| `takeoff` | `altitude` | immediately; a plan sent during the climb starts when it completes |
| `fly_plan` | `segments`, optional `mode` (`replace` / `append`) | immediately, with a preview |
| `wait` | `seconds`, optional `until` (`event` / `plan_complete` / `landed`) | at that, or after `seconds` |
| `hold` | none | immediately; the drone brakes along its course and hovers |
| `land` | none | immediately; straight to the launch point at the current altitude, then lands |
| `check_path` | `east`, `north` | immediately |

Segments, in order:

| `type` | Fields | Flown as |
|---|---|---|
| `waypoint` | `east`, `north`, `altitude`, `speed?` | flown **through**; the corner is rounded (at most `corner_radius_m`) |
| `region` | `region`, `altitude`, `speed?` | a waypoint well inside the region |
| `orbit` | `east`, `north`, `altitude`, `radius`, `turns?`, `direction?` | circle(s) from the nearest point |
| `survey` | `region`, `altitude`, `spacing?`, `speed?` | lawnmower rows along the region's long side |
| `hold` | `duration` | stop at the previous point and hover |
| `land` | none (must be last) | straight to above the launch point, then land |

The planner (`trajectory.py`) turns segments into one timed path inside the
flight envelope (`config/limits.yaml`): speed, acceleration shared between
turning (70%) and braking (60%), climb and descent rates. It slows only where a
turn or a stop demands it. A replacing plan starts from the drone's reference
position *and velocity*: the path carries on along the current course for as
long as braking takes, then turns, so the drone never stops to change its mind.

The safety arbiter checks every point of a plan (geofence, ceiling, floor,
hold and plan durations). A rejected plan leaves the current one flying. The
preview tells the pilot the plan's length, duration, top speed, when each
segment completes and which map regions the path passes through: geometry
only, never the STL score.

On AAS, plans fly through `offboard_action` with an avians trajectory
controller in AAS's `offboard_control` package. That controller is still to
be written (Linux machine); until then `--control plan` runs on the fake
backend only.

## Step tools

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

Plan control adds `plan_status` (what the plan is doing), `events` (what
happened since the pilot's last call: segments completed, regions entered or
left, the plan completing, landing), and, for plots only (never sent to the
model), `plan_path` and `plan_age_s` for a plan that started since the last
observation.

Two clocks are recorded on every observation: `t_sim` (PX4 timestamp) and
`t_wall` (host).

## Changing this spec

Bump `SCHEMA_VERSION`, re-record, and never mix versions in one dataset.
