# Tool and observation spec (schema_version 4)

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

## Plan-ahead: decisions on a receding horizon

Under plan-ahead control the pilot flies by making **one decision per turn**,
each from a fresh, self-contained observation (`tools.DecisionToolLayer`,
`runner._decision_loop`). A turn is one request with no conversation history
beyond what its prompt holds: the mission, the current state, the events since
the last decision, the rest of the plan, the time spent in each region, the
map as the pilot knows it, the pilot's own notes and its last few decisions.
Every turn is one complete training example.

| Decision | Arguments | Effect |
|---|---|---|
| `takeoff` | `altitude` | climb; the first plan follows |
| `fly_plan` | `segments`, optional `mode` (`replace` / `append`) | a new plan from where the drone is (momentum kept), or an extension; any length |
| `keep` | none | fly on unchanged |
| `hold` | none | brake along the course and hover |
| `land` | none | straight to the launch point at the current altitude, then land |
| `check_path` | `east`, `north` | free: answers at once and does not use up the turn |

Every decision may carry `notes` (the pilot's memory, shown back every turn,
replacing the previous notes) and `reason` (one sentence, logged for training).

After a decision the drone flies on to the **next decision point**: an event
(segment completed, region entered or left, region discovered, "plan ends in
N s", landed) or `wake_s` at most. Settings, in `config/limits.yaml` under
`decisions:` (a mission can override them):

| Setting | Default | Meaning |
|---|---|---|
| `horizon_s` | 15 | dead-man's switch: each decision (keep included) authorises this much flight; with no decision within it the drone brakes and holds (a plan ending with a landing is finished) |
| `wake_s` | 3 | the pilot decides again at the next event, or within this |
| `plan_ending_s` | 7 | the "plan ends in N s" event: at least `wake_s` + `latency_s` + a margin |
| `latency_s` | 1 | lockstep only: a decision takes effect this long after its observation |
| `history` | 3 | the pilot sees its last N decisions |

**Clock.** `--clock lockstep` (default on the fake backend, for data): the
simulation stands still while the model thinks, and each decision takes effect
`latency_s` later, so the data does not depend on how long the teacher took.
`--clock realtime` (demos, AAS): the drone keeps flying while the model thinks.

**Sensor.** A mission can hide regions until the drone comes within range
(`sensor: {range_m, hidden: [...]}`, `sensing.py`): they then appear on the
pilot's map and the next observation reports `discovered <region>`. Scoring
always uses the full map.

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
happened since the pilot's last decision), and, for plots only (never sent to
the model), `plan_path` and `plan_age_s` for a plan that started since the
last observation.

Two clocks are recorded on every observation: `t_sim` (PX4 timestamp) and
`t_wall` (host).

## Changing this spec

Bump `SCHEMA_VERSION`, re-record, and never mix versions in one dataset.
