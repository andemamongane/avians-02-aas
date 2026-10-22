# avians-02-aas

**Milestone 2 of the avians project.** Claude pilots a drone through the
[aerial-autonomy-stack](https://github.com/JacopoPan/aerial-autonomy-stack) (AAS)
interface, and every flight is scored against a mission specification written
in signal temporal logic (STL) over a map of polygon regions.

| In scope | Out of scope (later) |
|---|---|
| AAS's `autopilot_interface` (takeoff, reposition, orbit, land) on PX4 v1.17.0 | Perception (YOLO, cameras, LiDAR) |
| Maps of named polygon regions; tools that use them | Multi-drone, VTOLs, ArduPilot |
| STL specs, robustness scoring per clause | Gazebo worlds/wind on the laptop (remote machine only) |
| One quadrotor, PX4 SIH locally | Fine-tuning, RL |

## Two ways to run

| | Local (this repo, any laptop) | Remote (full AAS, Linux + NVIDIA GPU) |
|---|---|---|
| PX4 | v1.17.0 + AAS's patch and DDS topic list | same |
| Autopilot interface | AAS `px4_interface`, built from the pinned submodule | same |
| Physics | PX4's built-in SIH (no world, no wind) | Gazebo Harmonic, wind, worlds |
| QGroundControl | none | AAS ground container |
| avians code | identical | identical |

The local container runs the **real** AAS interface node against the same PX4
version, so the tools behave exactly as they will on the remote machine. Only
the physics differs. The remote setup is not wired up yet (see the end).

## Quick start (local)

Clone with the submodule:

```bash
git clone --recursive <repo-url>
cd avians-02-aas
```

Then either open it in VS Code and run **Dev Containers: Reopen in Container**,
or from a terminal:

```bash
./scripts/up.sh                                   # build + start + health check
```

The first build compiles PX4 and takes 20-30 minutes; later starts take seconds.
A `.env` file is created on first open: paste your `ANTHROPIC_API_KEY` into it,
then rebuild the container (or rerun `./scripts/up.sh`).

Inside the container (the VS Code terminal, or `./scripts/shell.sh`):

```bash
avians doctor                                     # PX4 topics + AAS actions up?
avians check missions/survey_then_land.yaml       # print exactly what the agent will see
avians fly missions/survey_then_land.yaml --dry-run   # scripted plan, no API calls
avians fly missions/survey_then_land.yaml         # Claude flies it
avians fly missions/survey_then_land.yaml --pilot deepseek   # or codex, claude-opus, ...
avians score 4                                    # re-score episode 4 against its spec
avians plot --all                                 # figures for every episode
avians video 4                                    # episode 4 as an MP4
avians showcase --fly                             # every mission x pilot, filmed
pytest                                            # 83 offline tests
```

Keep the laptop awake during a flight. If the host sleeps, the simulator
freezes mid-air and every running action times out.

## Pilots

Any AI agent in `config/pilots.yaml` can fly, through the same tools, prompt
and safety layer, so their flights are directly comparable:

| Pilot | Provider | Model | Key in `.env` |
|---|---|---|---|
| `claude` (default), `claude-opus`, `claude-haiku` | Anthropic Messages API | Claude Sonnet / Opus / Haiku 5.5 | `ANTHROPIC_API_KEY` |
| `deepseek`, `deepseek-flash` | DeepSeek, OpenAI-compatible API | `deepseek-v4-pro`, `deepseek-flash` | `DEEPSEEK_API_KEY` |
| `codex`, `codex-luna` | OpenAI Responses API | `gpt-6.1-sol`, `gpt-6-luna` (Codex's models) | `OPENAI_API_KEY` |

`codex` is the model behind OpenAI Codex, called directly with our tools, not
the Codex CLI agent. Adding a pilot is a few lines of YAML for any provider
that speaks one of these three APIs. Every episode logs the conversation in
one format whatever the provider (`src/avians/pilots.py`), and `meta.json`
records the pilot, provider and model. After adding a key, recreate the
container so it sees it.

## Videos

`avians video <episode>` replays a flight as a 1920x1080 MP4 at 4x speed: the
map with regions and the growing path, altitude, every tool call with its
outcome (and when the model was thinking), and the STL verdict per clause at
the end. Everything comes from what the episode recorded.

`avians showcase` films the latest flight of every mission x pilot pair
(`showcase:` in `config/pilots.yaml`) into `data/showcase/<mission>/<pilot>.mp4`,
plus `compare.mp4` with all pilots side by side on one clock, and an index
`data/showcase/README.md`. Add `--fly` to fly the missing pairs first.

## Missions: map + spec + prompt

A mission is a YAML file that points at a map, states an STL spec, and gives
the agent a plain-language prompt (`missions/prompts/`).

```yaml
name: survey_then_land
prompt_file: survey_then_land.txt
map: maps/field_01.yaml
spec: >-
  G[0,240] !in(nofly_1)
  & F[0,120] G[0,10] in(survey_a)
  & F[0,240] (in(home_pad) & alt <= 1)
show_spec: true          # false = the agent gets only the prompt
early_stop: violated     # off | violated (default) | decided; see "Online scoring"
dry_run_script: [...]    # flown by --dry-run instead of the model
```

A map is a list of polygons in **metres east and north of the launch point**:

```yaml
regions:
  - id: nofly_1
    name: Power lines
    kind: no_fly            # goal | no_fly | corridor | landing | other
    description: Overhead lines between the pad and field A.
    polygon: [[4, 8], [18, 8], [18, 13], [4, 13]]   # [east, north]
    altitude: [0, 20]       # optional floor/ceiling
```

Regions are **information, never enforced**. The safety layer only enforces
the global geofence, altitude band and speed limit (`config/limits.yaml`).
Whether the agent stays out of a no-fly zone is what the STL score measures.

### STL syntax

| Syntax | Meaning | Robustness |
|---|---|---|
| `in(r)` | inside region r (and its altitude band) | signed distance to the boundary, metres (+ inside) |
| `alt <= 5`, `speed < 2` | also `>`, `>=`; signals `alt speed east north` | margin, e.g. `5 - alt` |
| `!p`, `p & q`, `p \| q`, `p -> q` | not, and, or, implies | negate, min, max |
| `G[a,b] p` | p at every moment in [a, b] s | min over the window |
| `F[a,b] p` | p at some moment in [a, b] s | max over the window |
| `p U[a,b] q` | q at some moment in [a, b] s, and p before then | standard bounded until |

Times are seconds from mission start. Robustness > 0 means satisfied with that
margin. Flights shorter than the spec's horizon are padded by holding the last
state (the drone stays where it landed); the report says how much.

## Tools the agent gets

They mirror AAS's interface, in AAS's frame (east, north, altitude above launch).

| Tool | AAS call | Returns when |
|---|---|---|
| `takeoff(altitude)` | `takeoff_action` | altitude reached |
| `reposition(east, north, altitude, speed?)` | `set_speed` + `set_reposition` | arrived (avians checks; AAS returns at once) |
| `goto_region(region, altitude, speed?)` | `set_reposition` to a point well inside the region | arrived |
| `orbit(east, north, altitude, radius, duration)` | `orbit_action`, then a hold | duration elapsed |
| `hover(duration)` | `set_reposition` to the current position | duration elapsed |
| `land()` | `land_action`: flies home at the current altitude, lands there | disarmed |
| `abort(reason?)` | `set_reposition` to the current position | immediately |
| `check_path(east, north)` | none (map only) | immediately |
| `get_state()` | none | immediately |

Every result reports position, mode, which regions the drone is in and its
signed distance to each, and the mission time. See `docs/tool_api.md`.

## What each episode contains

```
data/episodes/ep_0004/
├── trajectory.jsonl   # system prompt (with map + spec), every request/response, actions, observations
├── bag/               # rosbag2: PX4 position, status, commands (the trace STL is scored on)
├── map.yaml           # copy of the mission map
├── stl.json           # robustness overall and per clause
├── stl_online.jsonl   # robustness interval per second during the flight (online monitor)
├── meta.json          # spec, model, PX4 / px4_msgs / AAS commit, frame origin, usage
├── adp.json           # ADP-style action/observation trajectory
└── plots/             # ground track with regions, altitude, timeline, summary.md
```

## Online scoring and early stop

While the drone flies, a monitor scores the flight so far as a *robust
satisfaction interval* `[lo, hi]`: the range the final robustness can still
end up in, whatever the pilot does next (`src/avians/monitor.py`,
`stl.score_online`). The future is bounded by physics: the drone cannot move
faster than `max_speed_ms x monitor_speed_margin` (`config/limits.yaml`), so
"reach field A within 3 s" from 40 m away is lost at once, not after 3 s.

| `early_stop` | The mission ends when | `terminated_by` |
|---|---|---|
| `off` | never (intervals are still logged) | |
| `violated` (default) | `hi <= 0`: the spec can no longer be satisfied | `stl_violated` |
| `decided` | also `lo > 0`: satisfied whatever happens next | `stl_satisfied` |

When it ends a mission, the action in progress returns `aborted` with the
reason, the model gets no further turn, and avians holds and lands the drone.
That landing is logged as notes, never as tool calls. The pilot never sees
the live verdict. `meta.json` records the policy and the stop under
`early_stop`. The monitor scores the agent-side state stream; the final
`stl.json` is still computed offline from the rosbag.

`plots/07_stl_online.png` shows the interval over time for the whole spec and
each clause, when each clause was decided, the stop, and the final offline
score (always inside the band). `summary.md` lists when each clause was known;
`avians video` shows the live verdict per clause until the end of the flight.

## Known behaviour (read before debugging)

| Symptom | Cause |
|---|---|
| `land` flies somewhere first | AAS's land always returns to the launch point, then descends |
| Battery always "unknown" | AAS's DDS topic list does not bridge `battery_status` |
| Landed drone reads alt -0.7..0.7 m | SIH altitude estimate drifts; specs use `alt <= 1` for "landed" |
| Everything times out at once | the host slept; restart the container |
| `already armed` at the start of a flight | a previous flight didn't land; restart the container |
| PX4 logs `command 192 unsupported` | harmless: AAS's reposition asks for no mode switch; PX4's navigator still flies it |

avians never cancels an AAS action: cancelling makes AAS climb to its abort
altitude (100 m by default). `hover` and `abort` hold position with a
reposition instead.

## Remote AAS (not done yet)

The AAS side is ready to run unmodified from `external/aerial-autonomy-stack`
(pinned commit in `meta.json` of every episode). Still to do on the Linux
machine: build and run AAS with one quad, and run the avians container on the
same ROS domain (1) and network as AAS's aircraft container, so `/Drone1/*`
is visible. Then show QGroundControl and apply wind per mission.
