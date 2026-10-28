# Running simulations

Step by step, from a fresh machine to a recorded flight. Two setups:

- **Linux GPU machine:** the full aerial-autonomy-stack (AAS) with Gazebo. Wind,
  worlds, the Gazebo video (`gazebo.mp4`). Sections 1-6.
- **Laptop:** PX4 SIH inside the avians container. No Gazebo, no video; good
  for trying missions and pilots. Section 7.

Which shell you are in matters. Commands marked **host** run in a normal
terminal on the machine (prompt like `andema@460N-EE12392:~$`). Commands
marked **container** run inside the avians container (prompt like
`root@5ddbf1c24f5c:/workspace#`), which you enter with `./scripts/shell.sh`.
The `avians` command only exists inside the container.

## 1. Once per machine: check the prerequisites (host)

```bash
docker ps >/dev/null && echo "docker ok"     # you must be in the docker group
nvidia-smi -L                                # the GPU is visible
command -v tmux                              # needed for step 2 and section 3
```

If `docker ps` fails, ask an admin to add you to the `docker` group: nothing
else works without it. No sudo is needed for anything below.

## 2. Once per machine: get the code and build (host)

```bash
git clone --recursive git@github.com:andemamongane/avians-02-aas.git
cd avians-02-aas
cp .env.example .env              # then put your API keys in .env (ANTHROPIC_API_KEY, ...)

cd external/aerial-autonomy-stack/tools_and_docs
./sim_build.sh                    # builds the AAS images; long, once
```

AAS's `sim_run.sh` opens every container in an `xterm` window. Without a
display (or without xterm installed), give it a stand-in that runs each
container in a background tmux session instead:

```bash
mkdir -p ~/bin
cat > ~/bin/xterm <<'EOF'
#!/usr/bin/env bash
# Stand-in for xterm (no display, no sudo): runs `-e <command>` in a detached
# tmux session named after -title, so `docker run -it` still gets a terminal.
title=xterm cmd=()
while [ $# -gt 0 ]; do
  case "$1" in
    -title) title="$2"; shift 2 ;;
    -e) shift; cmd=("$@"); break ;;
    *) shift ;;
  esac
done
name="aas-$(echo "$title" | tr -c 'A-Za-z0-9_\n' '-')"
exec tmux new-session -d -s "$name" -x 200 -y 50 "$(printf '%q ' "${cmd[@]}"); exec bash"
EOF
chmod +x ~/bin/xterm
```

It also runs `xhost` to give containers the X display, and stops if that
fails (`xhost: unable to open display ""`). Headless, there is nothing to
grant, so give it a stand-in that does nothing:

```bash
cat > ~/bin/xhost <<'EOF'
#!/usr/bin/env bash
# Stand-in for xhost (no display): nothing to grant.
exit 0
EOF
chmod +x ~/bin/xhost
```

To update an existing clone later: `git pull && git submodule update --init --recursive`.
Code changes need no rebuild: the container mounts `src/`, `config/`,
`missions/` and `maps/` live.

## 3. Each session: start AAS (host, terminal 1)

```bash
cd ~/avians/avians-02-aas/external/aerial-autonomy-stack/tools_and_docs
tmux new -s aas-launch "PATH=~/bin:\$PATH XDG_CURRENT_DESKTOP=GNOME HEADLESS=true GND_CONTAINER=false NUM_QUADS=1 WORLD=impalpable_greyness ./sim_run.sh; echo; read -p 'sim_run.sh exited; press Enter to close'"
```

The first time, or whenever it fails, run the same command **without** tmux
first, so any error stays on screen:

```bash
PATH=~/bin:$PATH XDG_CURRENT_DESKTOP=GNOME HEADLESS=true GND_CONTAINER=false \
  NUM_QUADS=1 WORLD=impalpable_greyness ./sim_run.sh
```

- Wait for `Fly, my pretties, fly!`, then **detach** with `Ctrl-b` then `d`.
  If you see `sim_run.sh exited` instead, AAS did not start: read the error
  above it (or run it without tmux, as above).
  AAS keeps running. Do not press another key in that window: any key stops AAS.
- Running it inside tmux means a dropped SSH connection does not stop AAS.
- `HEADLESS=true` runs Gazebo without its window (the video is still
  recorded); `GND_CONTAINER=false` skips QGroundControl, which needs a display.
  `XDG_CURRENT_DESKTOP=GNOME` gets past AAS's desktop check.
- `WORLD` must match the mission's `environment: world`, if it names one.
- The containers run in tmux sessions too: `tmux ls`, and e.g.
  `tmux attach -t aas-Simulation` to see Gazebo's output (detach: `Ctrl-b d`).

## 4. Each session: start avians and enter it (host, terminal 2)

```bash
cd ~/avians/avians-02-aas
./scripts/up_remote.sh        # starts the avians container next to AAS; the first run builds it (15-30 min)
./scripts/shell.sh            # puts you INSIDE the container: prompt becomes root@...:/workspace#
```

If `./scripts/shell.sh` complains, enter with the explicit form:

```bash
docker compose -f docker/compose.yml -f docker/compose.remote.yml exec avians bash
```

## 5. Fly (container)

```bash
avians doctor                                                       # must list /Drone1 topics and actions
avians check missions/survey_then_land.yaml --control step          # what the pilot will see; nothing flies
avians fly missions/survey_then_land.yaml --control step --dry-run  # scripted flight, no LLM, no API key
avians fly missions/survey_then_land.yaml --control step            # an LLM (Claude by default) flies it
avians fly missions/survey_then_land.yaml --control step --pilot codex   # another pilot
```

- On AAS, use `--control step` for now: plan-ahead flight (the default) needs
  the avians trajectory controller in AAS, which is not written yet.
- `--dry-run` replaces the LLM with the mission's scripted plan: everything
  else (tools, safety, AAS, PX4, Gazebo, scoring, video) is real. With
  `--control step` it exists for `hover`, `survey_then_land` and `two_fields`.
- A real pilot needs its API key in `.env`. After editing `.env`, rerun
  `./scripts/up_remote.sh` (host) so the container sees it.

Each flight writes `data/episodes/ep_NNNN/`, visible on the host too:

| File | What |
|---|---|
| `gazebo.mp4` | the chase-camera video from Gazebo |
| `stl.json`, `plots/summary.md` | the score, and a readable summary |
| `trajectory.jsonl` | the whole conversation and every tool call |
| `meta.json` | pilot, envelope, environment; `"sim"` says what the simulator applied and any problems |
| `px4.ulg`, `gazebo_poses.jsonl`, `state.jsonl` | PX4's log, every Gazebo model's pose, the sampled vehicle state |

Review (container):

```bash
avians score 12       # re-score episode 12
avians plot 12        # regenerate its figures
avians video 12       # the map-and-timeline replay video (separate from gazebo.mp4)
```

## 6. Stop

```bash
exit                                        # leave the container (back on the host)
docker compose -f docker/compose.yml -f docker/compose.remote.yml down   # stop avians (host)
tmux attach -t aas-launch                   # then press any key: AAS stops and cleans up
```

## 7. Laptop (PX4 SIH, no Gazebo)

```bash
./scripts/up.sh        # host: builds the first time, starts PX4 SIH + AAS's interface
./scripts/shell.sh     # host: enter the container
avians doctor          # container
avians fly missions/survey_then_land.yaml --dry-run                    # container: plan-ahead (default)
avians fly missions/survey_then_land.yaml --control step --dry-run     # container: step-by-step
avians fly missions/slalom.yaml --backend fake --dry-run               # container: no simulator at all
avians compare-controls --fly --backend fake                           # container: plan-ahead vs step-by-step, filmed
```

`compare-controls` flies each mission both ways with the default pilot (an API
key is needed; `--dry-run` uses the scripted plans instead) and writes the
videos and a results table to `data/showcase/controls/`. It needs no simulator,
so it also runs on the Linux machine inside the avians container.

Open the QGroundControl desktop app on the laptop to watch flights live: the
container streams PX4's telemetry to it. Stop with `exit`, then
`docker compose -f docker/compose.yml down` (host).

## When something goes wrong

| Symptom | Likely cause |
|---|---|
| `avians: command not found` | you are on the host: run `./scripts/shell.sh` first |
| `Untested (non-GNOME, non-WSL) desktop environment, exiting.` | add `XDG_CURRENT_DESKTOP=GNOME` (section 3) |
| `sim_run.sh` stops at `xhost: unable to open display ""` | no display: create the `~/bin/xhost` stand-in (section 2) and keep `PATH=~/bin:$PATH` |
| `tmux new -s aas-launch ...` prints `[exited]` at once | `sim_run.sh` failed: run it without tmux to see why (section 3); often the AAS images are not built (`./sim_build.sh`) |
| `avians doctor` shows no `/Drone1` topics | AAS is not running (`docker ps` on the host shows no `simulation-container-inst0`), or ROS discovery across containers fails |
| a flight leaves only `map.yaml` and an empty `trajectory.jsonl` | it could not connect to the drone (AAS not running) or the simulator was not ready: the reason is in the terminal output |
| `up_remote.sh`: "AAS is not running" | start AAS first (section 3) |
| `the mission needs world ... restart AAS with WORLD=...` | the running Gazebo world differs from the mission's |
| `gazebo.mp4` missing or empty | see `meta.json` → `"sim"` → `"notes"`, and `tmux attach -t aas-Simulation` for Gazebo errors; without a display the camera may not render |
| `plan-ahead control is not available on AAS yet` | add `--control step` |
| `already armed` at the start of a flight | the previous flight did not land: restart AAS |
