#!/usr/bin/env bash
# =============================================================================
#  Create and drive the training VM from this machine.
#
#    ./gcp/gcp_run.sh create     make the VM and start the run
#    ./gcp/gcp_run.sh watch      poll until it finishes, then report (exit = event)
#    ./gcp/gcp_run.sh status     one-shot status
#    ./gcp/gcp_run.sh log        tail the run log over SSH
#    ./gcp/gcp_run.sh fetch      download the finished model
#    ./gcp/gcp_run.sh stop       stop the VM now (keeps the disk)
#    ./gcp/gcp_run.sh destroy    delete the VM and its disk
#
#  The VM shuts itself down when the run reaches a terminal state, so the normal
#  path costs nothing after it finishes. `watch` additionally enforces a hard
#  budget ceiling in case the in-VM logic never gets to run.
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="$(cd "$HERE/.." && pwd)"
GCLOUD="${GCLOUD_BIN:-$HOME/google-cloud-sdk/bin/gcloud}"

VM="${WAKEWORD_VM:-wakeword-trainer}"
ZONE="${WAKEWORD_ZONE:-us-central1-a}"
# GPU capacity is per-zone and varies through the day. Rather than fail on the
# first "does not have enough resources", walk a candidate list; the zone that
# succeeds is recorded so every later command targets the right one.
ZONE_CANDIDATES="${WAKEWORD_ZONES:-us-central1-a us-central1-b us-central1-c us-central1-f us-west1-b us-west4-a us-east1-c us-east1-d us-east4-b us-south1-a}"
ZONE_FILE="${WAKEWORD_ZONE_FILE:-$PIPELINE/.gcp_zone}"
[ -f "$ZONE_FILE" ] && ZONE="$(cat "$ZONE_FILE")"
MACHINE="${WAKEWORD_MACHINE:-n1-standard-16}"
GPU="${WAKEWORD_GPU:-nvidia-tesla-t4}"
DISK="${WAKEWORD_DISK:-200GB}"
CONFIG="${WAKEWORD_CONFIG:-config/fbi_guy.yaml}"
BRANCH="${WAKEWORD_BRANCH:-claude/colab-cli-runner}"
REPO_URL="${WAKEWORD_REPO_URL:-https://github.com/trout1758-cpu/custom-micro-wake-word-model}"
BUCKET="${WAKEWORD_BUCKET:-}"
MAX_HOURS="${WAKEWORD_MAX_HOURS:-14}"   # hard ceiling; see cost guard in cmd_watch

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m    %s\033[0m\n' "$*"; }
die()  { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

[ -x "$GCLOUD" ] || die "gcloud not found at $GCLOUD"
g() { "$GCLOUD" "$@"; }

vm_exists()  { g compute instances describe "$VM" --zone "$ZONE" >/dev/null 2>&1; }
vm_status()  { g compute instances describe "$VM" --zone "$ZONE" --format='value(status)' 2>/dev/null; }
ssh_vm()     { g compute ssh "$VM" --zone "$ZONE" --tunnel-through-iap --command "$1" 2>/dev/null; }

cmd_create() {
  vm_exists && die "$VM already exists. Use 'status', or 'destroy' to start over."
  local project; project="$(g config get-value project 2>/dev/null)"
  [ -n "$project" ] && [ "$project" != "(unset)" ] || die "no project set: gcloud config set project PROJECT_ID"

  warn "It shuts itself down when the run ends — success or terminal failure."

  local created=""
  for z in $ZONE_CANDIDATES; do
    say "Trying $MACHINE + $GPU in $z (project $project)"
    if _create_in_zone "$z"; then
      created="$z"; ZONE="$z"; echo "$z" > "$ZONE_FILE"
      say "Created in $z"
      break
    fi
    warn "$z unavailable; trying the next zone"
  done
  [ -n "$created" ] || die "no zone had capacity for $MACHINE + $GPU.
  Try again shortly, or set WAKEWORD_MACHINE=n1-standard-8 (smaller shapes
  place more easily) and re-run."

  say "Boot + driver install + bootstrap takes ~10-15 min."
  echo "  watch:  $0 watch"
  echo "  log:    $0 log"
}

_create_in_zone() {
  local z="$1"
  g compute instances create "$VM" \
    --zone="$z" \
    --machine-type="$MACHINE" \
    --accelerator="type=$GPU,count=1" \
    --maintenance-policy=TERMINATE \
    --image-family="${WAKEWORD_IMAGE_FAMILY:-common-cu129-ubuntu-2204-nvidia-580}" \
    --image-project=deeplearning-platform-release \
    --boot-disk-size="$DISK" \
    --boot-disk-type=pd-balanced \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --metadata="install-nvidia-driver=True,\
wakeword-repo-url=$REPO_URL,wakeword-branch=$BRANCH,wakeword-config=$CONFIG,\
wakeword-bucket=$BUCKET,wakeword-user=root" \
    --metadata-from-file=startup-script="$HERE/startup.sh" \
    >/dev/null 2>"$PIPELINE/.gcp_create_err"
  local rc=$?
  [ "$rc" -eq 0 ] || grep -qE "resource_availability|does not have enough resources" \
    "$PIPELINE/.gcp_create_err" || { tail -6 "$PIPELINE/.gcp_create_err" >&2; }
  return "$rc"
}

remote_state() { ssh_vm "sudo cat /mnt/work/vm_state 2>/dev/null" ; }

cmd_status() {
  vm_exists || { echo "VM $VM does not exist (already destroyed, or never created)"; return 0; }
  echo "vm      : $VM ($(vm_status))"
  echo "machine : $MACHINE + $GPU, zone $ZONE"
  local st; st="$(remote_state)"
  if [ -n "$st" ]; then
    echo "--- vm_state ---"; printf '%s\n' "$st"
  else
    echo "state   : (not reachable yet — still booting, or shut down)"
  fi
}

cmd_log() {
  vm_exists || die "no VM"
  ssh_vm "sudo tail -40 /mnt/work/run.log 2>/dev/null || sudo tail -40 /mnt/work/vm_pipeline.log 2>/dev/null || sudo tail -30 /var/log/wakeword-startup.log"
}

cmd_fetch() {
  local dest="$PIPELINE/output"; mkdir -p "$dest"
  if [ -n "$BUCKET" ]; then
    say "Fetching from $BUCKET"
    g storage cp "$BUCKET/output/*" "$dest/" 2>&1 | tail -5 && { ls -lh "$dest"; return 0; }
  fi
  vm_exists && [ "$(vm_status)" = RUNNING ] || die "VM not running and no bucket copy; try 'start' or check $BUCKET"
  say "Fetching over SSH"
  g compute scp --zone "$ZONE" --tunnel-through-iap --recurse \
    "$VM:/mnt/work/output/*" "$dest/" 2>&1 | tail -5
  ls -lh "$dest"
}

# Exit code IS the event, same contract as the Colab watcher:
#   0 complete   3 terminal failure   4 budget ceiling hit   5 no VM
cmd_watch() {
  vm_exists || { echo "no VM $VM"; exit 5; }
  local poll="${WAKEWORD_WATCH_POLL:-120}"
  local started; started="$(date +%s)"
  say "Watching $VM (poll ${poll}s, ceiling ${MAX_HOURS}h)"

  while true; do
    local status; status="$(vm_status)"
    local elapsed=$(( ($(date +%s) - started) / 60 ))

    # The VM powering itself off is the designed end state.
    if [ "$status" = "TERMINATED" ] || [ "$status" = "STOPPED" ]; then
      say "VM is $status after ${elapsed} min — it shut itself down"
      local st; st="$(g storage cat "$BUCKET/vm_state" 2>/dev/null)"
      [ -n "$st" ] && { echo "--- final vm_state ---"; printf '%s\n' "$st"; }
      if printf '%s' "$st" | head -1 | grep -q complete; then
        cmd_fetch; exit 0
      fi
      printf '%s' "$st" | head -1 | grep -q failed && exit 3
      # No bucket configured: the disk still holds the answer.
      warn "no terminal state readable; start the VM and check '$0 log'"
      exit 3
    fi

    # Cost guard: if the in-VM shutdown never fires, this stops the bleeding.
    if [ "$elapsed" -gt $(( MAX_HOURS * 60 )) ]; then
      warn "exceeded ${MAX_HOURS}h ceiling — stopping the VM to protect the budget"
      g compute instances stop "$VM" --zone "$ZONE" 2>&1 | tail -2
      exit 4
    fi

    local st; st="$(remote_state)"
    printf '[%3d min] %s | %s\n' "$elapsed" "$status" "$(printf '%s' "$st" | head -1 || echo booting)"
    { printf 'elapsed_min=%s\nvm_status=%s\n' "$elapsed" "$status"
      printf '%s\n' "$st"; } > "$PIPELINE/gcp_watch_state" 2>/dev/null || true
    case "$(printf '%s' "$st" | head -1)" in
      complete) say "run complete"; cmd_fetch; exit 0 ;;
      failed)   say "terminal failure"; printf '%s\n' "$st"; exit 3 ;;
    esac
    sleep "$poll"
  done
}

cmd_stop()    { vm_exists && g compute instances stop "$VM" --zone "$ZONE" && say "stopped (disk kept)"; }
cmd_destroy() { vm_exists && g compute instances delete "$VM" --zone "$ZONE" --quiet && say "destroyed"; }

case "${1:-}" in
  create) cmd_create ;; watch) cmd_watch ;; status) cmd_status ;;
  log) cmd_log ;; fetch) cmd_fetch ;; stop) cmd_stop ;; destroy) cmd_destroy ;;
  *) sed -n '3,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
