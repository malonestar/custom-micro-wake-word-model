#!/usr/bin/env bash
# =============================================================================
#  Place the training VM wherever there is GPU capacity, keeping its disk.
#
#  A single-GPU quota exists for T4, L4, V100 and P100 in every US region, but
#  actual capacity for any one of them in any one zone comes and goes. Rather
#  than wait on the zone that happened to work first, restore the snapshot into
#  whichever (zone, GPU) pair will take it right now — the disk carries the
#  downloaded datasets and a completed bootstrap, so nothing is re-done.
#
#  L4 is preferred over T4: newer, usually more available, and the machine
#  shape that carries it (g2) has more vCPU, which is what feature generation
#  is actually bound by.
# =============================================================================
set -uo pipefail
G="${GCLOUD_BIN:-$HOME/google-cloud-sdk/bin/gcloud}"
VM="${WAKEWORD_VM:-wakeword-trainer}"
SNAP="${WAKEWORD_SNAPSHOT:-wakeword-snap}"
BUCKET="${WAKEWORD_BUCKET:-gs://gen-lang-client-0502362419-wakeword}"
CONFIG="${WAKEWORD_CONFIG:-config/fbi_guy.yaml}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE="$(cd "$HERE/.." && pwd)"

# "machine:accelerator" — g2 shapes have their L4 attached implicitly.
CANDIDATES="${WAKEWORD_CANDIDATES:-g2-standard-8:nvidia-l4 n1-standard-8:nvidia-tesla-t4 n1-standard-8:nvidia-tesla-v100 n1-standard-8:nvidia-tesla-p100}"
ZONES="${WAKEWORD_ZONES:-us-central1-a us-central1-b us-central1-c us-central1-f us-east1-b us-east1-c us-east1-d us-east4-a us-east4-b us-east4-c us-west1-a us-west1-b us-west4-a us-west4-b us-west2-b us-west3-b}"

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[33m    %s\033[0m\n' "$*"; }

$G compute instances describe "$VM" --zone "$(cat "$PIPELINE/.gcp_zone" 2>/dev/null || echo us-east1-c)" >/dev/null 2>&1 \
  && { warn "an instance named $VM still exists; delete it first"; exit 1; }

for pair in $CANDIDATES; do
  machine="${pair%%:*}"; accel="${pair##*:}"
  for z in $ZONES; do
    printf '  trying %-18s %-20s %s\n' "$machine" "$accel" "$z"
    disk="${VM}-${z}"
    $G compute disks describe "$disk" --zone "$z" >/dev/null 2>&1 || \
      $G compute disks create "$disk" --zone "$z" --source-snapshot "$SNAP" \
         --type pd-balanced >/dev/null 2>&1 || continue

    # g2 machines come with their L4; everything else needs it attached.
    accel_flag=(--accelerator="type=$accel,count=1")
    [[ "$machine" == g2-* ]] && accel_flag=()

    if $G compute instances create "$VM" --zone "$z" \
        --machine-type="$machine" "${accel_flag[@]}" \
        --maintenance-policy=TERMINATE \
        --disk="name=$disk,boot=yes,auto-delete=yes" \
        --scopes=https://www.googleapis.com/auth/cloud-platform \
        --metadata="install-nvidia-driver=True,wakeword-config=$CONFIG,\
wakeword-bucket=$BUCKET,wakeword-user=root" \
        --metadata-from-file=startup-script="$HERE/startup.sh" \
        >/dev/null 2>&1; then
      echo "$z" > "$PIPELINE/.gcp_zone"
      printf '%s\n' "$machine" > "$PIPELINE/.gcp_machine"
      say "PLACED: $machine + $accel in $z"
      exit 0
    fi
    # No capacity here — do not leave a 200 GB disk behind in every zone tried.
    $G compute disks delete "$disk" --zone "$z" --quiet >/dev/null 2>&1
  done
done
warn "no (machine, GPU, zone) combination had capacity"
exit 1
