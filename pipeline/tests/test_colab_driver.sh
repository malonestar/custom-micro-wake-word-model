#!/usr/bin/env bash
# Tests for the Colab driver's remote-execution shim, against a stub CLI.
#
# remote_sh has to carry an arbitrary shell command through bash, the colab CLI
# and a Jupyter kernel without anything mangling it, then get a return code back
# out of a channel that does not carry one. Quoting bugs here are silent and
# would only show up as a confusing failure on a live runtime, so they are worth
# pinning down locally.
set -uo pipefail

STUB_DIR="$(mktemp -d)"
trap 'rm -rf "$STUB_DIR"' EXIT

# Stands in for the real CLI: `exec` runs the piped Python the way the kernel
# would, so a quoting error surfaces as a genuine failure.
cat > "$STUB_DIR/colab" <<'STUB'
#!/usr/bin/env bash
case "$1" in
  exec)     exec python3 ;;
  sessions) echo "[wakeword] gpu-t4-stub | Hardware: T4 | Status: IDLE" ;;
  *)        echo "stub: $*" ;;
esac
STUB
chmod +x "$STUB_DIR/colab"
export PATH="$STUB_DIR:$PATH"

source "$(dirname "${BASH_SOURCE[0]}")/../colab/colab_run.sh"
# The driver sets -e for its own dispatch; these tests deliberately run commands
# that fail, so turn it back off after sourcing.
set +e

PASS=0; FAIL=0
check() {
  local label="$1" expected="$2" actual="$3"
  if [ "$expected" = "$actual" ]; then
    PASS=$((PASS + 1)); printf '  ok    %s\n' "$label"
  else
    FAIL=$((FAIL + 1)); printf '  FAIL  %s\n        expected: %q\n        actual:   %q\n' \
      "$label" "$expected" "$actual"
  fi
}

check "plain command output" "hello" "$(remote_sh 30 'echo hello')"
check "single quotes survive" "it's fine" "$(remote_sh 30 "echo \"it's fine\"")"
check "double quotes survive" 'say "hi"' "$(remote_sh 30 'echo '"'"'say "hi"'"'"'')"
check "dollar signs are not expanded locally" '$HOME $(whoami) `id`' \
  "$(remote_sh 30 'echo '"'"'$HOME $(whoami) `id`'"'"'')"
check "backslashes survive" 'a\b\c' "$(remote_sh 30 'printf "%s" '"'"'a\b\c'"'"'')"
check "remote expansion still works" "expanded" \
  "$(remote_sh 30 'X=expanded; echo $X')"
check "multiline output" "$(printf 'one\ntwo')" "$(remote_sh 30 'echo one; echo two')"
check "sentinel is stripped from output" "clean" "$(remote_sh 30 'echo clean')"

remote_sh 30 'exit 0' >/dev/null; check "zero exit propagates" "0" "$?"
remote_sh 30 'exit 7' >/dev/null; check "nonzero exit propagates" "7" "$?"
remote_sh 30 'false'  >/dev/null; check "failing command propagates" "1" "$?"

# A command whose own output contains the sentinel must not fool the parser.
remote_sh 30 'echo __RC__=99; exit 3' >/dev/null
check "spoofed sentinel does not win" "3" "$?"

echo
if [ "$FAIL" -eq 0 ]; then
  echo "colab driver: $PASS passed"
else
  echo "colab driver: $PASS passed, $FAIL FAILED"; exit 1
fi
