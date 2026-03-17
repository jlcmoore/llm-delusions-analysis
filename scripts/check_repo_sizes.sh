set -euo pipefail

# Check Git LFS and regular Git file sizes against configurable limits.
# Configuration (can be overridden via environment variables):
#   LFS_MAX_FILE_GIB      - maximum allowed single LFS file size in GiB (default: 1)
#   LFS_MAX_TOTAL_GIB     - maximum allowed total LFS size in GiB (default: 10)
#   REGULAR_MAX_FILE_MIB  - maximum allowed non-LFS Git file size in MiB (default: 50)

: "${LFS_MAX_FILE_GIB:=1}"
: "${LFS_MAX_TOTAL_GIB:=10}"
: "${REGULAR_MAX_FILE_MIB:=50}"

LFS_MAX_FILE_BYTES=$((LFS_MAX_FILE_GIB * 1024 * 1024 * 1024))
LFS_MAX_TOTAL_BYTES=$((LFS_MAX_TOTAL_GIB * 1024 * 1024 * 1024))

check_lfs_sizes() {
  # If there are no LFS files, this will print a message and succeed.
  if ! git lfs ls-files -l >/dev/null 2>&1; then
    echo "git lfs not available or no LFS tracking configured; skipping LFS checks."
    return 0
  fi

  # Use --size so we can parse the actual object size from output like:
  #   <oid> * path/to/file (50 MB)
  git lfs ls-files -l --size | awk -v max_file_bytes="$LFS_MAX_FILE_BYTES" -v max_total_bytes="$LFS_MAX_TOTAL_BYTES" '
  BEGIN {
    max = 0;
    sum = 0;
  }
  {
    line = $0;

    # Extract the trailing parenthesized size, e.g. "(50 MB)" at end of line,
    # ignoring any earlier parentheses that may appear in the file path.
    if (match(line, /\([^()]*[0-9][^()]*\)[ \t]*$/) == 0) {
      next;
    }

    size_str = substr(line, RSTART + 1, RLENGTH - 2);
    gsub(/^[ \t]+|[ \t]+$/, "", size_str);

    n = split(size_str, parts, /[ \t]+/);
    if (n < 1) {
      next;
    }

    value = parts[1] + 0;
    unit = "B";
    if (n >= 2) {
      unit = toupper(parts[2]);
    }

    multiplier = 1;
    if (unit == "KB") {
      multiplier = 1024;
    } else if (unit == "MB") {
      multiplier = 1024 * 1024;
    } else if (unit == "GB") {
      multiplier = 1024 * 1024 * 1024;
    } else if (unit == "TB") {
      multiplier = 1024 * 1024 * 1024 * 1024;
    }

    s = value * multiplier;
    if (s > max) {
      max = s;
    }
    sum += s;
  }
  END {
    if (NR == 0) {
      print "No Git LFS files found; skipping LFS checks.";
      exit 0;
    }

    gib = 1024 * 1024 * 1024;

    if (max > max_file_bytes) {
      printf "FAIL: LFS max file size %.2f GiB exceeds limit %.2f GiB\n", max / gib, max_file_bytes / gib;
      exit 1;
    }
    if (sum > max_total_bytes) {
      printf "FAIL: LFS total size %.2f GiB exceeds limit %.2f GiB\n", sum / gib, max_total_bytes / gib;
      exit 1;
    }

    printf "OK: LFS max=%.2f GiB, total=%.2f GiB (limits: max file %.2f GiB, total %.2f GiB)\n",
           max / gib, sum / gib, max_file_bytes / gib, max_total_bytes / gib;
  }'
}

check_regular_files() {
  limit_mib="$REGULAR_MAX_FILE_MIB"
  limit_bytes=$((limit_mib * 1024 * 1024))

  echo "Checking tracked non-LFS Git files for size > ${limit_mib} MiB (gitignored files are skipped)..."

  # Use git ls-files so the check is sensitive to .gitignore and only considers tracked files.
  if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Not inside a Git repository; skipping regular file checks."
    return 0
  fi

  # Build a list of paths that are tracked by Git LFS so we can exclude them
  # from the "regular file" size check. We avoid Bash-specific associative
  # arrays so this script can run under /bin/sh as well.
  lfs_list_file="$(mktemp)"
  git lfs ls-files --name-only -z 2>/dev/null >"$lfs_list_file" || true

  oversized_files=()

  # Read NUL-delimited list of tracked files.
  while IFS= read -r -d '' path; do
    # Skip if path is not a regular file (just in case).
    if [ ! -f "$path" ]; then
      continue
    fi

    # Skip files that are tracked by Git LFS; they are covered by the LFS checks.
    # Use grep -Fz to match the NUL-delimited list.
    if grep -Fz -- "$path" "$lfs_list_file" >/dev/null 2>&1; then
      continue
    fi

    size_bytes=$(wc -c < "$path")
    # shellcheck disable=SC2086
    if [ "$size_bytes" -gt "$limit_bytes" ]; then
      size_mib=$(( (size_bytes + 1024 * 1024 - 1) / (1024 * 1024) ))
      oversized_files+=("$path (${size_mib} MiB)")
    fi
  done < <(git ls-files -z)

  rm -f "$lfs_list_file"

  if [ "${#oversized_files[@]}" -gt 0 ]; then
    echo "FAIL: The following tracked files exceed ${limit_mib} MiB:"
    for entry in "${oversized_files[@]}"; do
      echo "  $entry"
    done
    return 1
  fi

  echo "OK: No tracked regular files exceed ${limit_mib} MiB."
}

main() {
  echo "LFS_MAX_FILE_GIB=${LFS_MAX_FILE_GIB}"
  echo "LFS_MAX_TOTAL_GIB=${LFS_MAX_TOTAL_GIB}"
  echo "REGULAR_MAX_FILE_MIB=${REGULAR_MAX_FILE_MIB}"
  echo

  check_lfs_sizes
  check_regular_files
}

main "$@"
