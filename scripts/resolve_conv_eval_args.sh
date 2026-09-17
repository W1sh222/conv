#!/usr/bin/env bash

# Resolve an optional Conv checkpoint for the RULER wrappers.
#
# The caller must initialise EXTRA_ARGS as an array and pass the default
# weight to resolve_conv_eval_args.  A final positional *.pt argument is
# accepted for convenience, as are --conv_weight_path PATH and the
# CONV_WEIGHT_PATH environment variable.  The explicit CLI value wins.
resolve_conv_eval_args() {
  local default_weight="$1"
  local -a filtered=()
  local cli_weight=""
  local i=0
  local n="${#EXTRA_ARGS[@]}"

  while (( i < n )); do
    local arg="${EXTRA_ARGS[$i]}"
    if [[ "${arg}" == "--conv_weight_path" ]]; then
      if (( i + 1 >= n )); then
        echo "--conv_weight_path requires a value" >&2
        return 2
      fi
      cli_weight="${EXTRA_ARGS[$((i + 1))]}"
      ((i += 2))
      continue
    fi
    if [[ "${arg}" == --conv_weight_path=* ]]; then
      cli_weight="${arg#--conv_weight_path=}"
      ((i += 1))
      continue
    fi
    # All normal RULER extras are option/value pairs.  Only consume a final
    # .pt positional argument so task/filter values are left untouched.
    if (( i == n - 1 )) && [[ "${arg}" == *.pt ]] && [[ "${arg}" != -* ]]; then
      cli_weight="${arg}"
      ((i += 1))
      continue
    fi
    filtered+=("${arg}")
    ((i += 1))
  done

  local resolved="${cli_weight:-${CONV_WEIGHT_PATH:-${default_weight}}}"
  EXTRA_ARGS=("${filtered[@]}" --conv_weight_path "${resolved}")
  CONV_WEIGHT_PATH="${resolved}"

  local tag
  if [[ "${resolved}" == "initial_vertical_diag" || "${resolved}" == "__initial_vertical_diag__" ]]; then
    tag="conv_kernel_7x7_initial_vertical_diag"
  else
    tag="$(basename -- "${resolved}")"
    tag="${tag%.pt}"
  fi
  # The checkpoint name is part of the result identity; do not let a stale
  # inherited tag make two different weights share one output directory.
  export RULER_RUN_TAG="${tag}"
}
