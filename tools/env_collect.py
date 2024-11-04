# Diagnose your system and show basic information.
# Inspired by
# https://raw.githubusercontent.com/Lightning-AI/lightning/master/requirements/collect_env_details.py
import os
import platform
import subprocess
import sys

import pkg_resources

sys.path += [os.path.abspath(".."), os.path.abspath("")]


LEVEL_OFFSET = "\t"
KEY_PADDING = 20


def info_system() -> dict:
    """Get system information."""
    return {
        "OS": platform.system(),
        "architecture": platform.architecture(),
        "version": platform.version(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }


def info_cuda():
    """Get CUDA information."""
    try:
        info = subprocess.check_output("nvidia-smi").decode("utf-8")
    except FileNotFoundError:
        info = "No CUDA device found"
    info = info.replace("\n", f"\n{LEVEL_OFFSET}")
    return LEVEL_OFFSET + info


def info_packages() -> dict:
    """Get name and version of all installed packages."""
    packages = {}
    for dist in pkg_resources.working_set:
        package = dist.as_requirement()
        packages[package.key] = package.specs[0][1]
    return packages


def nice_print(details: dict, level: int = 0) -> list:
    """Print system information in a nice way."""
    lines = []
    for k in details:
        key = f"* {k}:" if level == 0 else f"- {k}:"
        if isinstance(details[k], dict):
            lines += [level * LEVEL_OFFSET + key]
            lines += nice_print(details[k], level + 1)
        elif isinstance(details[k], (set, list, tuple)):
            lines += [level * LEVEL_OFFSET + key]
            lines += [(level + 1) * LEVEL_OFFSET + "- " + v for v in details[k]]
        else:
            template = "{:%is} {}" % KEY_PADDING
            key_val = template.format(key, details[k])
            lines += [(level * LEVEL_OFFSET) + key_val]
    return lines


def main() -> None:
    """Print system information."""
    details = {"System": info_system(), "Packages": info_packages()}

    lines = nice_print(details)
    text = os.linesep.join(lines)
    text += "\n * CUDA:\n"
    text += str(info_cuda())
    print(f"Current environment \n\n {text}")


if __name__ == "__main__":
    main()
