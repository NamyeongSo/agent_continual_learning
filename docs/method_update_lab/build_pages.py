# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The AgentStream organization and its contributors.

"""Produce direct-entry method pages and a dependency-free, single-file edition."""

from pathlib import Path


def main() -> None:
    dist = Path(__file__).resolve().parent / "dist"
    shell = (dist / "index.html").read_text()
    for slug, title in {
        "a_mem": "A-MEM",
        "autoskill": "AutoSkill",
        "harness": "Harness",
        "reasoning_bank": "ReasoningBank",
    }.items():
        (dist / f"{slug}.html").write_text(
            shell.replace("<title>One Update Lab — Agent Memory</title>",
                          f"<title>{title} · One Update Lab — Agent Memory</title>")
        )
    standalone = shell.replace('<link rel="stylesheet" href="styles.css">',
                               "<style>\n" + (dist / "styles.css").read_text() + "\n</style>")
    for script in ["source-data.js", "model.js", "app.js"]:
        code = (dist / script).read_text().replace("</script", "<\\/script")
        prefix = "window.LAB_STANDALONE = true;\n" if script == "source-data.js" else ""
        standalone = standalone.replace(f'<script defer src="{script}"></script>',
                                        f"<script>\n{prefix}{code}\n</script>")
    # Inline application code must run after the body exists, just as deferred scripts do.
    scripts = []
    while "<script>" in standalone:
        start = standalone.index("<script>")
        end = standalone.index("</script>", start) + len("</script>")
        scripts.append(standalone[start:end])
        standalone = standalone[:start] + standalone[end:]
    standalone = standalone.replace("</body>", "\n".join(scripts) + "\n</body>")
    license_path = dist / "LICENSE.txt"
    if license_path.exists():
        notice = (dist / "attribution.txt").read_text() + "\n" + license_path.read_text()
        standalone = standalone.replace("<head>", "<head>\n<!--\n" + notice.replace("-->", "-- >") + "\n-->")
    (dist / "standalone.html").write_text(standalone)
    print("Built index + 4 method pages + standalone.html")


if __name__ == "__main__":
    main()
