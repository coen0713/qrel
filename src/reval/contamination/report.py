"""Rendering for the M3 experiment.

The acceptance table (rows = eval condition, columns = metric, cells = point
estimate with 95% CI) plus the four pre-registered hypothesis tests, each
labelled with the verdict its pre-stated threshold produces — so the reader sees
the criterion and the result together rather than being asked to trust a
narrative built after the fact.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

console = Console()


def kendall_tau(a: list[str], b: list[str]) -> float:
    """Kendall's tau between two orderings of the same items."""
    items = [x for x in a if x in b]
    if len(items) < 2:
        return 1.0
    pos_a = {x: i for i, x in enumerate(a)}
    pos_b = {x: i for i, x in enumerate(b)}
    concordant = discordant = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            x, y = items[i], items[j]
            sign = (pos_a[x] - pos_a[y]) * (pos_b[x] - pos_b[y])
            if sign > 0:
                concordant += 1
            elif sign < 0:
                discordant += 1
    total = concordant + discordant
    return (concordant - discordant) / total if total else 1.0


def h1_verdict(points: float, significant: bool) -> str:
    """Apply the pre-registered H1 thresholds (preregistration §4)."""
    if not significant or abs(points) < 2.0:
        return "[yellow]NULL[/yellow]"
    if points >= 5.0:
        return "[green]CONFIRMED[/green]"
    if points <= -2.0:
        return "[red]OPPOSITE DIRECTION[/red]"
    return "[cyan]weak/ambiguous[/cyan]"


def render_acceptance_table(report, dataset: str) -> Table:
    """The M3 acceptance criterion: one table, conditions x metrics, with CIs."""
    measures = sorted(
        report.conditions[0].estimates,
        key=lambda m: (int(m.split("@")[1]), m.split("@")[0]),
    )
    headline = [m for m in measures if m.endswith("@10")] or measures[:3]

    table = Table(
        title=(
            f"{dataset} — {report.chunker} chunks, {report.retriever}  "
            f"(mean [95% CI], percentage points)"
        ),
        header_style="bold",
    )
    table.add_column("eval condition", style="bold")
    table.add_column("n", justify="right")
    table.add_column("overlap", justify="right")
    table.add_column("q chars", justify="right")
    for m in headline:
        table.add_column(m, justify="right")

    for c in report.conditions:
        table.add_row(
            c.name,
            f"{c.n_queries:,}",
            f"{c.overlap_means.get('jaccard_5gram', 0):.3f}",
            f"{c.mean_query_chars:.0f}",
            *(c.estimates[m].as_pct() for m in headline),
        )
    return table


def render(reports, primary: str, dataset: str) -> None:
    """Print the acceptance table and every hypothesis test."""
    for r in reports:
        console.print()
        console.print(render_acceptance_table(r, dataset))
        _render_h1(r, primary)
        _render_h2(r, primary)
    _render_h3(reports)
    _render_h4(reports, primary)


def _render_h1(report, primary: str) -> None:
    if not report.contrasts:
        return
    table = Table(
        title=f"H1 — {primary}: synthetic minus human (UNPAIRED bootstrap)",
        header_style="bold",
    )
    table.add_column("condition")
    table.add_column("difference", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("p", justify="right")
    table.add_column("pre-registered verdict")
    for name, cmp in sorted(report.contrasts.items()):
        points = cmp.mean_diff * 100
        table.add_row(
            name,
            f"{points:+.2f} pts",
            f"[{cmp.lo * 100:+.2f}, {cmp.hi * 100:+.2f}]",
            f"{cmp.p_value:.4f}",
            h1_verdict(points, cmp.significant),
        )
    console.print(table)


def _render_h2(report, primary: str) -> None:
    for name, rows in sorted(report.deciles.items()):
        if not rows:
            continue
        table = Table(
            title=f"H2 — {name}: {primary} by query-gold overlap decile", header_style="bold"
        )
        table.add_column("decile", justify="right")
        table.add_column("n", justify="right")
        table.add_column("mean Jaccard", justify="right")
        table.add_column(primary, justify="right")
        for row in rows:
            table.add_row(
                str(row.decile),
                str(row.n),
                f"{row.mean_overlap:.3f}",
                row.estimates[primary].as_pct(),
            )
        console.print(table)

        rho, lo, hi = report.overlap_correlation.get(name, (0.0, 0.0, 0.0))
        excludes_zero = lo > 0 or hi < 0
        gap = (rows[-1].estimates[primary].mean - rows[0].estimates[primary].mean) * 100
        confirmed = rho > 0 and excludes_zero and gap >= 10
        verdict = "[green]CONFIRMED[/green]" if confirmed else "[yellow]NULL[/yellow]"
        console.print(
            f"  Spearman rho(overlap, {primary}) = {rho:+.3f} [{lo:+.3f}, {hi:+.3f}]   "
            f"top-bottom decile = {gap:+.1f} pts   {verdict}"
        )


def _render_h3(reports) -> None:
    """BM25 should inflate more than dense; it scores lexical overlap directly."""
    by_chunker: dict[str, dict[str, object]] = {}
    for r in reports:
        by_chunker.setdefault(r.chunker, {})[r.retriever] = r

    table = Table(
        title="H3 — inflation by retriever (BM25 predicted to exceed dense)", header_style="bold"
    )
    table.add_column("chunker")
    table.add_column("condition")
    table.add_column("BM25", justify="right")
    table.add_column("dense", justify="right")
    table.add_column("BM25 - dense", justify="right")

    rows = 0
    for chunker, group in sorted(by_chunker.items()):
        if "bm25" not in group or "dense" not in group:
            continue
        for name in sorted(group["bm25"].contrasts):
            if name not in group["dense"].contrasts:
                continue
            b = group["bm25"].contrasts[name].mean_diff * 100
            d = group["dense"].contrasts[name].mean_diff * 100
            table.add_row(chunker, name, f"{b:+.2f}", f"{d:+.2f}", f"{b - d:+.2f}")
            rows += 1
    if rows:
        console.print(table)


def _render_h4(reports, primary: str) -> None:
    """Does the eval set change which chunker looks best?"""
    by_retriever: dict[str, list] = {}
    for r in reports:
        by_retriever.setdefault(r.retriever, []).append(r)

    for retriever, group in sorted(by_retriever.items()):
        if len(group) < 2:
            continue  # an ordering needs at least two chunkers
        condition_names = [c.name for c in group[0].conditions]

        table = Table(title=f"H4 — chunker ranking by {primary} ({retriever})", header_style="bold")
        table.add_column("eval condition")
        for r in group:
            table.add_column(r.chunker, justify="right")
        table.add_column("ordering (best first)")

        orderings: dict[str, list[str]] = {}
        for cond_name in condition_names:
            cells: list[str] = []
            scored: list[tuple[float, str]] = []
            for r in group:
                c = r.condition(cond_name)
                if c is None:
                    cells.append("-")
                    continue
                cells.append(c.estimates[primary].as_pct())
                scored.append((c.estimates[primary].mean, r.chunker))
            orderings[cond_name] = [name for _, name in sorted(scored, reverse=True)]
            table.add_row(cond_name, *cells, " > ".join(orderings[cond_name]))
        console.print(table)

        human_order = orderings.get("human")
        if human_order is None:
            continue
        for name, order in orderings.items():
            if name == "human":
                continue
            tau = kendall_tau(human_order, order)
            if tau == 1.0:
                verdict = "[yellow]identical ordering (NULL)[/yellow]"
            else:
                # The pre-registration is explicit that tau < 1 alone is not
                # evidence of a reordering when the chunkers' intervals overlap.
                # With four chunkers this test is weak by construction, and
                # saying so is part of reporting it honestly.
                verdict = (
                    f"[cyan]reordered, tau={tau:+.2f}[/cyan] "
                    "— not a finding unless the swapped pair's CIs separate"
                )
            console.print(f"  human vs {name}: {verdict}")
