"""Engine: stripe.

After ucalyptus/ParkSolver (MIT) — a parking layout built with no solver and
no model at all, purely by construction. Rotate the site to a trial angle,
subtract the obstacles, stripe what is left into rows one stall deep, run
stalls along each row at a fixed pitch, score the result, sweep the angle,
keep the best.

Rows are how an event site is actually drawn: grandstands face a stage in
ranks, market stalls line an aisle, toilet blocks stand in a bank, food trucks
park nose to tail. Every other engine here chooses positions one at a time and
gets a layout that validates; this one chooses a DIRECTION and gets a layout a
site manager recognises as designed. That regularity is the whole value — an
output that is not visibly in rows is a failure here even when `validate`
is happy.

It is also the only engine with no randomness: the answer is a function of the
ground and the programme, and the same site gives the same plan every time,
seed or no seed. `seconds` is read as a guard, not as a budget to spend — the
sweep is finite and normally finishes in a tenth of it.

Four places where the paper method needed adapting to a real venue, each with
its reason:

  * ParkSolver stripes the largest polygon. On this masterplan the free ground
    is 149 disjoint parts and the largest is 26 % of the area, so that would
    throw away three quarters of the site. The stripe lines here are global —
    one lattice of rows across the whole rotated frame, clipped to every part
    it crosses — which keeps the rows aligned with each other, which is what
    the eye reads as design, while still using the small ground.

  * Stalls in a car park sit on a fixed column lattice. Applied here, a 30 m
    grandstand found two legal columns on the entire site: the lattice phase
    is set by a bounding box that has nothing to do with where the ground is.
    So a row's positions are probed finely and the units are then laid at a
    fixed pitch from the probed run — even spacing within the row, no
    dependence on where the bounding box happens to start.

  * The rows are centred on an anchor rather than on the bottom of the frame:
    the centroid of whatever the group wants to be `near`, or the middle of
    the best ground when it wants nothing. That is the brief's "start the
    stripes from the side closest to that kind", and it is the difference
    between three grandstands facing the stage and three grandstands in the
    first gap the scan happened to reach.

  * A car park has one kind of stall. A programme has a dozen kinds placed in
    sequence, and the one placed first can ruin the ones that follow: a stage
    dropped in the middle of the only 240 m band leaves two half-bands, and no
    rank of seating can then face it. So a group that others must sit beside
    is charged with its own consequence — the ground is taken away and the
    FOLLOWER's unit striped into what is left within `ROOM` — and scored on
    how many of them could still stand there.

An item's `rotations` are read as turns RELATIVE TO THE ROW, so a stage whose
brief allows 0 and 90 may be laid square to a row running at 40 degrees. The
angle sweep is the method; reading `rotations` as absolute would reduce it to
a single trial.
"""

from __future__ import annotations

import math
import time

from core.placement import Placement

NAME = "stripe"
DOC = ("Constructive rows, after ParkSolver: rotate the ground to a trial angle, "
       "stripe it into rows one unit deep, run units along each row at a fixed "
       "pitch, sweep the angle and keep the best. Deterministic, needs no time "
       "budget, and the only engine whose output reads as rows rather than as "
       "scattered objects.")

SWEEP = tuple(range(0, 180, 10))   # trial row directions, degrees
ROOM = 60.0                        # metres: what `near` means to a visitor,
                                   # and to core.placement.score
EPS = 0.05                         # slack on every keep-out buffer, metres
MAX_PROBES = 1200                  # x samples per row
MAX_BOXES = 60000                  # containment tests before a trial gives up


# ── the geometry of one trial ────────────────────────────────────────────

def _turn(x, y, deg, ox, oy):
    """A point rotated by `deg` about (ox, oy) — the inverse of the frame
    rotation, used to bring an accepted centre back to site coordinates."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    dx, dy = x - ox, y - oy
    return ox + dx * c - dy * s, oy + dx * s + dy * c


def _runs(mask):
    """Maximal runs of True in a boolean array, as inclusive index pairs.

    A run is a stretch of the row where the unit fits at every probed x, so
    the unit also fits everywhere between them: one run is one continuous
    interval of legal left edges, which is what a pitch can be laid along."""
    import numpy as np
    m = np.asarray(mask).astype(np.int8)
    d = np.diff(np.concatenate(([0], m, [0])))
    return list(zip(np.flatnonzero(d == 1).tolist(),
                    (np.flatnonzero(d == -1) - 1).tolist()))


def _lay(ground, uw, uh, gap, n, anchor, phase=0.0):
    """Stripe `ground` into rows `uh` deep and fill them with up to `n` units.

    Returns (centres, units_per_row). The ground is already rotated into the
    trial frame, so everything here is axis-aligned. Rows are visited from the
    anchor outwards and abandoned as soon as `n` units are down, which is what
    keeps a sweep over 18 angles cheap: an easy group never scans the site.

    `phase` slides the whole lattice of rows along its own pitch. The rows stay
    evenly spaced whatever it is — it only decides where the rhythm starts, and
    on ground this broken that is worth two extra tries for a unit that would
    otherwise find no row at all."""
    import numpy as np
    import shapely

    if ground.is_empty:
        return [], []
    x0, y0, x1, y1 = ground.bounds
    if x1 - x0 < uw or y1 - y0 < uh:
        return [], []
    shapely.prepare(ground)

    ax, ay = anchor
    pitch_x, pitch_y = uw + gap, uh + gap
    base = ay - uh / 2 + phase * pitch_y     # bottom edge of the anchor's row
    lo_k = math.ceil((y0 - base) / pitch_y)
    hi_k = math.floor((y1 - uh - base) / pitch_y)
    if hi_k < lo_k:
        return [], []
    # Outwards from the anchor row, so a group grows around what it belongs to.
    ks = sorted(range(lo_k, hi_k + 1), key=lambda k: (abs(k), k))

    probe = max(0.5, min(uw, uh) / 4, (x1 - x0 - uw) / MAX_PROBES)
    xs = np.arange(x0, x1 - uw + 1e-9, probe)
    if not len(xs):
        return [], []

    out, per_row, spent = [], [], 0
    for k in ks:
        if len(out) >= n or spent > MAX_BOXES:
            break
        yb = base + k * pitch_y
        spent += len(xs)
        ok = shapely.contains(ground, shapely.box(xs, yb, xs + uw, yb + uh))
        if not ok.any():
            continue
        want = n - len(out)
        runs = [(xs[i], xs[j], int((xs[j] - xs[i]) / pitch_x) + 1)
                for i, j in _runs(ok)]
        # A stretch that can take the whole remainder first, then the nearest.
        # Both halves matter: a bank of six units is a bank, and six units in
        # one row but in six different stretches of it is not, however happily
        # it validates.
        runs.sort(key=lambda r: (-(r[2] >= want),
                                 min(abs(r[0] + uw / 2 - ax),
                                     abs(r[1] + uw / 2 - ax))))
        got = []
        for lo, hi, cap in runs:
            if len(out) >= n:
                break
            # Fill from whichever end of the run is closer to the anchor.
            near_lo = abs(lo + uw / 2 - ax) <= abs(hi + uw / 2 - ax)
            start, step = (lo, pitch_x) if near_lo else (hi, -pitch_x)
            took = 0
            for t in range(cap):
                if len(out) >= n:
                    break
                cx, cy = start + t * step + uw / 2, yb + uh / 2
                # Two identical axis-aligned units clear each other exactly
                # when they are a pitch apart on ONE axis. Within a run the
                # pitch guarantees it; across two runs of the same row it does
                # not — a rule that separates a group from its own kind is
                # otherwise broken by a unit at the near end of the next
                # stretch, which is a 15 m rule failing by 6 m.
                if any(abs(cx - qx) < pitch_x - 1e-6
                       and abs(cy - qy) < pitch_y - 1e-6 for qx, qy in out):
                    continue
                out.append((cx, cy))
                took += 1
            if took:
                got.append(took)
        if got:
            per_row.append(got)
    return out, per_row


def _compactness(pts, uw, uh):
    """Placed area over the area they span. The tie-break between two angles
    that fit the same number of units: the one that keeps them together."""
    if not pts:
        return 0.0
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    span = ((max(xs) - min(xs)) + uw) * ((max(ys) - min(ys)) + uh)
    return len(pts) * uw * uh / max(span, 1e-9)


def _key(pts, uw, uh, anchor, on_grain, follows, room=None):
    """How good a trial is, in the order the terms actually matter.

    Units placed first, always. Then:

      * a group with dependants — a stage three grandstands must sit beside —
        is ranked by how many of those dependants could still stand within
        `ROOM` of it afterwards. Without this the stage takes the middle of
        the best band, which is the one position from which no rank of
        seating can face it;
      * a group that is following something is ranked by how close it got,
        because being beside what it was told to be beside IS the placement;
      * everything else by how tightly its units sit together, which is what
        makes a row read as a row.

    Every quantity is rounded before it is compared. A metre of distance or a
    percent of compactness is not a reason to turn the whole site off its own
    grain, and `on_grain` is what breaks the tie once they are rounded away."""
    ax, ay = anchor
    far = sum(math.dist(p, (ax, ay)) for p in pts) / len(pts)
    comp = round(_compactness(pts, uw, uh), 1)
    near = -round(far / 5)
    if room is not None:
        return (len(pts), comp, room, int(on_grain), near)
    return ((len(pts), near, comp, int(on_grain)) if follows
            else (len(pts), comp, int(on_grain), near))


# ── the programme, cut into rows ─────────────────────────────────────────

def _room(ground, pts, uw, uh, gap, follower):
    """How much of what must sit beside this group could still stand there.

    One step of lookahead, and it earns its cost. A stage is placed before the
    grandstands that face it, and the position that looks best for the stage —
    the middle of the longest open band — is the position that cuts that band
    in two and leaves no rank able to face it. So the trial is charged with
    its own consequence: take the ground away, keep what is left within `ROOM`,
    and stripe the FOLLOWER's unit into it. The number that comes back is how
    many stands this stage position can be faced by.

    Falls back to bare open area when the follower's geometry is unknown."""
    from shapely.geometry import Point, box
    from shapely.ops import unary_union
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    left = ground.intersection(Point(cx, cy).buffer(ROOM, quad_segs=6))
    left = left.difference(unary_union(
        [box(x - uw / 2 - gap, y - uh / 2 - gap,
             x + uw / 2 + gap, y + uh / 2 + gap) for x, y in pts]))
    if follower is None or left.is_empty:
        return 0, round(left.area / 250)
    fw, fh, fgap, fn = follower
    fits = max(len(_lay(left, a, b, fgap, fn, (cx, cy))[0])
               for a, b in {(fw, fh), (fh, fw)})
    return fits, round(left.area / 250)


def _units(items):
    """Group items into rows-to-be: same kind, same block, same size.

    Kind alone is not enough — a shade group holding 15 m sails and 5 m
    umbrellas has no single row depth, and a row of mismatched units is not a
    row. Sorted by area so the group that has the fewest legal positions
    chooses its ground first."""
    groups = {}
    for it in items:
        tag = (it.kind, it.block, round(it.w, 2), round(it.h, 2))
        groups.setdefault(tag, []).append(it)
    out = []
    for tag, got in groups.items():
        out.append({"tag": tag, "kind": tag[0], "items": got,
                    "w": got[0].w, "h": got[0].h,
                    "area": got[0].w * got[0].h,
                    "overhead": any(i.overhead for i in got),
                    "gap": max(i.clearance for i in got)})
    out.sort(key=lambda g: -g["area"])
    return out


def _wants(a, b):
    """Does group `a` ask to be near group `b`?"""
    keys = {i.key for i in b["items"]}
    return any(w == b["kind"] or w in keys for i in a["items"] for w in i.near)


def _order(groups):
    """Targets before the groups that want to be near them, then biggest first.

    A grandstand's whole placement is "facing the stage", so the stage has to
    exist before the grandstands are striped. Everything else keeps the
    largest-first order that stops a bin taking the ground a stage needed."""
    todo, out = list(groups), []
    while todo:
        ready = [g for g in todo
                 if not any(_wants(g, o) for o in todo if o is not g)]
        pool = ready or todo          # a cycle of near rules: biggest wins
        pick = max(pool, key=lambda g: g["area"])
        out.append(pick)
        todo.remove(pick)
    return out


def _pitch_gap(group):
    """The gap to leave between two units of this group, and why.

    Usually the clearance. But a rule can name the group's own kind — six
    generators that must each stand 25 m from the next — and then the row's
    pitch IS the rule: nothing else in this engine separates two units of one
    group, because they are laid in a single pass. Widening both pitches is
    exact, since two rectangles offset by at least the gap on either axis are
    at least that far apart edge to edge, diagonal neighbours included."""
    keys = {i.key for i in group["items"]}
    gap = max(i.clearance for i in group["items"])
    rule = 0.0
    for it in group["items"]:
        if it.min_far > 0 and (it.far and
                               (group["kind"] in it.far or keys & set(it.far))):
            rule = max(rule, it.min_far)
    return max(gap, rule), rule


def _rule(group, other):
    """Metres a declared separation rule demands between this group and one
    already-placed item, or zero when no rule names the pair.

    Separation is a property of the pair, so a rule declared on the toilet
    still binds the food truck striped four groups later. Reading it in one
    direction only is the trap cpsat.py documents."""
    keys = {i.key for i in group["items"]}
    d = 0.0
    for it in group["items"]:
        if it.min_far > 0 and (other.key in it.far or other.kind in it.far):
            d = max(d, it.min_far)
    if other.min_far > 0 and (other.far and
                              (keys & set(other.far) or group["kind"] in other.far)):
        d = max(d, other.min_far)
    return d


def _ground(free, group, occupied, index):
    """The ground this group may stripe: the site, less what is standing.

    An overhead group — a sail, a canopy — is meant to span what is under it,
    so a footprint it merely covers takes nothing away; it still keeps off the
    venue's own structures, because a sail slung over a building is not a plan.

    A separation rule is not a statement about touching, though, and `validate`
    measures `min_far` between a sail and a generator exactly as it does
    between two tents. So a rule naming the pair cuts its distance out of the
    ground whichever of the two is overhead — otherwise a sail told to stand
    25 m off the power lands on top of it and the engine reports success."""
    from shapely.ops import unary_union
    keep = []
    for p in occupied:
        o = index.get(p.key)
        if o is None:
            continue
        d = _rule(group, o)
        if group["overhead"] or o.overhead:
            if d <= 0:
                continue
        else:
            d = max(d, group["gap"], o.clearance)
        keep.append(p.footprint(o).buffer(d + EPS))
    if not keep:
        return free
    return free.difference(unary_union(keep))


def _anchor(ground, group, placed, index, fallback):
    """Where this group's rows start from: the thing it wants to be near, or
    the middle of the biggest piece of ground it has."""
    want = {w for i in group["items"] for w in i.near}
    if want:
        hits = [p for p in placed
                if p.key in want or (index.get(p.key) and index[p.key].kind in want)]
        if hits:
            return ((sum(p.x for p in hits) / len(hits),
                     sum(p.y for p in hits) / len(hits)),
                    f"starts from {hits[0].key.rsplit('-', 1)[0]}", True)
    parts = list(ground.geoms) if ground.geom_type.startswith("Multi") else [ground]
    parts = [p for p in parts if p.area > 0]
    if parts:
        big = max(parts, key=lambda p: p.area)
        c = big.centroid
        if not big.contains(c):
            c = big.representative_point()
        return (c.x, c.y), "centred on the open ground", False
    return fallback, "centred on the site", False


def _grain(free):
    """The site's own directions: the long side of the minimum rotated
    rectangle of the free ground, and of its largest piece. On a waterfront
    strip that is the water's edge — the angle a designer would reach for
    before trying anything on a 10 degree sweep."""
    parts = list(free.geoms) if free.geom_type.startswith("Multi") else [free]
    out = []
    for g in [free, max(parts, key=lambda p: p.area)] if parts else [free]:
        try:
            c = list(g.minimum_rotated_rectangle.exterior.coords)
        except Exception:
            continue
        best = (0.0, 0.0)
        for i in range(len(c) - 1):
            dx, dy = c[i + 1][0] - c[i][0], c[i + 1][1] - c[i][1]
            if (L := math.hypot(dx, dy)) > best[0]:
                best = (L, math.degrees(math.atan2(dy, dx)) % 180)
        out.append(round(best[1], 1))
    return out


# ── the engine ───────────────────────────────────────────────────────────

def solve(items, free, region, fixed=(), items_by_key=None, seconds=20.0,
          seed=0, **_):
    from shapely.geometry import box

    t0 = time.time()
    notes, placed = [], []
    occupied = list(fixed)
    index = dict(items_by_key or {})
    index.update({it.key: it for it in items})

    # A committed placement whose Item nobody passed has no footprint, so it
    # cannot be subtracted from the ground and this engine will stripe over
    # it. Say so rather than quietly lay a row through it.
    if blind := [p.key for p in fixed if p.key not in index]:
        notes.append(f"already-placed {', '.join(blind)} carry no item and "
                     f"were treated as empty ground")

    # `free` is simplified to 5 cm when it is built, which can push it a
    # hair outside the region it was cut from — and `validate` measures the
    # site boundary with a strict containment. One intersection removes the
    # whole class of off-site-by-four-centimetres failure.
    free = free.intersection(box(*region))
    if free.is_empty:
        return [], ["no free ground in the adopted region"]
    o = free.centroid
    origin = (o.x, o.y)

    todo = []
    for it in items:
        if it.fixed:                       # pinned by the brief: geometry, not a choice
            x, y, rot = (list(it.fixed) + [0.0])[:3]
            p = Placement(it.key, it.block, float(x), float(y), float(rot),
                          label=it.label)
            placed.append(p)
            occupied.append(p)
        else:
            todo.append(it)
    if not todo:
        return placed, notes + ["every item was pinned by the brief"]

    grain = _grain(free)
    # The site's own directions first, then the square ones, then the
    # diagonals. Trials are kept on a strict improvement, so an angle that
    # merely equals what the grain already achieved never displaces it: a row
    # is turned off the grain only when turning it fits more units.
    angles = list(dict.fromkeys(
        grain + [a for a in SWEEP if a % 90 == 0]
        + [a for a in SWEEP if a % 90]))
    notes.append(f"site grain {'/'.join(f'{a:g}' for a in grain)} deg, "
                 f"swept against {len(angles)} directions")

    order = _order(_units(todo))
    for i, group in enumerate(order):
        n = len(group["items"])
        # What still has to sit beside this group once it is down. Only groups
        # not yet placed count: a rank already striped cannot be helped by
        # where the stage goes now.
        after = [g for g in order[i + 1:] if _wants(g, group)]
        follower = None
        if after:
            f = max(after, key=lambda g: g["area"])
            follower = (f["w"], f["h"], _pitch_gap(f)[0], len(f["items"]))
        ground = _ground(free, group, occupied, index)
        anchor, why, follows = _anchor(ground, group, placed, index, origin)
        gap, self_rule = _pitch_gap(group)
        if self_rule:
            notes.append(f"{group['kind']}: pitch widened to {gap:g} m — the "
                         f"group's own separation rule binds its own units")

        # One in-row orientation per distinct footprint: 0 and 90 over an
        # 18-direction sweep already covers every row direction and every way
        # a unit can sit in it, so a third turn would only repeat one.
        sizes = {}
        for r in group["items"][0].rotations:
            if r % 90:
                continue
            sizes.setdefault(tuple(round(v, 3) for v in group["items"][0].size(r)),
                             float(r % 360))

        from shapely import affinity
        best = None
        # One pass on the anchor's own phase; a second, only for a group that
        # came up short, on the two intermediate phases. Two thirds of the
        # sweep's cost is thereby never paid on the groups that fit.
        for phases in ((0.0,), (1 / 3, 2 / 3)):
            for angle in angles:
                rot = affinity.rotate(ground, -angle, origin=origin)
                ax, ay = _turn(anchor[0], anchor[1], -angle, *origin)
                for (uw, uh), turn in sizes.items():
                    for ph in phases:
                        pts, rows = _lay(rot, uw, uh, gap, n, (ax, ay), ph)
                        if not pts:
                            continue
                        key = _key(pts, uw, uh, (ax, ay),
                                   angle in grain or angle % 90 == 0, follows,
                                   _room(rot, pts, uw, uh, gap, follower)
                                   if follower else None)
                        if best is None or key > best[0]:
                            best = (key, angle, turn, pts, rows, uw, uh)
                if time.time() - t0 > seconds * 0.9:
                    break
            if best and best[0][0] >= n:
                break
            if time.time() - t0 > seconds * 0.9:
                notes.append(f"{group['kind']}: sweep cut short by the "
                             f"{seconds:g}s budget")
                break

        if best is None:
            for it in group["items"]:
                notes.append(f"{it.key}: no row {group['w']:g}x{group['h']:g} m "
                             f"fits anywhere on the ground left to it")
            continue

        _, angle, turn, pts, rows, uw, uh = best
        comp = _compactness(pts, uw, uh)
        for it, (cx, cy) in zip(group["items"], pts):
            x, y = _turn(cx, cy, angle, *origin)
            p = Placement(it.key, it.block, round(x, 2), round(y, 2),
                          round((angle + turn) % 360, 1), label=it.label)
            placed.append(p)
            occupied.append(p)
        notes.append(
            f"{group['kind']} x{len(pts)}/{n}: {angle:g} deg"
            f"{' (site grain)' if angle in grain else ''}, "
            f"{len(rows)} row(s) of "
            f"{', '.join('+'.join(str(c) for c in r) for r in rows)}, "
            f"{why}, compactness {comp:.2f}")
        for it in group["items"][len(pts):]:
            notes.append(f"{it.key}: no further row of "
                         f"{group['w']:g}x{group['h']:g} m on the ground left")

    notes.append(f"{len(placed)}/{len(items)} placed by construction in "
                 f"{time.time() - t0:.1f}s, no search")
    return placed, notes
