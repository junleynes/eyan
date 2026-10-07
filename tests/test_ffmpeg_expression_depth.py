"""
Sums handed to ffmpeg as expressions must stay shallow however long they get.

FFmpeg 8 refuses an expression whose parse tree is more than 100 levels
deep, and does it silently: the filter using it reports only "Failed to
configure input pad on Parsed_crop_4" and the render is lost. ffmpeg reads
a+b+c+d as ((a+b)+c)+d -- one level per term -- so every expression PRISM
builds by adding up one term per shot, per keyframe or per line of dialogue
hit that wall once there were about a hundred of them. On a real episode
that was 15 of 20 vertical shorts: the ones that followed movement for most
of a minute.

The ffmpeg on a test machine may well be an older one that has no such
limit, so these tests measure the depth the way FFmpeg's parser counts it
rather than hoping the local binary objects.
"""
import ast
import math
import re

import pipeline
import shorts_core as sc

FFMPEG_MAX_DEPTH = 100          # libavutil/eval.c, MAX_DEPTH
ROOM = 40                       # what we allow ourselves: well clear of it


def _depth(expr):
    """How deep ffmpeg's parser would build `expr`: 0 for a number or a
    variable, one more than its deepest operand for every operator and
    function call. (These expressions are valid Python too, which saves
    writing a parser; walked without recursion, since the point is that
    some of them are very deep.)"""
    tree = ast.parse(expr, mode='eval').body
    deepest, todo = 0, [(tree, 0)]
    while todo:
        node, above = todo.pop()
        if isinstance(node, ast.BinOp):
            kids = [node.left, node.right]
        elif isinstance(node, ast.UnaryOp):
            kids = [node.operand]
        elif isinstance(node, ast.Call):
            kids = list(node.args)
        else:
            deepest = max(deepest, above)
            continue
        todo += [(k, above + 1) for k in kids]
    return deepest


def _value(expr, **names):
    def between(v, a, b):
        return 1.0 if a <= v <= b else 0.0

    def clip(v, lo, hi):
        return max(lo, min(hi, v))
    return eval(expr, {'__builtins__': {}}, dict(names, between=between, clip=clip, min=min))


def test_a_plain_sum_is_as_deep_as_it_is_long_which_is_the_problem():
    flat = '+'.join(f'between(n,{k},{k})*{k}' for k in range(150))
    assert _depth(flat) > FFMPEG_MAX_DEPTH


def test_expr_sum_stays_shallow_and_adds_up_to_the_same_thing():
    assert sc.expr_sum([]) == '0' and sc.expr_sum(['a']) == 'a'
    assert sc.expr_sum(['a', 'b', 'c']) == 'a+b+c', 'a short sum is written plainly'
    assert sc.expr_sum(str(k) for k in range(8)) == '0+1+2+3+4+5+6+7'
    assert sc.expr_sum(str(k) for k in range(9)) == '(0+1+2+3+4+5+6+7)+(8)'
    for count in (9, 64, 65, 100, 513, 5000):
        terms = [f'between(n,{k},{k})*{k}' for k in range(count)]
        expr = sc.expr_sum(terms)
        assert _depth(expr) <= ROOM, count
        assert max(len(m) for m in re.findall(r'\(+', expr)) < 20, 'and never deeply bracketed either'
        for n in (0, 7, 8, count // 2, count - 1):
            assert _value(expr, n=n) == n, (count, n)
        assert expr.count('between') == count


def _following(seconds, fps=25.0):
    """A plan for someone walking about for the whole clip, a cut every six
    seconds: every shot follows movement, which is what makes the crop
    path long."""
    n = int(seconds * fps)
    samples = [(i, [(960 + 500 * math.sin(i / 55.0) + 120 * math.sin(i / 9.0), 400, 200, 200)])
               for i in range(0, n, 5)]
    return sc.plan_reframe(samples, list(range(150, n, 150)), n, 1920, 1080, 608, fps=fps), n


def _planned_x(segs, n):
    for s in segs:
        if s['a'] <= n <= s['b']:
            if not s.get('keys'):
                return float(round(s['x'] or 0))
            for (f0, x0), (f1, x1) in zip(s['keys'], s['keys'][1:]):
                if f0 <= n <= f1:
                    return x0 + (x1 - x0) * (n - f0) / float(f1 - f0)
    raise AssertionError(n)


def test_a_minute_of_following_movement_still_makes_a_crop_ffmpeg_will_take():
    segs, n = _following(75)
    pieces = sum(len(s['keys']) - 1 for s in segs if s.get('keys'))
    assert pieces > FFMPEG_MAX_DEPTH, 'the case that failed: more pieces to the path than ffmpeg allows levels'
    expr = sc.fitted_crop_x_expr(segs)
    assert _depth(expr) <= ROOM
    spans = [(int(a), int(b)) for a, b in re.findall(r'between\(n,(\d+),(\d+)\)', expr)]
    for k in range(n):
        assert sum(1 for a, b in spans if a <= k <= b) == 1, f'frame {k} is covered exactly once'
    worst = max(abs(_value(expr, n=k) - _planned_x(segs, k)) for k in range(n))
    assert worst <= 1.1, 'and the window is where the planner put it, to the pixel'
    g = sc.build_filtergraph({'disp_w': 1920, 'disp_h': 1080, 'sd_matrix': False}, segs)
    assert f"x='{expr}'" in g


def test_keys_that_say_nothing_are_dropped_and_the_ones_that_turn_the_path_are_kept():
    straight = [(f, 100.0 + 2.0 * f) for f in range(0, 101, 10)]
    assert sc.thin_keys(straight) == [straight[0], straight[-1]]
    bend = [(0, 0.0), (10, 50.0), (20, 100.0), (30, 100.0), (40, 100.0), (50, 40.0)]
    assert sc.thin_keys(bend) == [(0, 0.0), (20, 100.0), (40, 100.0), (50, 40.0)]
    assert sc.thin_keys(bend, tol=0) == bend and sc.thin_keys(bend[:2]) == bend[:2] and sc.thin_keys([]) == []
    wobble = [(f, 300.0 + 80.0 * math.sin(f / 30.0)) for f in range(0, 601, 10)]
    for tol in (1.0, 4.0, 16.0):
        kept = sc.thin_keys(wobble, tol)
        assert kept[0] == wobble[0] and kept[-1] == wobble[-1] and len(kept) < len(wobble)
        for f, x in wobble:
            (f0, x0), (f1, x1) = next((a, b) for a, b in zip(kept, kept[1:]) if a[0] <= f <= b[0])
            assert abs(x - (x0 + (x1 - x0) * (f - f0) / float(f1 - f0))) <= tol + 1e-6
    assert len(sc.thin_keys(wobble, 16.0)) < len(sc.thin_keys(wobble, 1.0))


def test_the_longest_clip_allowed_still_fits_on_a_windows_command_line():
    segs, n = _following(300)
    assert len(sc.crop_x_expr(segs, tol=0)) > 32767, 'unthinned, the crop path alone is over the limit'
    expr = sc.fitted_crop_x_expr(segs)
    assert len(expr) <= sc.MAX_CROP_EXPR and _depth(expr) <= ROOM
    spans = [(int(a), int(b)) for a, b in re.findall(r'between\(n,(\d+),(\d+)\)', expr)]
    assert sorted(spans)[0][0] == 0 and sorted(spans)[-1][1] == n - 1
    assert all(b + 1 == a2 for (_, b), (a2, _) in zip(sorted(spans), sorted(spans)[1:])), 'no frame left out'
    assert len(sc.build_filtergraph({'disp_w': 1920, 'disp_h': 1080, 'sd_matrix': False}, segs)) < 20000


def test_a_clip_that_changes_layout_on_many_cuts_keeps_its_switches_shallow():
    segs = []
    for k in range(240):
        layout = ('crop', 'fit', 'split')[k % 3]
        seg = {'a': k * 25, 'b': k * 25 + 24, 'layout': layout, 'x': 300.0 if layout == 'crop' else None, 'keys': None}
        if layout == 'split':
            seg.update(size=(1216, 1080), panes=[(0, 0), (704, 0)])
        segs.append(seg)
    g = sc.build_filtergraph({'disp_w': 1920, 'disp_h': 1080, 'sd_matrix': False}, segs)
    quoted = [q for q in re.findall(r"'([^']*)'", g) if 'between' in q]
    assert len(quoted) == 7, 'the crop path, four split-pane origins and two layout switches'
    assert all(q.count('between') == 80 and _depth(q) <= ROOM for q in quoted)
    (tx, ty), (bx, by) = sc.split_exprs(segs)
    assert _value(bx, n=25 * 2 + 3) == 704 and _value(bx, n=3) == 0 and _value(tx, n=25 * 2 + 3) == 0


def test_music_ducking_under_a_long_programme_of_dialogue_stays_shallow():
    windows = [(k * 1.5, k * 1.5 + 0.8) for k in range(300)]
    expr = pipeline._build_duck_volume_expr(windows, -15)
    assert _depth(expr) <= ROOM
    gain = 10 ** (-15 / 20)
    assert abs(_value(expr, t=150.4) - gain) < 1e-4, 'ducked inside a window'
    assert abs(_value(expr, t=1000.0) - 1.0) < 1e-9, 'and not after the last one'
    few = pipeline._build_duck_volume_expr(windows[:3], -15)
    assert few.count('clip(') == 6 and ')+clip(' in few, 'a handful of windows is still written as a plain sum'
