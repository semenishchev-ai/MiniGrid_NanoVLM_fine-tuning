from collections import deque

DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def _passable(grid, x, y, width, height):
    if x < 0 or y < 0 or x >= width or y >= height:
        return False
    cell = grid.get(x, y)
    if cell is None:
        return True
    return cell.type in ("goal", "floor")


def _find_goal(grid, width, height):
    for x in range(width):
        for y in range(height):
            cell = grid.get(x, y)
            if cell is not None and cell.type == "goal":
                return (x, y)
    return None


def _shortest_path(env, start, goal):
    width, height = env.width, env.height
    if start == goal:
        return [start]
    prev = {start: None}
    queue = deque([start])
    while queue:
        pos = queue.popleft()
        if pos == goal:
            path = []
            while pos is not None:
                path.append(pos)
                pos = prev[pos]
            path.reverse()
            return path
        x, y = pos
        for dx, dy in DIRS:
            nxt = (x + dx, y + dy)
            if nxt in prev:
                continue
            if not _passable(env.grid, *nxt, width, height):
                continue
            prev[nxt] = pos
            queue.append(nxt)
    return None


def _dir_between(src, dst):
    sx, sy = src
    dx, dy = dst[0] - sx, dst[1] - sy
    for d, (vx, vy) in enumerate(DIRS):
        if (dx, dy) == (vx, vy):
            return d
    return None


def _turn_action(agent_dir, target_dir):
    if agent_dir == target_dir:
        return 2
    diff = (target_dir - agent_dir) % 4
    if diff == 1:
        return 1
    return 0


class ExpertPolicy:
    def act(self, env):
        u = env.unwrapped
        start = tuple(u.agent_pos)
        goal = _find_goal(u.grid, u.width, u.height)
        if goal is None:
            return 2
        path = _shortest_path(u, start, goal)
        if path is None or len(path) < 2:
            return 2
        target_dir = _dir_between(start, path[1])
        if target_dir is None:
            return 2
        return _turn_action(u.agent_dir, target_dir)
