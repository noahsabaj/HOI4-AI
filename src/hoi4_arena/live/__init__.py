"""HOI4 Live: the games on a phone, as they are played and afterwards.

`hoi4-arena live --peer PAIRING` serves a page on 127.0.0.1 (`tailscale serve --bg
--https=8443 http://127.0.0.1:8765` publishes it to the tailnet alone) with:

- a stream per PC: the second PC's game window at 60 frames a second from its worker's
  own capture, menus and loading included, and this PC's while a recorder plays here
  (media.py);
- what is going on: the game's map, side, plan and start, how each side stands in the
  game, the scripted player's orders, and whether any runs are going at all, or since
  when none has (state.py), so a finished game never passes for a live one;
- the feed, like a stream's chat: the games narrated as they go, beside messages from
  watchers and from Claude (feed.py);
- each game as it looked live, kept small at 30 frames a second (archive.py), and
  replays of any other game played, made from its 5-a-second recording on request
  (replay.py);
- the record by map and plan, and training in progress.

Added to an iPhone's or iPad's home screen (Share, Add to Home Screen) the page opens full
screen with its own icon.
"""

from .app import LiveApp, draw_icon, install_app, watch
from .archive import Archive, archive_command, covering, pieces
from .feed import Feed, Flags, Narrator, say
from .media import (
    Follower,
    LocalView,
    PeerView,
    capture_command,
    clear_stream,
    hls_command,
    next_segment,
    read_json,
    read_shared,
    update_pending,
    view_command,
)
from .replay import Replays, replay_command
from .server import serve
from .state import History, game_card, live_game, live_games, result, station_of

__all__ = [
    "Archive", "Feed", "Flags", "Follower", "History", "LiveApp", "LocalView", "Narrator",
    "PeerView", "Replays", "archive_command", "capture_command", "covering", "pieces", "clear_stream", "draw_icon", "game_card", "hls_command",
    "install_app", "live_game", "live_games", "next_segment", "read_json", "read_shared",
    "replay_command", "result", "say", "serve", "station_of", "update_pending",
    "view_command", "watch",
]  # fmt: skip
