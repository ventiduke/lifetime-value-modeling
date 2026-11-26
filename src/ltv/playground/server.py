"""The playground's HTTP layer: one page and four JSON endpoints, on 127.0.0.1 only.

Standard library only, so there is nothing to install beyond the package itself. It is a
tool for one person on their own machine: there is no authentication, and it refuses to
bind to anything but the loopback address.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ltv.playground import core

PAGE = Path(__file__).with_name("index.html")
MAX_BODY = 256 * 1024


def make_handler(bundle: core.Bundle):
    info = json.dumps(core.info(bundle)).encode()

    class Handler(BaseHTTPRequestHandler):
        server_version = "ltv-playground"

        def log_message(self, fmt, *args):                 # quiet: one line per request is noise here
            pass

        def _send(self, code: int, body: bytes, kind: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind + "; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html")
            elif self.path == "/engine.js":
                self._send(200, PAGE.with_name("engine.js").read_bytes(), "text/javascript")
            elif self.path == "/mode.json":
                self._send(200, b'{"mode": "server"}')
            elif self.path == "/api/info":
                self._send(200, info)
            else:
                self._send(404, b'{"error": "not found"}')

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length") or 0)
                if n > MAX_BODY:
                    raise ValueError("request too large")
                q = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/api/forecast":
                    out = core.forecast(bundle, q["hours"], q.get("spend"))
                elif self.path == "/api/timeline":
                    out = core.timeline(bundle, q["hours"], q.get("spend"), min(int(q.get("weeks", 104)), 520),
                                        max(int(q.get("step", 1)), 1))
                elif self.path == "/api/simulate":
                    out = core.simulate(bundle, q.get("world", "rhythm"), q.get("want", "any"), q.get("seed"))
                else:
                    return self._send(404, b'{"error": "not found"}')
                self._send(200, json.dumps(out).encode())
            except (ValueError, KeyError, TypeError) as exc:      # a bad request, said plainly to the page
                self._send(400, json.dumps({"error": str(exc)}).encode())
            except Exception as exc:  # noqa: BLE001
                self._send(500, json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode())

    return Handler


def serve(bundle: core.Bundle, port: int = 8765, open_browser: bool = True) -> None:
    import torch
    torch.set_num_threads(1)                     # one player at a time: threads only add latency
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(bundle))
    url = f"http://127.0.0.1:{httpd.server_address[1]}/"
    print(f"LTV playground: {url}   (models: {', '.join(bundle.scorers)}; fit {bundle.fit}; Ctrl+C to stop)")
    if bundle.meta.get("synthetic"):
        print("These models were trained on SYNTHETIC players. They show how the models behave, nothing more.")
    if open_browser:
        import webbrowser
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
