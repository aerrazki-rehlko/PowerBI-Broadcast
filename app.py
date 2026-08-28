import asyncio
import ipaddress
import json
import logging
import os
import secrets
import sys
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.json"
OUTPUT_DIR = BASE_DIR / "output"
PROFILE_DIR = BASE_DIR / "browser-profile"
LOG_DIR = BASE_DIR / "logs"

OUTPUT_DIR.mkdir(exist_ok=True)
PROFILE_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

LIVE_IMAGE = OUTPUT_DIR / "dashboard.png"
TEMP_IMAGE = OUTPUT_DIR / ".dashboard.tmp.png"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("powerbi-signage")

app = FastAPI(
    title="Power BI Signage POC",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)

CONFIG = None
PARSED_NETWORKS = []
PLAYWRIGHT = None
BROWSER_CONTEXT = None

STATUS = {
    "state": "starting",
    "last_capture": None,
    "last_error": None,
}


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    url = cfg.get("powerbi_url", "").strip()
    if not url.startswith("https://app.powerbi.com/"):
        raise RuntimeError(
            "powerbi_url must be a normal secure https://app.powerbi.com/ report URL."
        )

    token = cfg.get("token", "").strip()
    if len(token) < 32:
        raise RuntimeError(
            "token must be at least 32 characters. Generate one with "
            "python -c \"import secrets; print(secrets.token_urlsafe(48))\""
        )

    networks = cfg.get("allowed_networks", [])
    if not networks:
        raise RuntimeError("allowed_networks is empty. Refusing to start.")

    parsed = [ipaddress.ip_network(cidr, strict=False) for cidr in networks]

    interval = int(cfg.get("capture_interval_minutes", 10))
    if interval < 1:
        raise RuntimeError("capture_interval_minutes must be >= 1")

    cfg["_parsed_networks"] = parsed
    return cfg


def client_ip_allowed(ip_text: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_text)
    except ValueError:
        return False

    if ip.is_loopback:
        return True

    return any(ip in network for network in PARSED_NETWORKS)


def require_internal(request: Request):
    if request.client is None:
        raise HTTPException(status_code=403, detail="Forbidden")

    ip = request.client.host
    if not client_ip_allowed(ip):
        log.warning("Blocked non-approved source IP: %s", ip)
        raise HTTPException(status_code=403, detail="Forbidden")

    return ip


def require_token(token: str):
    if not secrets.compare_digest(token or "", CONFIG["token"]):
        raise HTTPException(status_code=403, detail="Forbidden")


def looks_like_login_page(url: str) -> bool:
    value = url.lower()
    indicators = (
        "login.microsoftonline.com",
        "login.live.com",
        "/signin",
    )
    return any(x in value for x in indicators)


async def capture_dashboard():
    page = None

    try:
        STATUS["state"] = "loading"
        STATUS["last_error"] = None

        page = await BROWSER_CONTEXT.new_page()

        await page.goto(
            CONFIG["powerbi_url"],
            wait_until="domcontentloaded",
            timeout=90_000,
        )

        await page.wait_for_timeout(
            int(CONFIG.get("render_wait_seconds", 15)) * 1000
        )

        current_url = page.url

        if looks_like_login_page(current_url):
            STATUS["state"] = "authentication_required"
            STATUS["last_error"] = "Microsoft authentication required."
            log.warning("Authentication required; keeping last good screenshot.")
            return

        if "app.powerbi.com" not in current_url.lower():
            STATUS["state"] = "unexpected_page"
            STATUS["last_error"] = f"Unexpected URL: {current_url}"
            log.warning("Unexpected page; keeping last good screenshot.")
            return

        body_text = await page.locator("body").inner_text(timeout=10_000)
        if len(body_text.strip()) < 20:
            STATUS["state"] = "invalid_page"
            STATUS["last_error"] = "Power BI page did not appear to render."
            log.warning("Page validation failed; keeping last good screenshot.")
            return

        await page.screenshot(
            path=str(TEMP_IMAGE),
            full_page=False,
        )

        os.replace(TEMP_IMAGE, LIVE_IMAGE)

        STATUS["state"] = "ok"
        STATUS["last_capture"] = time.strftime("%Y-%m-%d %H:%M:%S")
        STATUS["last_error"] = None

        log.info("Dashboard updated successfully.")

    except PlaywrightTimeoutError:
        STATUS["state"] = "error"
        STATUS["last_error"] = "Timed out while loading Power BI."
        log.exception("Power BI capture timed out; keeping last good screenshot.")

    except Exception as exc:
        STATUS["state"] = "error"
        STATUS["last_error"] = str(exc)
        log.exception("Capture failed; keeping last good screenshot.")

    finally:
        if TEMP_IMAGE.exists():
            try:
                TEMP_IMAGE.unlink()
            except OSError:
                pass

        if page is not None:
            await page.close()


async def capture_loop():
    while True:
        await capture_dashboard()
        await asyncio.sleep(CONFIG["capture_interval_minutes"] * 60)


@app.get("/dashboard")
async def dashboard(request: Request, token: str = ""):
    require_internal(request)
    require_token(token)

    if not LIVE_IMAGE.exists():
        raise HTTPException(
            status_code=503,
            detail="No valid dashboard screenshot is available yet."
        )

    return FileResponse(
        LIVE_IMAGE,
        media_type="image/png",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )


@app.get("/health")
async def health(request: Request):
    require_internal(request)

    return JSONResponse(
        {
            "state": STATUS["state"],
            "last_capture": STATUS["last_capture"],
            "last_error": STATUS["last_error"],
            "image_available": LIVE_IMAGE.exists(),
        }
    )


@app.on_event("startup")
async def startup():
    global CONFIG, PARSED_NETWORKS, PLAYWRIGHT, BROWSER_CONTEXT

    CONFIG = load_config()
    PARSED_NETWORKS = CONFIG["_parsed_networks"]

    PLAYWRIGHT = await async_playwright().start()

    BROWSER_CONTEXT = await PLAYWRIGHT.chromium.launch_persistent_context(
        user_data_dir=str(PROFILE_DIR),
        headless=bool(CONFIG.get("headless", False)),
        viewport={
            "width": int(CONFIG.get("viewport_width", 1920)),
            "height": int(CONFIG.get("viewport_height", 1080)),
        },
    )

    asyncio.create_task(capture_loop())
    log.info("Power BI Signage POC started.")


@app.on_event("shutdown")
async def shutdown():
    if BROWSER_CONTEXT is not None:
        await BROWSER_CONTEXT.close()

    if PLAYWRIGHT is not None:
        await PLAYWRIGHT.stop()


if __name__ == "__main__":
    if not CONFIG_PATH.exists():
        raise SystemExit(
            "Missing config.json. Copy config.example.json to config.json first."
        )

    cfg = load_config()

    uvicorn.run(
        app,
        host=cfg.get("bind_host", "0.0.0.0"),
        port=int(cfg.get("port", 8000)),
        access_log=False,
    )
