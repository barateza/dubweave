import os
import sys
from pathlib import Path

# Ensure project root is on path so tests can import src package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.ingest import _build_yt_download_profiles, _cookie_opts


def test_cookie_opts_prefers_cookiefile(tmp_path):
    logs: list[str] = []
    cookie_file = tmp_path / "cookies.txt"
    cookie_file.write_text("# Netscape HTTP Cookie File\n", encoding="utf-8")

    opts = _cookie_opts(browser="edge", cookies_file=str(cookie_file), logs=logs)

    assert "cookiefile" in opts
    assert "cookiesfrombrowser" not in opts
    assert not logs


def test_cookie_opts_falls_back_to_browser_when_file_missing():
    logs: list[str] = []

    opts = _cookie_opts(browser="edge", cookies_file="missing-file.txt", logs=logs)

    assert opts.get("cookiesfrombrowser") == ("edge", None, None, None)
    assert any("cookies.txt not found" in line for line in logs)


def test_cookie_opts_returns_empty_for_none_mode_without_file():
    logs: list[str] = []

    opts = _cookie_opts(browser="none", cookies_file=None, logs=logs)

    assert opts == {}
    assert not logs


def test_profiles_include_expected_fallbacks_with_aria2c_and_cookies(tmp_path):
    job_dir = Path(tmp_path)
    profiles = _build_yt_download_profiles(
        job_dir=job_dir,
        cookie_opts={"cookiesfrombrowser": ("edge", None, None, None)},
        has_aria2c=True,
        has_deno=True,
    )

    names = [name for name, _ in profiles]
    assert names == [
        "aria2c",
        "builtin-http",
        "youtube-client-fallback",
        "youtube-client-ipv4",
        "anonymous-ipv4",
    ]

    aria2_opts = profiles[0][1]
    assert aria2_opts.get("external_downloader") == "aria2c"
    assert aria2_opts.get("remote_components") == ["ejs:github"]


def test_profiles_without_aria2c_or_cookies(tmp_path):
    job_dir = Path(tmp_path)
    profiles = _build_yt_download_profiles(
        job_dir=job_dir,
        cookie_opts={},
        has_aria2c=False,
        has_deno=False,
    )

    names = [name for name, _ in profiles]
    assert names == ["builtin-http", "youtube-client-fallback", "youtube-client-ipv4"]

    first_opts = profiles[0][1]
    assert "external_downloader" not in first_opts
    assert "remote_components" not in first_opts
