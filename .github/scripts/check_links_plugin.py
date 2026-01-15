# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import functools
import os
import time


def _parse_timeout(value, default):
    if value is None or value == "":
        return default
    if "," in value:
        parts = value.split(",", 1)
        return (float(parts[0]), float(parts[1]))
    return float(value)


def _get_int_env(name, default):
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return max(value, 0)


def pytest_configure(config):
    if not getattr(config.option, "check_links", False):
        return

    try:
        import pytest_check_links.plugin as pcl
    except Exception:
        return

    if getattr(pcl, "_rfs_link_patch", False):
        return
    pcl._rfs_link_patch = True

    retries_default = _get_int_env("CHECK_LINKS_RETRIES", 3)
    timeout_default = _parse_timeout(os.getenv("CHECK_LINKS_TIMEOUT"), 10.0)
    retry_after_max = os.getenv("CHECK_LINKS_RETRY_AFTER_MAX")

    if retry_after_max is not None:
        try:
            retry_after_max = float(retry_after_max)
        except ValueError:
            retry_after_max = None

    orig_fetch = pcl.LinkItem.fetch_with_retries
    orig_sleep = pcl.LinkItem.sleep

    def _sleep(self, headers):
        if retry_after_max is None:
            return orig_sleep(self, headers)
        if headers is None:
            return False
        header = headers.get("Retry-After")
        if header is None:
            return False
        if header == "1m0s":
            sleep_time = 60.0
        else:
            try:
                sleep_time = float(header)
            except ValueError:
                sleep_time = 10.0
        if retry_after_max <= 0:
            return False
        time.sleep(min(sleep_time, retry_after_max))
        return True

    def _fetch_with_retries(self, url, retries=None):
        effective_retries = retries_default if retries is None else retries
        session = self.parent.requests_session
        if session is None:
            return orig_fetch(self, url, retries=effective_retries)
        orig_get = session.get
        session.get = functools.partial(orig_get, timeout=timeout_default)
        try:
            return orig_fetch(self, url, retries=effective_retries)
        finally:
            session.get = orig_get

    pcl.LinkItem.sleep = _sleep
    pcl.LinkItem.fetch_with_retries = _fetch_with_retries
