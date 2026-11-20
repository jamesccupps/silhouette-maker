"""The local server, with the cut-out model replaced by a drawn mask."""
import io
import zipfile

import numpy as np
import pytest
from PIL import Image

import app as server
import engine
from test_engine import bust

BASE = "http://127.0.0.1:8777"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DATA", tmp_path / "data")
    server._cache.clear()

    def fake_segment(img, quality="best"):
        return bust(img.width, img.height), engine.MODELS["fast"]

    monkeypatch.setattr(engine, "segment", fake_segment)
    return server.app.test_client()


def photo(w=600, h=800, name="Ava.jpg"):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 200, 200)).save(buf, "JPEG")
    buf.seek(0)
    return {"photo": (buf, name)}


def add(client, **kw):
    r = client.post("/api/items", base_url=BASE, data={**photo(**kw), "quality": "fast"},
                    content_type="multipart/form-data")
    assert r.status_code == 200
    return r.get_json()


def test_only_this_computers_page_may_talk_to_it(client):
    assert client.get("/api/items", base_url=BASE).status_code == 200
    assert client.get("/api/items", base_url="http://evil.example").status_code == 403
    assert client.get("/api/items", base_url="http://127.0.0.1:9999").status_code == 403
    assert client.get("/api/items", base_url=BASE, headers={"Origin": "http://evil.example"}).status_code == 403
    assert client.get("/api/items", base_url=BASE, headers={"Origin": BASE}).status_code == 200


def test_add_edit_export_delete(client):
    it = add(client)
    assert it["name"] == "Ava" and it["outline"]["height_in"] == 2.25 and it["note"] == ""
    url = f"/api/items/{it['id']}"

    r = client.patch(url, base_url=BASE, json={"name": "Ava B", "done": True,
                                               "settings": {"height_in": 3, "flip": True, "tilt_deg": 99}})
    got = r.get_json()
    assert got["name"] == "Ava B" and got["done"] is True
    assert got["settings"]["tilt_deg"] == 45 and got["outline"]["height_in"] == 3

    svg = client.get(url + "/export.svg?border=circle&orn=8", base_url=BASE)
    assert svg.status_code == 200 and b"<circle" in svg.data and b'width="8.0000in"' in svg.data
    assert "Ava B.svg" in svg.headers["Content-Disposition"]
    png = client.get(url + "/export.png", base_url=BASE)
    assert Image.open(io.BytesIO(png.data)).mode == "RGBA"
    assert client.get(url + "/thumb", base_url=BASE).status_code == 200

    assert client.delete(url, base_url=BASE).status_code == 200
    assert client.get("/api/items", base_url=BASE).get_json() == []
    assert client.get(url + "/photo", base_url=BASE).status_code == 404


def test_best_falling_back_to_fast_is_reported(client):
    r = client.post("/api/items", base_url=BASE, data={**photo(), "quality": "best"},
                    content_type="multipart/form-data")
    assert "fast one was used" in r.get_json()["note"]


def test_a_file_that_is_not_a_photo_is_refused(client):
    r = client.post("/api/items", base_url=BASE, data={"photo": (io.BytesIO(b"not a picture"), "x.jpg")},
                    content_type="multipart/form-data")
    assert r.status_code == 400 and "photo" in r.get_json()["error"]


def test_sideways_photo_is_stood_up_and_can_be_turned(client, monkeypatch):
    def sideways(img, quality="best"):  # the body runs off the left edge
        return np.ascontiguousarray(np.rot90(bust(img.height, img.width), 3)), engine.MODELS["fast"]

    monkeypatch.setattr(engine, "segment", sideways)
    it = add(client, w=800, h=600)
    assert (it["w"], it["h"]) == (600, 800)  # turned upright on the way in

    r = client.post(f"/api/items/{it['id']}/rotate", base_url=BASE, json={"turns": 1}).get_json()
    assert (r["w"], r["h"], r["rev"]) == (800, 600, 1)
    saved = Image.open(server.DATA / it["id"] / "photo.jpg")
    assert saved.size == (800, 600)


def test_reset_goes_back_to_the_first_cut_out(client):
    it = add(client)
    url = f"/api/items/{it['id']}"
    start = (server.DATA / it["id"] / "mask.png").read_bytes()

    blank = io.BytesIO()
    Image.new("L", (it["w"], it["h"]), 0).save(blank, "PNG")  # everything erased
    assert client.put(url + "/mask", base_url=BASE, data=blank.getvalue()).get_json()["bbox"] is None
    client.patch(url, base_url=BASE, json={"name": "Kept", "settings": {"height_in": 3, "tilt_deg": 12, "flip": True}})

    r = client.post(url + "/reset", base_url=BASE, json={}).get_json()
    assert (server.DATA / it["id"] / "mask.png").read_bytes() == start
    assert r["outline"]["bbox"] and r["name"] == "Kept"
    s = r["settings"]
    assert (s["tilt_deg"], s["flip"], s["height_in"]) == (0.0, False, 3)  # the size is kept


def test_reset_follows_a_turn_and_works_for_older_photos(client):
    it = add(client)
    d = server.DATA / it["id"]
    client.post(f"/api/items/{it['id']}/rotate", base_url=BASE, json={"turns": 1})
    assert Image.open(d / "start.png").size == Image.open(d / "photo.jpg").size == (it["h"], it["w"])

    (d / "start.png").unlink()  # a photo added before the starting copy was kept
    r = client.post(f"/api/items/{it['id']}/reset", base_url=BASE, json={"quality": "fast"})
    assert r.status_code == 200 and (d / "start.png").exists()


def test_zip_and_sheet_take_the_chosen_photos(client):
    a, b, c = add(client), add(client), add(client, name="Ben.jpg")
    z = client.get(f"/api/svgs.zip?ids={a['id']},{b['id']},{c['id']}", base_url=BASE)
    names = sorted(zipfile.ZipFile(io.BytesIO(z.data)).namelist())
    assert names == ["Ava (2).svg", "Ava.svg", "Ben.svg"]
    sheet = client.get(f"/api/sheet?ids={a['id']},{c['id']}", base_url=BASE)
    assert sheet.data.count(b"<path") == 2
    assert client.get("/api/sheet", base_url=BASE).data.count(b"<path") == 3
    assert client.get("/api/sheet?ids=000000000000", base_url=BASE).status_code == 404
