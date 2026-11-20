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
    assert svg.status_code == 200 and b"<ellipse" in svg.data and b'width="8.0000in"' in svg.data
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


def test_nonsense_numbers_never_reach_the_settings(client):
    it = add(client)
    url = f"/api/items/{it['id']}"
    r = client.patch(url, base_url=BASE, json={"settings": {
        "height_in": "NaN", "smooth": "lots", "tilt_deg": 1e999, "detail_mm": None,
        "cut": [[0.1, "x"], [float("inf"), 0.9], [0.9, 0.8]]}})
    s = r.get_json()["settings"]
    assert (s["height_in"], s["smooth"], s["tilt_deg"], s["detail_mm"]) == (2.25, 3.0, 0.0, 0.5)
    assert all(-2 <= v <= 3 for point in s["cut"] for v in point)
    assert client.patch(url, base_url=BASE, json=["not", "an", "object"]).status_code == 400
    assert client.get(url + "/export.svg?space=abc&orn=nan", base_url=BASE).status_code == 200
    assert client.get("/api/sheet?w=wide", base_url=BASE).status_code == 200
    r = client.post("/api/items", base_url=BASE, data={**photo(), "quality": "fast", "height_in": "tall"},
                    content_type="multipart/form-data")
    assert r.get_json()["settings"]["height_in"] == 2.25


def test_a_mask_that_is_not_a_picture_is_refused(client):
    it = add(client)
    r = client.put(f"/api/items/{it['id']}/mask", base_url=BASE, data=b"definitely not a png")
    assert r.status_code == 400
    assert client.get("/api/items", base_url=BASE).get_json()[0]["outline"]["bbox"]  # untouched


def test_outline_is_saved_so_reopening_does_not_trace_again(client, monkeypatch):
    it = add(client)
    assert (server.DATA / it["id"] / "outline.json").exists()
    server._cache.clear()
    server._outlines.clear()  # as after closing and starting the app again

    def boom(*a, **k):
        raise AssertionError("traced again")

    monkeypatch.setattr(engine, "outline", boom)
    again = client.get("/api/items", base_url=BASE).get_json()[0]
    assert again["outline"] == it["outline"]

    monkeypatch.setattr(engine, "REV", engine.REV + 1)  # new outline code: saved ones are stale
    server._outlines.clear()
    traced = []
    monkeypatch.setattr(engine, "outline", lambda m, s: traced.append(1) or {"loops": [], "bbox": None, "final": m})
    client.get("/api/items", base_url=BASE)
    assert traced == [1]


def test_pages_on_other_sites_cannot_call_the_api(client):
    it = add(client)
    for site in ("cross-site", "same-site"):
        r = client.get("/api/items", base_url=BASE, headers={"Sec-Fetch-Site": site})
        assert r.status_code == 403
        r = client.delete(f"/api/items/{it['id']}", base_url=BASE, headers={"Sec-Fetch-Site": site})
        assert r.status_code == 403
    ok = client.get("/api/items", base_url=BASE, headers={"Sec-Fetch-Site": "same-origin"})
    assert ok.status_code == 200 and len(ok.get_json()) == 1
    assert ok.headers["X-Content-Type-Options"] == "nosniff"
    assert "frame-ancestors 'none'" in ok.headers["Content-Security-Policy"]
    # a link to the page itself from elsewhere still opens it
    assert client.get("/", base_url=BASE, headers={"Sec-Fetch-Site": "cross-site"}).status_code == 200


def test_photos_keep_the_order_they_were_added(client):
    names = [add(client, name=f"{n}.jpg")["name"] for n in ("c", "a", "b")]
    got = client.get("/api/items", base_url=BASE).get_json()
    assert [g["name"] for g in got] == names and got[0]["added"] <= got[1]["added"] <= got[2]["added"]


def test_one_height_for_everyone(client):
    a, b = add(client), add(client, name="Ben.jpg")
    client.patch(f"/api/items/{a['id']}", base_url=BASE, json={"settings": {"height_in": 1.5, "flip": True}})
    got = client.post("/api/height", base_url=BASE, json={"height_in": 3.25}).get_json()
    assert [g["settings"]["height_in"] for g in got] == [3.25, 3.25]
    assert [g["outline"]["height_in"] for g in got] == [3.25, 3.25]
    assert got[0]["settings"]["flip"] is True  # nothing else is touched
    saved = client.get("/api/items", base_url=BASE).get_json()
    assert all(s["settings"]["height_in"] == 3.25 for s in saved)
    for bad in ({"height_in": "big"}, {}, None):
        assert client.post("/api/height", base_url=BASE, json=bad).status_code == 400


def test_sheet_is_laid_out_for_the_chosen_mat(client):
    for i in range(12):
        add(client, name=f"kid{i}.jpg")

    def sheet(query):
        r = client.get("/api/sheet" + query, base_url=BASE)
        if r.mimetype == "application/zip":
            names = zipfile.ZipFile(io.BytesIO(r.data)).namelist()
            return len(names), zipfile.ZipFile(io.BytesIO(r.data)).read(names[0]).decode()
        return 1, r.data.decode()

    n_joy, joy = sheet("?w=4.25&h=11.75")
    n_long, long_mat = sheet("?w=11.5&h=23.5")
    assert 'width="4.2500in" height="11.7500in"' in joy and n_joy > 1
    assert 'width="11.5000in" height="23.5000in"' in long_mat and n_long == 1
    assert long_mat.count("<path") == 12


def test_proof_sheet_shows_every_child_at_real_size(client):
    add(client, name="Ava <b>.jpg")
    add(client, name="Ben.jpg")
    r = client.get("/proof?orn=3", base_url=BASE)
    page = r.data.decode()
    assert r.status_code == 200 and page.count("<figure>") == 2
    assert 'width="3.0000in" height="3.0000in"' in page and "stroke-dasharray" in page
    assert "Ava &lt;b&gt;" in page and "<b>" not in page  # names cannot inject markup
    assert "2.25 in tall" in page and "/static/proof.js" in page and "<script>" not in page
    assert client.get("/proof?ids=000000000000", base_url=BASE).status_code == 404


def test_deleting_the_photo_keeps_the_silhouette(client):
    it = add(client)
    url = f"/api/items/{it['id']}"
    d = server.DATA / it["id"]
    client.get(url + "/thumb", base_url=BASE)
    r = client.post(url + "/forget", base_url=BASE).get_json()
    assert r["photo"] is False and r["outline"] == it["outline"]
    assert not (d / "photo.jpg").exists() and not (d / "thumb.jpg").exists() and (d / "mask.png").exists()
    assert client.get(url + "/photo", base_url=BASE).status_code == 404
    assert client.get(url + "/thumb", base_url=BASE).status_code == 404
    assert client.get(url + "/export.svg", base_url=BASE).status_code == 200  # still exports
    assert client.post(url + "/redo", base_url=BASE, json={}).status_code == 409
    assert client.post(url + "/reset", base_url=BASE, json={}).status_code == 200  # the starting shape is kept
    turned = client.post(url + "/rotate", base_url=BASE, json={"turns": 1}).get_json()
    assert (turned["w"], turned["h"]) == (it["h"], it["w"])
    (d / "start.png").unlink()
    assert client.post(url + "/reset", base_url=BASE, json={}).status_code == 409


def test_backup_and_restore_round_trip(client, tmp_path, monkeypatch):
    a, b = add(client), add(client, name="Ben.jpg")
    client.patch(f"/api/items/{a['id']}", base_url=BASE, json={"name": "Ava B", "done": True,
                                                               "settings": {"height_in": 3, "flip": True}})
    client.post(f"/api/items/{b['id']}/forget", base_url=BASE)
    before = client.get("/api/items", base_url=BASE).get_json()
    zipped = client.get("/api/backup", base_url=BASE)
    assert zipped.mimetype == "application/zip" and "silhouette-maker-backup-" in zipped.headers["Content-Disposition"]

    def restore(data):
        return client.post("/api/restore", base_url=BASE, data={"backup": (io.BytesIO(data), "backup.zip")},
                           content_type="multipart/form-data")

    assert restore(zipped.data).get_json() == {"added": 0, "skipped": 2}  # nothing is overwritten

    monkeypatch.setattr(server, "DATA", tmp_path / "another-computer")
    server._cache.clear()
    server._outlines.clear()
    assert restore(zipped.data).get_json() == {"added": 2, "skipped": 0}
    after = client.get("/api/items", base_url=BASE).get_json()
    strip = lambda items: [{k: v for k, v in it.items()} for it in items]
    assert strip(after) == strip(before)
    assert not list((tmp_path / "another-computer").glob(".restore-*"))


def test_restore_ignores_anything_that_is_not_ours(client):
    assert client.post("/api/restore", base_url=BASE, data={"backup": (io.BytesIO(b"not a zip"), "x.zip")},
                       content_type="multipart/form-data").status_code == 400
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("../../evil.txt", "x")
        z.writestr("aaaaaaaaaaaa/../../evil2.txt", "x")
        z.writestr("bbbbbbbbbbbb/meta.json", "{}")  # no mask
        z.writestr("cccccccccccc/meta.json", '{"name": "x", "settings": {}}')
        z.writestr("cccccccccccc/mask.png", b"not a picture")
        z.writestr("readme.txt", "hello")
    r = client.post("/api/restore", base_url=BASE, data={"backup": (io.BytesIO(buf.getvalue()), "x.zip")},
                    content_type="multipart/form-data")
    assert r.get_json() == {"added": 0, "skipped": 0}
    assert client.get("/api/items", base_url=BASE).get_json() == []
    assert not (server.DATA.parent / "evil.txt").exists() and not (server.DATA.parent.parent / "evil.txt").exists()


def test_fonts_and_lettering_in_exports(client):
    fonts = client.get("/api/fonts", base_url=BASE).get_json()
    if not fonts:
        pytest.skip("no usable font on this computer")
    it = add(client)
    url = f"/api/items/{it['id']}/export.svg"
    plain = client.get(url, base_url=BASE).data
    named = client.get(url + f"?text=both&line2=2026&font={fonts[0]['key']}&tsize=0.3", base_url=BASE).data
    assert named.count(b"<path") == plain.count(b"<path") + 1
    assert client.get(url + "?text=nonsense&font=nope&tsize=huge", base_url=BASE).data == plain
    png = client.get(f"/api/items/{it['id']}/export.png?text=name", base_url=BASE)
    assert png.status_code == 200


def test_camera_file_names_are_not_names(client):
    names = {f: add(client, name=f)["name"] for f in (
        "IMG_20261119_190204.jpg", "PXL_20261119_233031583.jpg", "DSC01234.jpg", "P1040123.jpg",
        "20261119_101500.jpg", "image.jpg", "Ava.jpg", "Paul.jpg", "Anna-Marie 2.jpg", "Imogen.jpg")}
    assert [k for k, v in names.items() if v == ""] == [
        "IMG_20261119_190204.jpg", "PXL_20261119_233031583.jpg", "DSC01234.jpg", "P1040123.jpg",
        "20261119_101500.jpg", "image.jpg"]
    assert (names["Ava.jpg"], names["Paul.jpg"], names["Anna-Marie 2.jpg"], names["Imogen.jpg"]) == (
        "Ava", "Paul", "Anna-Marie 2", "Imogen")
    unnamed = client.get("/api/items", base_url=BASE).get_json()[0]
    svg = client.get(f"/api/items/{unnamed['id']}/export.svg", base_url=BASE)
    assert "silhouette.svg" in svg.headers["Content-Disposition"]
    assert "Unnamed" in client.get("/proof", base_url=BASE).data.decode()


def test_preview_is_drawn_like_the_files(client):
    it = add(client)
    url = f"/api/items/{it['id']}/preview"
    r = client.get(url + "?shape=circle&orn=3&fg=%23224466&bg=%23ffffff", base_url=BASE).get_json()
    assert r["fits"] and "<ellipse" in r["html"] and 'fill="#224466"' in r["html"] and 'fill="#ffffff"' in r["html"]
    x0, y0, w, h = map(float, r["viewBox"].split())
    assert (w, h) == (pytest.approx(3.24 * 72), pytest.approx(3.24 * 72))
    assert not client.get(url + "?shape=circle&orn=1", base_url=BASE).get_json()["fits"]
    heart = client.get(url + "?shape=heart&orn=4", base_url=BASE).get_json()
    assert "<polygon" in heart["html"]
    bad = client.get(url + "?shape=dodecahedron&fg=red&bg=javascript:alert(1)", base_url=BASE).get_json()
    assert "<ellipse" in bad["html"] and "javascript" not in bad["html"] and "red" not in bad["html"]
    if client.get("/api/fonts", base_url=BASE).get_json():
        named = client.get(url + "?name_on=1&name=Ava%20Typed%20Just%20Now&line2=2026", base_url=BASE).get_json()
        assert named["html"].count("<path") == r["html"].count("<path") + 1  # the unsaved name is used

    blank = io.BytesIO()
    Image.new("L", (it["w"], it["h"]), 0).save(blank, "PNG")
    client.put(f"/api/items/{it['id']}/mask", base_url=BASE, data=blank.getvalue())
    empty = client.get(url, base_url=BASE).get_json()
    assert empty["fits"] and "<path" not in empty["html"]  # nothing to cut: just the ornament
