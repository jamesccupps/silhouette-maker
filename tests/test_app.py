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


def test_print_page_shows_every_child_at_real_size(client):
    add(client, name="Ava <b>.jpg")
    add(client, name="Ben.jpg")
    r = client.get("/print?orn=3", base_url=BASE)
    page = r.data.decode()
    assert r.status_code == 200 and page.count('<div class="sheet">') == 1
    assert 'width="8.5000in" height="11.0000in"' in page and "@page{size:8.5in 11.0in;margin:0}" in page
    assert "stroke-dasharray" in page  # the ornament, dashed
    assert f'width="{2 * 72:.2f}"' in page  # the 2 inch bar that proves the scale
    assert "Ava &lt;b&gt;" in page and "<b>.jpg" not in page and "Ava <b>" not in page  # names cannot inject markup
    assert "2.25 in tall" in page and "/static/print.js" in page and "<script>" not in page
    assert client.get("/print?ids=000000000000", base_url=BASE).status_code == 404
    assert client.get("/proof?orn=3", base_url=BASE).status_code == 200  # the old address still works


def test_print_page_choices(client):
    a = add(client)
    q = f"/print?ids={a['id']}&orn=3&name_on=1&font=serif"
    plain = client.get(q, base_url=BASE).data.decode()
    page = client.get(q + "&paper=a4&show=shape&ink=outline&names=0&copies=20", base_url=BASE).data.decode()
    assert 'width="8.2677in"' in page and "stroke-dasharray" not in page  # A4, no ornament
    assert 'fill="none" stroke="#000000"' in page and 'fill="none" stroke="#000000"' not in plain
    assert "in tall" not in page and "in tall" in plain
    assert page.count('<div class="sheet">') > plain.count('<div class="sheet">') == 1  # 20 copies need more pages
    assert f'name="ids" value="{a["id"]}"' in page and 'name="font" value="serif"' in page  # choices keep the rest
    assert '<option value="a4" selected>' in page


def _stl(data):
    import numpy as np
    n = int(np.frombuffer(data[80:84], "<u4")[0])
    assert len(data) == 84 + 50 * n
    return np.frombuffer(data[84:], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))["v"]


def test_3d_files_download(client):
    a = add(client)
    url = f"/api/items/{a['id']}/model"
    r = client.get(f"{url}.stl?what=silhouette&thick_mm=2.5", base_url=BASE)
    v = _stl(r.data)
    assert r.status_code == 200 and "Ava silhouette.stl" in r.headers["Content-Disposition"]
    size = v.reshape(-1, 3).max(0) - v.reshape(-1, 3).min(0)
    assert abs(size[1] - 2.25 * 25.4) < 0.3 and abs(size[2] - 2.5) < 1e-4  # millimetres, at the size set
    r = client.get(f"{url}.stl?what=ornament&orn=3&hang=none&base_mm=3&raise_mm=1.2", base_url=BASE)
    size = _stl(r.data).reshape(-1, 3).max(0) - _stl(r.data).reshape(-1, 3).min(0)
    assert abs(size[0] - 76.2) < 0.05 and abs(size[1] - 76.2) < 0.05 and abs(size[2] - 4.2) < 1e-4
    z = zipfile.ZipFile(io.BytesIO(client.get(f"{url}.zip?what=parts&orn=3&name_on=1", base_url=BASE).data))
    assert sorted(z.namelist()) == ["Ava ornament - base.stl", "Ava ornament - lettering.stl",
                                    "Ava ornament - silhouette.stl", "How to use these files.txt"]
    r = client.get(f"{url}.png?orn=3&fg=%23112233", base_url=BASE)
    assert r.status_code == 200 and r.data[:4] == b"\x89PNG"
    assert client.get(f"{url}.stl?what=nonsense", base_url=BASE).status_code == 404
    b = add(client, name="Ben.jpg")
    z = zipfile.ZipFile(io.BytesIO(client.get(f"/api/models.zip?ids={a['id']},{b['id']}", base_url=BASE).data))
    assert sorted(z.namelist()) == ["Ava ornament.stl", "Ava silhouette.stl", "Ben ornament.stl", "Ben silhouette.stl"]


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
    listing = client.get("/api/fonts", base_url=BASE).get_json()
    fonts = listing["fonts"]
    if not fonts:
        pytest.skip("no usable font on this computer")
    assert listing["default"] in {f["key"] for f in fonts} and set(listing["old"]) == {"serif", "sans", "bold", "script"}
    it = add(client)
    url = f"/api/items/{it['id']}/export.svg"
    plain = client.get(url, base_url=BASE).data
    named = client.get(url + f"?text=both&line2=2026&font={fonts[0]['key']}&tsize=0.3", base_url=BASE).data
    assert named.count(b"<path") == plain.count(b"<path") + 1
    assert client.get(url + "?text=nonsense&font=nope&tsize=huge", base_url=BASE).data == plain
    two = client.get(url + f"?name_on=1&line2=2026&font={fonts[0]['key']}&tsize=0.3&font2={fonts[-1]['key']}&tsize2=0.15",
                     base_url=BASE).data
    assert two.count(b"<path") == plain.count(b"<path") + 1 and (two != named or len(fonts) == 1)
    odd = client.get(url + "?name_on=1&font=../../etc/passwd&font2=%00&tsize2=nan", base_url=BASE)
    assert odd.status_code == 200 and odd.data.count(b"<path") == plain.count(b"<path") + 1  # falls back to a real font
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
    if client.get("/api/fonts", base_url=BASE).get_json()["fonts"]:
        named = client.get(url + "?name_on=1&name=Ava%20Typed%20Just%20Now&line2=2026", base_url=BASE).get_json()
        assert named["html"].count("<path") == r["html"].count("<path") + 1  # the unsaved name is used

    blank = io.BytesIO()
    Image.new("L", (it["w"], it["h"]), 0).save(blank, "PNG")
    client.put(f"/api/items/{it['id']}/mask", base_url=BASE, data=blank.getvalue())
    empty = client.get(url, base_url=BASE).get_json()
    assert empty["fits"] and "<path" not in empty["html"]  # nothing to cut: just the ornament


def test_hardware_report_and_real_timings(client, monkeypatch):
    monkeypatch.setattr(engine, "hardware", lambda: {"gpu": "Test Card 9000", "vram_gb": 12.0, "ram_gb": 32.0,
                                                    "gpu_runtime": True})
    r = client.get("/api/hardware", base_url=BASE).get_json()
    assert (r["recommend"], r["best"], r["secs"]) == ("best", "yes", {"best": None, "fast": None})
    assert "Test Card 9000" in r["why"]

    add(client)  # the stand-in model answers as the fast one: a timing is recorded
    r = client.get("/api/hardware", base_url=BASE).get_json()
    assert r["secs"]["fast"] is not None and r["secs"]["best"] is None and r["recommend"] == "best"

    client.post("/api/items", base_url=BASE, data={**photo(), "quality": "best"}, content_type="multipart/form-data")
    r = client.get("/api/hardware", base_url=BASE).get_json()  # best was asked for and could not run
    assert (r["recommend"], r["best"]) == ("fast", "no") and "could not run" in r["why"]
    assert len(client.get("/api/items", base_url=BASE).get_json()) == 2  # the timing file is not a photo


@pytest.mark.parametrize("hw, recommend, best", [
    ({"gpu": "Big", "vram_gb": 12.0, "ram_gb": 32, "gpu_runtime": True}, "best", "yes"),
    ({"gpu": "Mid", "vram_gb": 8.0, "ram_gb": 16, "gpu_runtime": True}, "best", "maybe"),
    ({"gpu": "Small", "vram_gb": 2.0, "ram_gb": 16, "gpu_runtime": True}, "fast", "no"),
    ({"gpu": None, "vram_gb": 0.0, "ram_gb": 8, "gpu_runtime": True}, "fast", "no"),
    ({"gpu": None, "vram_gb": 0.0, "ram_gb": 64, "gpu_runtime": False}, "fast", "maybe"),
    ({"gpu": None, "vram_gb": 0.0, "ram_gb": 8, "gpu_runtime": False}, "fast", "no"),
])
def test_advice_for_different_computers(hw, recommend, best):
    a = engine.advice(hw)
    assert (a["recommend"], a["best"]) == (recommend, best) and a["why"].endswith(".")
    assert engine.advice(hw, best_ran=True)["recommend"] == "best"  # it has worked here: that settles it
    assert engine.advice(hw, best_ran=True, best_failed=True)["recommend"] == "fast"


def test_this_computer_can_be_examined():
    hw = engine.hardware()
    assert set(hw) == {"gpu", "vram_gb", "ram_gb", "gpu_runtime", "runtime"} and hw["ram_gb"] > 0.5
    assert hw["runtime"] in ("nvidia", "directml", "cpu") and hw["gpu_runtime"] == (hw["runtime"] != "cpu")
    assert engine.cuda_works() in (True, False) and (engine.runtime() == "nvidia" or not engine.cuda_works())
    assert hw["gpu"] is None or isinstance(hw["gpu"], str)


@pytest.mark.parametrize("error, says", [
    (RuntimeError("bad allocation at C:\\secret\\path"), "ran out of memory"),
    (OSError("getaddrinfo failed for github.com"), "could not be downloaded"),
    (ValueError("something odd in C:\\secret\\path"), "Silhouette Maker window"),
])
def test_a_failed_cutout_gives_a_fixed_reason_not_the_error(client, monkeypatch, capsys, error, says):
    def fail(img, quality):
        raise error
    monkeypatch.setattr(server, "_cut_out", fail)
    buf = io.BytesIO()
    Image.new("RGB", (600, 800), "white").save(buf, "JPEG")
    buf.seek(0)
    r = client.post("/api/items", base_url=BASE, data={"photo": (buf, "a.jpg")},
                    content_type="multipart/form-data")
    assert r.status_code == 500
    msg = r.get_json()["error"]
    assert says in msg and "secret" not in msg and "github.com" not in msg
    assert str(error) in capsys.readouterr().err      # the detail is in the app's window


def test_a_picture_too_small_to_cut_out_is_refused(client):
    r = client.post("/api/items", base_url=BASE, data=photo(w=1, h=1), content_type="multipart/form-data")
    assert r.status_code == 400 and "too small" in r.get_json()["error"]
    r = client.post("/api/items", base_url=BASE, data=photo(w=900, h=150), content_type="multipart/form-data")
    assert r.status_code == 400
    assert client.get("/api/items", base_url=BASE).get_json() == []


def test_a_class_is_traced_once_not_at_every_download(client, monkeypatch):
    ids = [add(client, name=f"Kid{i}.jpg")["id"] for i in range(8)]  # more than the few bitmaps kept
    calls = []
    real = engine.outline
    monkeypatch.setattr(engine, "outline", lambda *a, **k: calls.append(1) or real(*a, **k))
    server._cache.clear()
    server._curves.clear()
    assert client.get("/api/svgs.zip", base_url=BASE).status_code == 200
    assert len(calls) == 8
    assert client.get("/api/svgs.zip", base_url=BASE).status_code == 200
    assert client.get("/print", base_url=BASE).status_code == 200
    assert len(calls) == 8                                  # nothing traced again
    client.patch(f"/api/items/{ids[0]}", base_url=BASE, json={"settings": {"flip": True}})
    assert client.get("/api/svgs.zip", base_url=BASE).status_code == 200
    assert len(calls) == 9                                  # only the child that changed
    server._cache.clear()                                   # the PNG needs the bitmap: traced again when it is gone
    assert client.get(f"/api/items/{ids[1]}/export.png", base_url=BASE).status_code == 200
    assert len(calls) == 10


def test_the_tab_icon_is_served(client):
    r = client.get("/favicon.ico", base_url=BASE)
    assert r.status_code == 200 and len(r.data) > 100


def test_last_backup_says_when_and_whether_anything_changed(client):
    assert client.get("/api/backup/last", base_url=BASE).get_json() == {"at": None, "photos": 0, "changed": False}
    it = add(client)
    assert client.get("/api/backup/last", base_url=BASE).get_json()["photos"] == 1
    assert client.get("/api/backup", base_url=BASE).status_code == 200
    r = client.get("/api/backup/last", base_url=BASE).get_json()
    assert r["at"] and not r["changed"]
    import os
    import time
    d = server.DATA / it["id"]
    os.utime(d / "meta.json", (time.time() + 5, time.time() + 5))   # a change after the backup
    assert client.get("/api/backup/last", base_url=BASE).get_json()["changed"]
    assert "backup.json" not in zipfile.ZipFile(io.BytesIO(client.get("/api/backup", base_url=BASE).data)).namelist()


def test_print_page_says_how_many_fit_and_wraps_its_pages(client):
    ids = [add(client, name=f"Kid{i}.jpg")["id"] for i in range(3)]
    page = client.get("/print?orn=3&copies=5", base_url=BASE).get_data(as_text=True)   # 15 ornaments: more than a page
    assert '<div id="sheets">' in page and "fit on a page" in page
    n = int(page.split(" fit on a page")[0].rsplit(" ", 1)[1])
    assert 2 <= n < 15
    one = client.get(f"/print?orn=3&ids={ids[0]}", base_url=BASE).get_data(as_text=True)
    assert "fit on a page" not in one                       # one page: nothing to say
