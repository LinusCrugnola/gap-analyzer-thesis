// Frame loader for the rollout viewer. Frames are fetched and decoded off-screen
// (createImageBitmap) and drawn onto one persistent canvas, so the old frame stays visible until
// the new one is ready (no blank flashes). The latest requested frame wins: while requests are in
// flight, newer ones replace older pending ones, so playback drops frames only when the server
// and network can't keep up. Upcoming frames are prefetched into a small cache.
window.gapViewer = (function () {
    const MAX_INFLIGHT = 3;     // concurrent fetches (show + prefetch)
    const MAX_CACHE = 160;      // decoded frames kept in memory
    const cache = new Map();    // url -> Promise<ImageBitmap>
    const times = [];
    let inflight = 0, pending = null, wanted = null, drawn = null, nDrawn = 0;

    function fetchBitmap(url) {
        let p = cache.get(url);
        if (p) { cache.delete(url); cache.set(url, p); return p; }  // refresh LRU position
        inflight++;
        p = fetch(url)
            .then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.blob(); })
            .then(b => createImageBitmap(b))
            .finally(() => { inflight--; drainPending(); });
        p.catch(() => cache.delete(url));
        cache.set(url, p);
        while (cache.size > MAX_CACHE) cache.delete(cache.keys().next().value);
        return p;
    }

    function canvas() {
        const box = document.getElementById("viewer");
        if (!box) return null;
        let c = box.querySelector("canvas");
        if (!c) {
            c = document.createElement("canvas");
            c.className = "viewer-img";
            box.replaceChildren(c);
        }
        return c;
    }

    function draw(bmp) {
        const c = canvas();
        if (!c) return;
        if (c.width !== bmp.width || c.height !== bmp.height) { c.width = bmp.width; c.height = bmp.height; }
        c.getContext("2d").drawImage(bmp, 0, 0);
    }

    function report() {
        const el = document.getElementById("step-time");
        if (!el || !times.length) return;
        const s = [...times].sort((a, b) => a - b);
        el.textContent = `viewer step (browser, request to display): median ${s[Math.floor(s.length / 2)].toFixed(0)} ms, ` +
            `p90 ${s[Math.floor(s.length * 0.9)].toFixed(0)} ms over last ${s.length}`;
    }

    function load(url, t0) {
        fetchBitmap(url).then(bmp => {
            if (wanted !== url) return;     // a newer frame was requested meanwhile
            draw(bmp);
            drawn = url;
            nDrawn++;
            times.push(performance.now() - t0);
            if (times.length > 100) times.shift();
            report();
        }).catch(() => {});
    }

    function drainPending() {
        if (pending && inflight < MAX_INFLIGHT) { const p = pending; pending = null; load(p.url, p.t0); }
    }

    function show(url) {
        wanted = url;
        const t0 = performance.now();
        if (cache.has(url) || inflight < MAX_INFLIGHT) load(url, t0);
        else pending = {url, t0};
    }

    function prefetch(urls) {
        for (const u of urls) {
            if (pending || inflight >= MAX_INFLIGHT - 1) return;  // leave a slot for the frame to show
            if (!cache.has(u)) fetchBitmap(u);
        }
    }

    function clear() {
        wanted = null;
        drawn = null;
        const box = document.getElementById("viewer");
        if (box) box.replaceChildren();
    }

    function moveMarker(x) {
        const gd = document.querySelector("#ep-plot .js-plotly-plot");
        if (window.Plotly && gd && gd.layout && gd.layout.shapes && gd.layout.shapes.length) {
            window.Plotly.relayout(gd, {"shapes[0].x0": x, "shapes[0].x1": x});
        }
    }

    function state() {  // for tests / debugging
        return {wanted, drawn, nDrawn, inflight, cached: cache.size};
    }

    return {show, prefetch, clear, moveMarker, state};
})();

// The stats-cursor slider (updatemode "mouseup") commits a typed step only on blur: make Enter commit too.
document.addEventListener("keydown", e => {
    if (e.key === "Enter" && e.target.matches && e.target.matches("#cutoff input")) e.target.blur();
});
