(() => {
    "use strict";
    const data = JSON.parse(document.getElementById("viewer-data").textContent);
    const maps = window.missionMaps;
    const $ = id => document.getElementById(id);
    const colors = ["#0072b2", "#d55e00", "#009e73"];
    const state = {mission: 0, frame: 0, playing: false, ready: false, selectedCell: null};
    let timer = null, revision = 0, syncing = false;
    const layers = maps.map((map, i) => {
        $(`slot-${i}`).appendChild(map.getContainer());
        map.createPane("fields").style.zIndex = 350;
        map.invalidateSize();
        map.fitBounds(data.bounds, {padding: [5, 5]});
        const roads = L.polyline(data.roads, {
            color: "#647781", weight: i === 0 ? .65 : .5, opacity: i === 0 ? .8 : .15, interactive: false
        }).addTo(map);
        L.rectangle(data.domain_bounds, {
            color: "#465f6c", weight: 1.5, fill: false, dashArray: "5 4", interactive: false
        }).addTo(map);
        map.attributionControl.addAttribution('Calles: &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>');
        return {
            roads,
            image: i ? L.imageOverlay("", data.bounds, {pane: "fields", opacity: 1}).addTo(map) : null,
            sensors: L.layerGroup().addTo(map),
            samples: L.layerGroup().addTo(map),
            routes: L.layerGroup().addTo(map),
            selection: L.layerGroup().addTo(map),
        };
    });
    for (const map of maps) {
        map.on("moveend", () => {
            if (syncing) return;
            syncing = true;
            for (const other of maps) {
                if (other !== map) other.setView(map.getCenter(), map.getZoom(), {animate: false});
            }
            syncing = false;
        });
        map.on("click", event => {
            state.selectedCell = findCell(event.latlng.lat, event.latlng.lng);
            showCell();
        });
    }

    const cellBounds = data.cells.map(c => {
        const lat = c.polygon.map(p => p[0]), lon = c.polygon.map(p => p[1]);
        return [Math.min(...lat), Math.min(...lon), Math.max(...lat), Math.max(...lon)];
    });
    function findCell(lat, lon) {
        for (let n = 0; n < data.cells.length; n++) {
            const b = cellBounds[n];
            if (lat < b[0] || lon < b[1] || lat > b[2] || lon > b[3]) continue;
            const p = data.cells[n].polygon;
            let inside = false;
            for (let i = 0, j = p.length - 1; i < p.length; j = i++) {
                if ((p[i][0] > lat) !== (p[j][0] > lat) &&
                    lon < (p[j][1]-p[i][1]) * (lat-p[i][0]) / (p[j][0]-p[i][0]) + p[i][1]) inside = !inside;
            }
            if (inside) return n;
        }
        return null;
    }
    const fmt = (value, digits=3) => value === null ? "—" : Number(value).toLocaleString("es-PE", {
        minimumFractionDigits: digits, maximumFractionDigits: digits
    });
    const esc = text => String(text).replace(/[&<>"']/g, c => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    })[c]);
    function legend(id, scale, palette, description) {
        const limits = data.limits[scale];
        $(id).innerHTML = `<span>${fmt(limits[0], 2)}</span><span class="color-strip"></span><span>${fmt(limits[1], 2)} µg/m³</span><span>${esc(description)}</span>`;
        $(id).querySelector(".color-strip").style.background = `linear-gradient(to right, ${data.palettes[palette].join(",")})`;
    }
    function updateLegends() {
        const std = $("estimate-mode").value === "std";
        legend("estimate-legend", std ? "std" : "concentration", std ? "magma" : "viridis", std ? "Desviación estándar" : "Media GP");
        legend("truth-legend", "concentration", "viridis", "Misma escala que la media");
        legend("error-legend", "error", "RdBu_r", "Azul: sobreestima · Rojo: subestima");
    }
    function showCell() {
        layers.forEach(layer => layer.selection.clearLayers());
        if (state.selectedCell === null) {
            $("cell-values").textContent = "Pulsa una celda activa para consultar sus valores.";
            return;
        }
        const n = state.selectedCell, c = data.cells[n], m = data.missions[state.mission], f = state.frame;
        const truth = m.values.truth[f][n], mean = m.values.mean[f][n], std = m.values.std[f][n];
        $("cell-values").textContent = `Celda [${c.row}, ${c.column}] · Real: ${fmt(truth)} · Estimado: ${fmt(mean)} · σ: ${fmt(std)} · Error real − estimado: ${fmt(truth-mean)} µg/m³`;
        maps.forEach((map, i) => L.polygon(c.polygon, {
            color: "#141c24", weight: 2, fill: false, interactive: false
        }).addTo(layers[i].selection));
    }
    function updateSensors(m, frame) {
        const time = m.times[frame];
        const visible = m.samples.filter(sample => sample.time <= time);
        const fixed = new Map();
        for (const sample of visible) if (sample.type === "fixed") fixed.set(sample.id, sample);
        layers.forEach((layer, index) => {
            layer.sensors.clearLayers(); layer.samples.clearLayers(); layer.routes.clearLayers();
            for (const sample of fixed.values()) {
                L.marker(sample.position, {icon: L.divIcon({
                    className: "station-symbol", html: "▲", iconSize: [20, 20], iconAnchor: [10, 10]
                })}).bindPopup(`${esc(sample.id)}<br>Medición sintética: ${fmt(sample.value)} µg/m³<br>Tomada en t=${sample.time/60} min`).addTo(layer.sensors);
            }
            for (const route of m.routes) {
                const end = route.ends[frame], color = colors[route.vehicle];
                L.circleMarker(route.points[end], {
                    radius: 6, color: "white", weight: 2, fillColor: color, fillOpacity: 1
                }).bindTooltip(`M${route.vehicle+1}`, {
                    permanent: true, direction: "right", className: "vehicle-label"
                }).addTo(layer.sensors);
                if (index === 0) L.polyline(route.points.slice(0, end+1), {
                    color, weight: 3, opacity: .9, interactive: false
                }).addTo(layer.routes);
            }
            if ($("samples-toggle").checked) {
                for (const sample of visible) if (sample.type === "mobile") {
                    L.circleMarker(sample.position, {
                        radius: 2, color: "#202b35", weight: .5, fillColor: "white", fillOpacity: .9
                    }).bindPopup(`${esc(sample.id)} · t=${sample.time/60} min<br>${fmt(sample.value)} µg/m³`).addTo(layer.samples);
                }
            }
        });
    }
    function loadImage(url) {
        return new Promise((resolve, reject) => {
            const image = new Image();
            image.onload = resolve;
            image.onerror = () => reject(new Error(`No se pudo cargar ${url}. Conserva la carpeta frames junto al HTML.`));
            image.src = url;
        });
    }
    async function renderFrame(frame, mission = Number($("mission-select").value || state.mission)) {
        const token = ++revision, m = data.missions[mission];
        frame = Math.max(0, Math.min(frame, m.times.length-1));
        state.ready = false;
        $("previous").disabled = true; $("next").disabled = true;
        $("viewer-status").textContent = "Cargando los tres campos del mismo instante…";
        const urls = [m.images[$("estimate-mode").value][frame], m.images.truth[frame], m.images.error[frame]];
        try {
            await Promise.all(urls.map(loadImage));
            if (token !== revision) return;
            await new Promise(resolve => requestAnimationFrame(resolve));
            if (token !== revision) return;
            urls.forEach((url, i) => layers[i+1].image.setUrl(url));
            state.mission = mission; state.frame = frame; state.ready = true;
            $("mission-select").value = String(mission);
            $("time-slider").max = String(m.times.length-1);
            $("time-slider").value = String(frame);
            $("time-label").textContent = `${m.times[frame]/60} min · ${frame}/${m.times.length-1}`;
            const metrics = m.metrics[frame];
            $("global-rmse").textContent = `${fmt(metrics.rmse)} µg/m³`;
            $("hot-rmse").textContent = metrics.hot_rmse === null ? "Sin focos" : `${fmt(metrics.hot_rmse)} µg/m³`;
            $("distance").textContent = `${fmt(metrics.distance_km, 2)} km`;
            $("recall").textContent = metrics.recall === null ? "Sin focos" : `${fmt(metrics.recall*100, 1)} %`;
            $("previous").disabled = frame === 0; $("next").disabled = frame === m.times.length-1;
            $("mission-info").textContent = `${m.study} · Caso ${m.case} · ${m.strategy} · ${m.episode === null ? "Referencia" : "Política del episodio "+m.episode} · Inicio UTC ${m.start} · Grilla ${data.cell_size_m} m · Umbral experimental ${fmt(data.threshold)} µg/m³ (no sanitario)`;
            $("viewer-status").textContent = m.routes.length && m.times[frame] > m.decision_end ?
                `Decisiones terminadas en ${m.decision_end/60} min; sin nuevas mediciones móviles.` :
                "Ticks registrados de 5 min · Escalas fijas · Calles OSM conservadas · Bibliotecas Folium por CDN.";
            updateSensors(m, frame); updateLegends(); showCell();
            $("startup-message").hidden = true;
        } catch (error) {
            if (token !== revision) return;
            pause(); $("viewer-status").textContent = error.message;
        }
    }
    function pause() {
        state.playing = false; clearTimeout(timer); $("play").textContent = "Reproducir";
    }
    async function playNext() {
        if (!state.playing) return;
        const last = data.missions[state.mission].times.length-1;
        if (state.frame >= last) { pause(); return; }
        await renderFrame(state.frame+1);
        if (state.playing) timer = setTimeout(playNext, 1000/Number($("speed").value));
    }
    $("play").onclick = async () => {
        if (state.playing) { pause(); return; }
        if (!state.ready) return;
        if (state.frame === data.missions[state.mission].times.length-1) await renderFrame(0);
        state.playing = true; $("play").textContent = "Pausar";
        timer = setTimeout(playNext, 1000/Number($("speed").value));
    };
    data.missions.forEach((m, i) => {
        const option = document.createElement("option");
        option.value = String(i); option.textContent = `${m.study} / caso ${m.case} / ${m.strategy}`;
        $("mission-select").appendChild(option);
    });
    $("mission-select").onchange = event => { pause(); renderFrame(0, Number(event.target.value)); };
    $("time-slider").oninput = event => { pause(); renderFrame(Number(event.target.value)); };
    $("previous").onclick = () => { pause(); renderFrame(state.frame-1); };
    $("next").onclick = () => { pause(); renderFrame(state.frame+1); };
    $("estimate-mode").onchange = () => { pause(); renderFrame(state.frame); };
    $("samples-toggle").onchange = () => updateSensors(data.missions[state.mission], state.frame);
    $("osm-toggle").onchange = event => maps.forEach((map, i) => {
        if (event.target.checked) layers[i].roads.addTo(map); else map.removeLayer(layers[i].roads);
    });
    $("reset-view").onclick = () => maps[0].fitBounds(data.bounds, {animate: false, padding: [5, 5]});
    $("print-view").onclick = () => { pause(); window.print(); };
    window.addEventListener("resize", () => maps.forEach(map => map.invalidateSize()));
    window.pm25Viewer = {state, renderFrame, findCell, layers};
    renderFrame(0);
})();
