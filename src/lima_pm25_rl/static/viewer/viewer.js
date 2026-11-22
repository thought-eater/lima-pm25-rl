(() => {
    "use strict";
    const data = JSON.parse(document.getElementById("viewer-data").textContent);
    const maps = window.missionMaps;
    const $ = id => document.getElementById(id);
    let historical=false, limits=data.limits;
    const connected=!!document.querySelector('meta[name="pm25-live"]');
    const localClock=(start,seconds)=>new Date(Date.parse(start)+seconds*1000).toLocaleTimeString("es-PE",{timeZone:"America/Lima",hour:"2-digit",minute:"2-digit",hourCycle:"h23"});
    function panelLabels() {
        $("truth-title").textContent=historical?"Observaciones SENAMHI":"Campo simulado";
        $("error-title").textContent=historical?"Error horario en estaciones":"Error: simulado − estimado";
        document.querySelector(".route-key span:last-child").textContent=historical?"▲ SENAMHI":"▲ Fijas sintéticas";
        for(const [id,h,s] of [["global-rmse","RMSE horario acumulado","RMSE global"],["hot-rmse","MAE horario acumulado","RMSE en focos"],["recall","Varianza integrada (10⁹)","Sensibilidad en focos"]])
            $(id).previousElementSibling.textContent=historical?h:s;
    }
    const colors = ["#0072b2", "#d55e00", "#009e73"];
    const state = {mission: 0, frame: 0, playing: false, ready: false, selectedCell: null};
    let timer = null, revision = 0, syncing = false, cellPopup = null, current = null;
    const directions=["N","NE","E","SE","S","SO","O","NO"];
    function status(message) {
        $("viewer-status").hidden = !message;
        $("viewer-status").textContent = message;
    }
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
            image: i ? L.imageOverlay("", data.bounds, {pane: "fields", opacity: 1}) : null,
            sensors: L.layerGroup().addTo(map),
            samples: L.layerGroup().addTo(map),
            routes: L.layerGroup().addTo(map),
            selection: L.layerGroup().addTo(map),
        };
    });
    const grid=document.querySelector(".map-grid"), cards=[...grid.querySelectorAll(".map-card")];
    let expanded=-1;
    function expand(index) {
        const old=expanded, source=maps[old===-1?0:old];
        const center=source.getCenter(), zoom=source.getZoom();
        expanded=old===index?-1:index;
        grid.classList.toggle("focused",expanded!==-1);
        cards.forEach((card,i)=>{
            card.classList.toggle("enlarged",expanded===i);
            card.inert=expanded!==-1 && expanded!==i;
            const b=card.querySelector(".expand");
            b.textContent=expanded===i?"Cerrar ×":"⛶";
            b.title=expanded===i?"Volver a los cuatro mapas":"Ampliar mapa";
            b.setAttribute("aria-label",b.title+": "+card.querySelector("h2").firstChild.textContent);
            b.setAttribute("aria-expanded",String(expanded===i));
        });
        syncing=true;
        maps.forEach(m=>{m.invalidateSize({pan:false});m.setView(center,zoom,{reset:true,animate:false});});
        syncing=false;
        cards[index].querySelector(".expand").focus({preventScroll:true});
    }
    cards.forEach((card,i)=>{
        const b=document.createElement("button");
        b.className="expand";b.textContent="⛶";b.title="Ampliar mapa";b.setAttribute("aria-expanded","false");
        b.setAttribute("aria-label","Ampliar mapa: "+card.querySelector("h2").firstChild.textContent);
        b.onclick=()=>expand(i);card.querySelector("h2").append(b);
    });
    document.addEventListener("keydown",e=>{if(e.key==="Escape" && expanded!==-1) expand(expanded);});
    const home=L.control({position:"topleft"});
    home.onAdd=()=>{
        const b=L.DomUtil.create("button","map-home");b.textContent="⌖";b.title="Encuadrar el dominio";b.setAttribute("aria-label",b.title);
        L.DomEvent.disableClickPropagation(b);b.onclick=()=>maps[0].fitBounds(data.bounds,{animate:false,padding:[5,5]});return b;
    };
    home.addTo(maps[0]);
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
            if (!state.ready) return;
            if (cellPopup) cellPopup.remove();
            state.selectedCell = findCell(event.latlng.lat, event.latlng.lng);
            cellPopup = state.selectedCell===null ? null : L.popup({autoPan:false}).setLatLng(event.latlng).setContent("…").openOn(map);
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
        const range = limits[scale];
        $(id).innerHTML = `<span>${fmt(range[0], 2)}</span><span class="color-strip"></span><span>${fmt(range[1], 2)} µg/m³</span><span>${esc(description)}</span>`;
        $(id).querySelector(".color-strip").style.background = `linear-gradient(to right, ${data.palettes[palette].join(",")})`;
    }
    function updateLegends() {
        const std = $("estimate-mode").value === "std";
        legend("estimate-legend", std ? "std" : "concentration", std ? "magma" : "viridis", std ? "Desviación estándar" : "Media GP");
        legend("truth-legend", "concentration", "viridis", historical ? "PM2.5 observado · solo puntos" : "Misma escala que la media");
        legend("error-legend", "error", "RdBu_r", historical ? "Observado − estimación horaria previa · solo puntos" : "Azul: sobreestima · Rojo: subestima");
    }
    function showCell() {
        layers.forEach(layer => layer.selection.clearLayers());
        if (state.selectedCell === null || !cellPopup?.isOpen() || !current) return;
        const n = state.selectedCell, c = data.cells[n], m = current, f = state.frame;
        const mean = m.values.mean[f][n], std = m.values.std[f][n];
        cellPopup.setContent(historical ?
            `Celda [${c.row}, ${c.column}] · Estimado: ${fmt(mean)} · σ: ${fmt(std)} µg/m³. No existe verdad histórica por celda.` :
            `Celda [${c.row}, ${c.column}] · Campo simulado: ${fmt(m.values.truth[f][n])} · Estimado: ${fmt(mean)} · σ: ${fmt(std)} · Error: ${fmt(m.values.truth[f][n]-mean)} µg/m³`);
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
                if (historical && index>=2) {
                    const r=m.hourly.filter(h=>h.station===sample.id && h.elapsed_s<=time).at(-1);
                    if (index===3 && !r) continue;
                    const value=index===2 ? sample.value : r.observed-r.predicted;
                    const [lo,hi]=limits[index===2 ? "concentration" : "error"];
                    const palette=data.palettes[index===2 ? "viridis" : "RdBu_r"];
                    const color=palette[Math.round(Math.max(0,Math.min(1,(value-lo)/(hi-lo)))*(palette.length-1))];
                    const text=index===2 ? `Observado: ${fmt(value)} · cierre ${sample.time/60} min` :
                        `Observado: ${fmt(r.observed)} · predicción horaria: ${fmt(r.predicted)} · diferencia: ${fmt(value)} · cierre ${r.elapsed_s/60} min`;
                    L.circleMarker(sample.position,{radius:8,color:"#222",weight:1,fillColor:color,fillOpacity:1})
                        .bindTooltip(`${esc(sample.id)}: ${fmt(value,2)} µg/m³`).bindPopup(`${esc(sample.id)}<br>${text}`).addTo(layer.sensors);
                    continue;
                }
                L.marker(sample.position, {icon: L.divIcon({
                    className: "station-symbol", html: "▲", iconSize: [20, 20], iconAnchor: [10, 10]
                })}).bindPopup(`${esc(sample.id)}<br>${historical ? "Observación SENAMHI" : "Medición sintética"}: ${fmt(sample.value)} µg/m³<br>t=${sample.time/60} min`).addTo(layer.sensors);
            }
            if (historical && index>=2) return;
            if (m.actions && index===0) for (const a of m.actions.filter(a=>a.time===time)) {
                if (!$("q-panel").hidden && $("q-panel").open) {
                    const step = m.q_trace.filter(r=>r.time===time)[Number($("q-step").value)];
                    if (step && Math.floor(step.chosen/8)!==a.vehicle) continue;
                }
                L.circleMarker(a.destination,{radius:9,color:colors[a.vehicle],fill:false,dashArray:"3 2"})
                    .bindTooltip(`M${a.vehicle+1}: ${a.direction===null ? "permanencia forzada" : "destino elegido · "+["N","NE","E","SE","S","SO","O","NO"][a.direction]}`).addTo(layer.routes);
            }
            for (const route of m.routes) {
                const end = route.ends ? route.ends[frame] : route.points.length-1, color = colors[route.vehicle];
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
    // Los datos numéricos y la proyección viajan una sola vez dentro del HTML.
    const decoded=new Map();
    async function unpack(s) {
        const bytes=Uint8Array.from(atob(s), c=>c.charCodeAt(0));
        const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
        return new Int32Array(await new Response(stream).arrayBuffer());
    }
    const indices=unpack(data.raster.index);
    const rgb=Object.fromEntries(Object.entries(data.palettes).map(([k,list])=>[k,list.map(c=>parseInt(c.slice(1),16))]));
    async function storedFrame(m,frame) {
        const values={},images={},n=data.cells.length;
        for(const [k,key] of Object.entries(m.fields)){
            if(!decoded.has(key)) decoded.set(key,unpack(data.fields[key]));
            const a=await decoded.get(key);
            values[k]={[frame]:Float64Array.from(a.subarray(frame*n,(frame+1)*n),v=>v/100000)};
        }
        const idx=await indices, canvas=document.createElement("canvas");
        canvas.width=data.raster.width;canvas.height=data.raster.height;
        const ctx=canvas.getContext("2d"), pic=ctx.createImageData(canvas.width,canvas.height);
        for(const k of [$("estimate-mode").value,...(m.historical?[]:["truth","error"])]){
            const scale=k==="std"?"std":k==="error"?"error":"concentration";
            const [lo,hi]=data.limits[scale], palette=rgb[{std:"magma",error:"RdBu_r"}[k]||"viridis"];
            const colors=new Uint32Array(n);
            for(let j=0;j<n;j++){
                const v=k==="error"?values.truth[frame][j]-values.mean[frame][j]:values[k][frame][j];
                const c=palette[Math.max(0,Math.min(palette.length-1,Math.floor((v-lo)/(hi-lo)*palette.length)))];
                colors[j]=(0xff000000|((c&255)<<16)|(c&0xff00)|(c>>16))>>>0;
            }
            const pixels=new Uint32Array(pic.data.buffer);
            for(let j=0;j<idx.length;j++) pixels[j]=idx[j]<0?0:colors[idx[j]];
            ctx.putImageData(pic,0,0);images[k]={[frame]:canvas.toDataURL()};
        }
        return {...m,values,images};
    }
    const strategyNames={neural:"Agente BiLG-D3QN",random:"Aleatoria",greedy:"Greedy",sweep:"Barrido",fixed:"Solo fijas"};
    const studyNames={"fijo":"GP fijo · 30 000","adaptativo-final":"GP adaptativo · final 30 000","adaptativo-mejor":"GP adaptativo · mejor 28 400"};
    function model(m) {
        const label=studyNames[m.study] || m.study;
        return m.strategy_id==="neural" ? label : (m.estimator==="adaptativo" ? "GP adaptativo" : "GP fijo");
    }
    function options(id,pairs,preferred) {
        const el=$(id);
        el.replaceChildren(...Array.from(new Map(pairs),([v,label])=>new Option(label,v)));
        if (Array.from(el.options).some(o=>o.value===String(preferred))) el.value=String(preferred);
        return el.value;
    }
    function selectors(preferred) {
        const m=preferred || {};
        const key=r=>(r.historical?"hist":r.live?"live":"sint")+r.case;
        const title=r=>r.historical?`Replay · ${r.start.slice(0,10)} · ${localClock(r.start,0)} (Lima)`:`${r.live?"En vivo":"Sintética"} · ${r.case}`;
        const scenario=options("mission-select",data.missions.map(r=>[key(r),title(r)]),preferred?key(m):$("mission-select").value);
        const cases=data.missions.filter(r=>key(r)===scenario);
        const st=options("strategy-select",cases.map(r=>[r.strategy_id,strategyNames[r.strategy_id] || r.strategy]),m.strategy_id ?? $("strategy-select").value);
        const candidates=cases.filter(r=>r.strategy_id===st);
        let chosenModel=m.study ? model(m) : $("model-select").value;
        // Pasar de una política a su referencia conserva la familia del GP.
        if (!candidates.some(r=>model(r)===chosenModel)) chosenModel=candidates.map(model).find(l=>/adaptativo/i.test(l)===/adaptativo/i.test(chosenModel));
        const which=options("model-select",candidates.map(r=>[model(r),model(r)]),chosenModel);
        return data.missions.indexOf(candidates.find(r=>model(r)===which));
    }
    function qPanel(reset=true) {
        const time=current.times[state.frame];
        const action=(current.actions || current.executed || []).filter(a=>a.time===time);
        const line=$("action-line");
        line.textContent=action.length ? (current.executed?"Movimiento ejecutado: ":"Destinos: ")+action.map(a=>`M${a.vehicle+1} ${a.direction===null?"permanece":"→ "+directions[a.direction]}`).join("  ·  ") :
            current.strategy_id==="fixed"?"Sin vehículos móviles":time===0?"Posiciones iniciales":current.times.at(-1)===time?"Fin de misión":"";
        const trace=(current?.q_trace || []).filter(r=>r.time===current.times[state.frame]);
        $("q-panel").hidden=!trace.length;
        if (!trace.length) return;
        if (reset) options("q-step",trace.map((r,i)=>[String(i),`${i+1} de ${trace.length}`]),0);
        const r=trace[Number($("q-step").value)];
        const vehicle=Math.floor(r.chosen/8), direction=r.chosen%8;
        $("q-choice").textContent=`M${vehicle+1} → ${directions[direction]} · Q ${fmt(r.q[vehicle][direction],4)}`;
        $("q-table").innerHTML=`<thead><tr><th>Vehículo</th>${directions.map(d=>`<th>${d}</th>`).join("")}</tr></thead><tbody>`+
            r.q.map((row,v)=>`<tr><th>M${v+1}</th>`+row.map((q,d)=>`<td class="${!r.valid[v][d]?"unavailable":r.chosen===8*v+d?"chosen":""}">${r.valid[v][d]?fmt(q,4):"—"}</td>`).join("")+"</tr>").join("")+"</tbody>";
    }
    $("q-step").onchange=()=>{qPanel(false);updateSensors(current,state.frame);};
    $("q-panel").ontoggle=()=>{
        if ($("q-panel").open) pause();
        if (current) updateSensors(current,state.frame);
    };
    async function renderFrame(frame, mission = selectors()) {
        const token = ++revision;
        let m = data.missions[mission];
        frame = Math.max(0, Math.min(frame, m.times.length-1));
        state.ready = false;
        $("previous").disabled = true; $("next").disabled = true;
        status(m.live ? "Calculando el instante solicitado…" : "");
        try {
            if (m.live) {
                const params=new URLSearchParams({case:m.case,model:m.study_id,strategy:m.strategy_id,tick:frame});
                const resp=await fetch("/frame?"+params);
                if (!resp.ok) throw Error(`No se pudo calcular el instante (${resp.status}). Comprueba la terminal del visor.`);
                const packet=await resp.json();
                if (token!==revision) return;
                const indexed=obj=>Object.fromEntries(Object.entries(obj).map(([k,v])=>[k,{[frame]:v}]));
                m={...m,...packet,values:indexed(packet.values),images:indexed(packet.images),metrics:{[frame]:packet.metrics}};
            } else m=await storedFrame(m,frame);
            if (token!==revision) return;
            const urls = [m.images[$("estimate-mode").value][frame]];
            if (!m.historical) urls.push(m.images.truth[frame], m.images.error[frame]);
            await Promise.all(urls.map(loadImage));
            if (token !== revision) return;
            await new Promise(resolve => requestAnimationFrame(resolve));
            if (token !== revision) return;
            historical=!!m.historical; limits=m.limits||data.limits;panelLabels();
            for(let i=1;i<4;i++){
                const img=layers[i].image;
                if(i<=urls.length)img.setUrl(urls[i-1]).addTo(maps[i]);else img.remove();
            }
            state.mission = mission; state.frame = frame; state.ready = true;
            current=m;
            selectors(m);
            $("time-slider").max = String(m.times.length-1);
            $("time-slider").value = String(frame);
            $("time-label").textContent = `${m.times[frame]/60} min · ${frame}/${m.times.length-1}`;
            const metrics = m.metrics[frame];
            const waiting=historical && metrics.rmse===null;
            $("first-hour").hidden=!waiting;
            $("first-hour").textContent=waiting?`Primera comparación a las ${localClock(m.start,3600)}`:"";
            $("global-rmse").parentElement.hidden=waiting;$("hot-rmse").parentElement.hidden=waiting;
            $("global-rmse").textContent = `${fmt(metrics.rmse)} µg/m³`;
            $("hot-rmse").textContent = historical ? `${fmt(metrics.mae)} µg/m³` : metrics.hot_rmse === null ? "Sin focos" : `${fmt(metrics.hot_rmse)} µg/m³`;
            $("distance").textContent = `${fmt(metrics.distance_km, 2)} km`;
            $("recall").textContent = historical ? fmt(metrics.variance/1e9) : metrics.recall === null ? "Sin focos" : `${fmt(metrics.recall*100, 1)} %`;
            $("previous").disabled = frame === 0; $("next").disabled = frame === m.times.length-1;
            status(m.routes.length && m.times[frame] > m.decision_end ? `Decisiones terminadas en ${m.decision_end/60} min.` : "");
            qPanel(); updateSensors(m, frame); updateLegends(); showCell();
            $("startup-message").hidden = true;
        } catch (error) {
            if (token !== revision) return;
            pause(); status(error.message);
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
    selectors(data.missions[0]);
    for (const id of ["mission-select","strategy-select","model-select"]) $(id).onchange=()=>{
        pause();renderFrame(id==="mission-select"?0:state.frame,selectors());
    };
    if (connected) {
        $("new-scenario").hidden=false;
        $("new-scenario").onclick=async()=>{
            pause();++revision;$("new-scenario").disabled=true;
            try {
                const resp=await fetch("/scenario",{method:"POST"});
                if (!resp.ok) throw Error("No se pudo crear la misión.");
                const missions=await resp.json();data.missions.push(...missions);
                await renderFrame(0,selectors(missions[0]));
            } catch(e){status(e.message);} finally{$("new-scenario").disabled=false;}
        };
    }
    $("time-slider").oninput = event => { pause(); renderFrame(Number(event.target.value)); };
    $("previous").onclick = () => { pause(); renderFrame(state.frame-1); };
    $("next").onclick = () => { pause(); renderFrame(state.frame+1); };
    $("estimate-mode").onchange = () => { pause(); renderFrame(state.frame); };
    $("samples-toggle").onchange = () => {if(current) updateSensors(current, state.frame);};
    $("osm-toggle").onchange = event => maps.forEach((map, i) => {
        if (event.target.checked) layers[i].roads.addTo(map); else map.removeLayer(layers[i].roads);
    });
    window.addEventListener("resize", () => maps.forEach(map => map.invalidateSize()));
    window.pm25Viewer = {state, renderFrame, findCell, layers};
    renderFrame(0);
})();
