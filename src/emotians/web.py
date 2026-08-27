"""Web server for browser-based expression inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from emotians.inference import ExpressionCNNClassifier, YuNetFaceDetector, process_frame

DEFAULT_FACE_MODEL = Path("artifacts/models/face_detection_yunet_2023mar.onnx")
DEFAULT_EXPRESSION_MODEL = Path("artifacts/runs/stage_2_balanced/best_expression_cnn.pt")


def create_app(
    face_model_path: str | Path = DEFAULT_FACE_MODEL,
    expression_model_path: str | Path = DEFAULT_EXPRESSION_MODEL,
) -> FastAPI:
    torch.set_num_threads(1)
    app = FastAPI(title="EMOTIANS", version="0.1.0")
    detector = YuNetFaceDetector(face_model_path)
    classifier = ExpressionCNNClassifier(expression_model_path)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return WEB_PAGE

    @app.post("/api/analyze")
    async def analyze(frame: UploadFile = File(...)) -> dict[str, object]:
        payload = await frame.read()
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")
        result = process_frame(image, detector, classifier, expression_threshold=0.45)
        return {
            "status": result.status,
            "frame_size": [image.shape[1], image.shape[0]],
            "detections": [
                {"box": detection.xyxy, "confidence": detection.confidence}
                for detection in result.detections
            ],
            "prediction": (
                {"label": result.prediction.label, "confidence": result.prediction.confidence}
                if result.prediction is not None
                else None
            ),
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the EMOTIANS web application")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--face-model", type=Path, default=DEFAULT_FACE_MODEL)
    parser.add_argument("--expression-checkpoint", type=Path, default=DEFAULT_EXPRESSION_MODEL)
    args = parser.parse_args()
    uvicorn.run(create_app(args.face_model, args.expression_checkpoint), host=args.host, port=args.port)


LEGACY_WEB_PAGE = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>EMOTIANS | Expression Lab</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono&family=Space+Grotesk:wght@400;600;700&display=swap');
:root{--ink:#18221f;--paper:#f4f1e8;--lime:#c5ed72;--coral:#f28b6d}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 10% 0,#fff9dc 0 18%,transparent 40%),var(--paper);font-family:'Space Grotesk',sans-serif}main{width:min(1180px,calc(100% - 36px));margin:auto;padding:34px 0 44px}header{display:flex;justify-content:space-between;align-items:end;border-bottom:2px solid var(--ink);padding-bottom:18px;margin-bottom:28px}h1{margin:0;font-size:clamp(2.4rem,7vw,5.5rem);line-height:.85}.mono{font:12px 'DM Mono',monospace;text-transform:uppercase}.kicker{text-align:right;line-height:1.5}.layout{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(250px,.65fr);gap:22px}.stage{position:relative;overflow:hidden;background:#24302b;border:2px solid var(--ink);box-shadow:10px 10px 0 var(--coral)}video{display:block;width:100%;min-height:420px;object-fit:cover;transform:scaleX(-1)}canvas{position:absolute;inset:0;width:100%;height:100%;transform:scaleX(-1)}.empty{display:grid;place-items:center;min-height:420px;color:var(--paper);padding:30px;text-align:center}.panel{border-top:2px solid var(--ink);padding-top:14px}.panel+.panel{margin-top:28px}.label{color:#5e6c61;margin-bottom:9px}#expression{font-size:clamp(2.4rem,5vw,4.2rem);font-weight:700;line-height:.95;text-transform:capitalize;min-height:70px}button{border:2px solid var(--ink);background:var(--lime);padding:13px 17px;font:600 1rem 'Space Grotesk';cursor:pointer;box-shadow:5px 5px 0 var(--ink)}button:active{transform:translate(3px,3px);box-shadow:2px 2px 0 var(--ink)}#status{margin-top:17px;line-height:1.4}@media(max-width:760px){main{width:calc(100% - 24px);padding-top:20px}header{align-items:start}.layout{display:block}.stage{margin-bottom:34px;box-shadow:6px 6px 0 var(--coral)}video,.empty{min-height:58vw}}
</style></head>
<body><main><header><h1>EMOTIANS</h1><div class="kicker mono">Live expression<br>inference / 01</div></header><section class="layout"><div class="stage"><video id="video" autoplay playsinline hidden></video><canvas id="overlay"></canvas><div class="empty" id="empty">Creado por Cristian Morales y Juan Benavides.</div></div><aside><div class="panel"><div class="label mono">Expression</div><div id="expression">-</div><div id="confidence" class="mono">Esperando rostro</div></div><div class="panel"><button id="toggle">Activar cámara</button><div id="status" class="mono">Sistema detenido</div></div></aside></section></main>
<script>
const video=document.querySelector('#video'),canvas=document.querySelector('#overlay'),empty=document.querySelector('#empty'),toggle=document.querySelector('#toggle'),expression=document.querySelector('#expression'),confidence=document.querySelector('#confidence'),status=document.querySelector('#status'),ctx=canvas.getContext('2d');let stream=null,timer=null,busy=false;
function draw(result){const sx=canvas.width/video.videoWidth,sy=canvas.height/video.videoHeight;ctx.clearRect(0,0,canvas.width,canvas.height);ctx.lineWidth=4;ctx.strokeStyle='#c5ed72';ctx.fillStyle='#c5ed72';ctx.font='600 18px Space Grotesk';for(const item of result.detections){const[x,y,w,h]=[item.box[0]*sx,item.box[1]*sy,(item.box[2]-item.box[0])*sx,(item.box[3]-item.box[1])*sy];ctx.strokeRect(x,y,w,h);if(result.prediction)ctx.fillText(result.prediction.label,x+8,Math.max(24,y-8))}if(result.prediction){expression.textContent=result.prediction.label;confidence.textContent=Math.round(result.prediction.confidence*100)+'% confidence'}else{expression.textContent='-';confidence.textContent=result.status==='no-face'?'Rostro no detectado':'Analizando'}status.textContent=result.status==='ok'?'Señal estable':result.status}
async function analyze(){if(busy||!stream||!video.videoWidth)return;busy=true;canvas.width=video.videoWidth;canvas.height=video.videoHeight;const shot=document.createElement('canvas');shot.width=video.videoWidth;shot.height=video.videoHeight;shot.getContext('2d').drawImage(video,0,0);try{const blob=await new Promise(resolve=>shot.toBlob(resolve,'image/jpeg',.78)),body=new FormData();body.append('frame',blob,'frame.jpg');const response=await fetch('/api/analyze',{method:'POST',body});if(!response.ok)throw Error();draw(await response.json())}catch(error){status.textContent='Error de conexión'}finally{busy=false}}
async function start(){try{stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'user'},audio:false});video.srcObject=stream;video.hidden=false;empty.hidden=true;toggle.textContent='Detener cámara';status.textContent='Conectando modelo';timer=setInterval(analyze,700)}catch(error){status.textContent='Permiso de cámara requerido'}}function stop(){if(timer)clearInterval(timer);timer=null;if(stream)stream.getTracks().forEach(track=>track.stop());stream=null;video.hidden=true;empty.hidden=false;ctx.clearRect(0,0,canvas.width,canvas.height);toggle.textContent='Activar cámara';status.textContent='Sistema detenido';expression.textContent='-';confidence.textContent='Esperando rostro'}toggle.addEventListener('click',()=>stream?stop():start());
</script></body></html>"""


WEB_PAGE = """<!doctype html>
<html lang="es">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>EMOTIANS | Expression Lab</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
        :root { --ink:#17211e; --muted:#68736a; --paper:#f5f2e9; --panel:#fffdf6; --lime:#c8f36b; --coral:#f08b70; --cyan:#bce6df; --line:#ccd0c2; }
        * { box-sizing:border-box; }
        body { margin:0; min-height:100vh; overflow-x:hidden; color:var(--ink); background:linear-gradient(135deg,#f5f2e9 0%,#eef2e4 100%); font-family:'Space Grotesk',sans-serif; }
        body:before { content:''; position:fixed; inset:0; pointer-events:none; opacity:.32; background-image:linear-gradient(rgba(23,33,30,.04) 1px,transparent 1px),linear-gradient(90deg,rgba(23,33,30,.04) 1px,transparent 1px); background-size:32px 32px; }
        main { position:relative; width:min(1240px,calc(100% - 48px)); margin:auto; padding:32px 0 52px; }
        header { display:flex; justify-content:space-between; align-items:center; padding-bottom:22px; border-bottom:1px solid var(--ink); }
        .brand { display:flex; align-items:center; gap:14px; }
        .mark { width:16px; height:16px; background:var(--coral); border:2px solid var(--ink); transform:rotate(45deg); }
        h1 { margin:0; font-size:clamp(2rem,5vw,4.6rem); line-height:.85; letter-spacing:0; }
        .mono { font:500 11px 'DM Mono',monospace; letter-spacing:.04em; text-transform:uppercase; }
        .header-meta { display:flex; align-items:center; gap:18px; color:var(--muted); }
        .status-dot { display:inline-block; width:8px; height:8px; margin-right:7px; border-radius:50%; background:#9ba399; }
        .status-dot.live { background:#58b86b; box-shadow:0 0 0 4px rgba(88,184,107,.16); }
        .intro { display:flex; justify-content:space-between; align-items:end; gap:30px; padding:28px 0 20px; }
        .intro h2 { max-width:580px; margin:0; font-size:clamp(1.7rem,3.5vw,3.1rem); line-height:1; letter-spacing:0; }
        .intro p { max-width:250px; margin:0; color:var(--muted); line-height:1.45; }
        .credits { display:block; margin-top:10px; color:var(--ink); font-weight:600; }
        .workspace { display:grid; grid-template-columns:minmax(0,1.6fr) minmax(280px,.7fr); gap:24px; align-items:stretch; }
        .stage { position:relative; width:100%; aspect-ratio:16 / 10; max-height:calc(100vh - 280px); overflow:hidden; background:#26322e; border:1px solid var(--ink); box-shadow:12px 12px 0 var(--coral); }
        video { display:block; width:100%; height:100%; object-fit:cover; transform:scaleX(-1); }
        canvas { position:absolute; inset:0; width:100%; height:100%; pointer-events:none; transform:scaleX(-1); }
        .stage-top,.stage-bottom { position:absolute; left:18px; right:18px; display:flex; justify-content:space-between; color:#eef2e4; pointer-events:none; }
        .stage-top { top:17px; }.stage-bottom { bottom:16px; color:#c2cec1; }
        .live-tag { padding:6px 9px; color:var(--ink); background:var(--lime); }
        .empty { position:absolute; inset:0; display:grid; place-items:center; padding:38px; color:var(--paper); text-align:center; }
        aside { display:flex; flex-direction:column; gap:16px; }
        .result-card,.control-card { padding:22px; background:var(--panel); border:1px solid var(--line); }
        .result-card { flex:1; }.eyebrow { margin-bottom:24px; color:var(--muted); }
        #expression { min-height:112px; font-size:clamp(2.7rem,5vw,5rem); font-weight:700; line-height:.86; letter-spacing:0; text-transform:capitalize; }
        .confidence-row { display:flex; justify-content:space-between; gap:12px; margin-top:30px; padding-top:14px; border-top:1px solid var(--line); color:var(--muted); }
        #confidence { color:var(--ink); }
        .control-card { background:var(--cyan); }.control-label { margin-bottom:14px; }
        button { width:100%; padding:14px 16px; color:var(--ink); background:var(--lime); border:2px solid var(--ink); font:600 1rem 'Space Grotesk',sans-serif; cursor:pointer; box-shadow:5px 5px 0 var(--ink); }
        button:active { transform:translate(3px,3px); box-shadow:2px 2px 0 var(--ink); } button:disabled { opacity:.55; cursor:not-allowed; }
        #status { min-height:32px; margin-top:17px; color:var(--muted); line-height:1.45; }
        @media (max-width:800px) { main{width:calc(100% - 24px);padding-top:20px}.header-meta{gap:8px}.header-meta .mono:last-child{display:none}.intro{display:block;padding-top:24px}.intro h2{max-width:430px}.intro p{margin-top:14px}.workspace{display:block}.stage{aspect-ratio:4 / 3;max-height:none;margin-bottom:28px;box-shadow:7px 7px 0 var(--coral)}.result-card{padding:19px}aside{gap:12px} }
    </style>
</head>
<body>
    <main>
        <header>
            <div class="brand"><span class="mark"></span><h1>EMOTIANS</h1></div>
            <div class="header-meta"><span class="mono"><span class="status-dot" id="status-dot"></span><span id="header-status">Standby</span></span><span class="mono">LAB / 01</span></div>
        </header>
        <section class="intro"><h2>Read the room.<br><em>One expression at a time.</em></h2><p>Real-time facial expression analysis, running privately through this session.<span class="credits">Creado por Cristian Morales y Juan Benavides.</span></p></section>
        <section class="workspace">
            <div class="stage">
                <video id="video" autoplay playsinline hidden></video><canvas id="overlay"></canvas>
                <div class="stage-top"><span class="mono">Camera feed</span><span class="live-tag mono" id="live-tag">Offline</span></div>
                <div class="stage-bottom"><span class="mono">YuNet / CNN</span><span class="mono">48 × 48 input</span></div>
            </div>
            <aside>
                <div class="result-card"><div class="eyebrow mono">Detected expression</div><div id="expression">-</div><div class="confidence-row mono"><span>Confidence</span><span id="confidence">Waiting for face</span></div></div>
                <div class="control-card"><div class="control-label mono">Session control</div><button id="toggle">Activar cámara</button><div id="status" class="mono">Sistema detenido</div></div>
            </aside>
        </section>
    </main>
    <script>
        const video=document.querySelector('#video'),canvas=document.querySelector('#overlay'),toggle=document.querySelector('#toggle'),expression=document.querySelector('#expression'),confidence=document.querySelector('#confidence'),status=document.querySelector('#status'),headerStatus=document.querySelector('#header-status'),statusDot=document.querySelector('#status-dot'),liveTag=document.querySelector('#live-tag'),ctx=canvas.getContext('2d');let stream=null,timer=null,busy=false;
        function draw(result){const[sentWidth,sentHeight]=result.frame_size||[video.videoWidth,video.videoHeight],sx=canvas.width/sentWidth,sy=canvas.height/sentHeight;ctx.clearRect(0,0,canvas.width,canvas.height);ctx.lineWidth=4;ctx.strokeStyle='#c8f36b';ctx.fillStyle='#c8f36b';ctx.font='600 18px Space Grotesk';for(const item of result.detections){const[x,y,w,h]=[item.box[0]*sx,item.box[1]*sy,(item.box[2]-item.box[0])*sx,(item.box[3]-item.box[1])*sy];ctx.strokeRect(x,y,w,h);if(result.prediction)ctx.fillText(result.prediction.label,x+8,Math.max(24,y-8))}if(result.prediction){expression.textContent=result.prediction.label;confidence.textContent=Math.round(result.prediction.confidence*100)+'%'}else{expression.textContent='-';confidence.textContent=result.status==='no-face'?'No face detected':'Analysing'}status.textContent=result.status==='ok'?'Signal stable':result.status}
        async function analyze(){if(busy||!stream||!video.videoWidth)return;busy=true;canvas.width=video.videoWidth;canvas.height=video.videoHeight;const shot=document.createElement('canvas'),scale=Math.min(1,640/video.videoWidth);shot.width=Math.round(video.videoWidth*scale);shot.height=Math.round(video.videoHeight*scale);shot.getContext('2d').drawImage(video,0,0,shot.width,shot.height);try{const blob=await new Promise(resolve=>shot.toBlob(resolve,'image/jpeg',.68)),body=new FormData();body.append('frame',blob,'frame.jpg');const response=await fetch('/api/analyze',{method:'POST',body});if(!response.ok)throw Error();draw(await response.json())}catch(error){status.textContent='Connection error';headerStatus.textContent='Connection error'}finally{busy=false}}
        async function start(){try{stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'user'},audio:false});video.srcObject=stream;video.hidden=false;toggle.textContent='Detener cámara';status.textContent='Conectando modelo';headerStatus.textContent='Live';statusDot.classList.add('live');liveTag.textContent='Live';timer=setInterval(analyze,700)}catch(error){status.textContent='Permiso de cámara requerido';headerStatus.textContent='Camera blocked'}}function stop(){if(timer)clearInterval(timer);timer=null;if(stream)stream.getTracks().forEach(track=>track.stop());stream=null;video.hidden=true;ctx.clearRect(0,0,canvas.width,canvas.height);toggle.textContent='Activar cámara';status.textContent='Sistema detenido';headerStatus.textContent='Standby';statusDot.classList.remove('live');liveTag.textContent='Offline';expression.textContent='-';confidence.textContent='Waiting for face'}toggle.addEventListener('click',()=>stream?stop():start());
    </script>
</body>
</html>"""
