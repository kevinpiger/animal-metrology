const $ = id => document.getElementById(id);
const labels = {valid:'通過檢查',unmatched:'未配對到眼點',outside_mask:'位於動物遮罩外',low_confidence:'信心不足',outside_image:'超出圖片',ambiguous_mask:'跨動物遮罩歧義',missing_confidence:'缺少信心值',invalid_coordinates:'無有效座標'};
const species = {zebra:'斑馬',sheep:'羊',horse:'馬',dog:'狗',cat:'貓',bird:'鳥',cow:'牛',bear:'熊',elephant:'象',giraffe:'長頸鹿'};
const state = {sample:null,file:null,url:null,result:null,job:null,view:'original',busy:false,ready:false,ruler:false,points:[],selected:null};
const esc = text => String(text ?? '').replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const px = value => value == null ? '不可量測' : `${value.toFixed(2)} px`;
const artifact = name => `/api/jobs/${state.job}/artifacts/${name}`;
function message(text, error=false) { $('jobStatus').textContent=text; $('jobStatus').classList.toggle('error',error); }
function setBusy(busy) {
  state.busy=busy; $('working').hidden=!busy;
  $('run').disabled=busy||!state.ready||(!state.sample&&!state.file);
  $('run').querySelector('span').textContent=busy?'分析進行中':state.result?'重新分析':'開始分析';
  document.querySelectorAll('.sample, #upload, #removeUpload, .settings input, .settings select').forEach(el=>el.disabled=busy);
}
function clearResult() {
  state.result=null; state.job=null;state.selected=null;state.points=[];state.view='original';state.ruler=false;
  $('animals').innerHTML='<div class="empty"><h3>影像已就緒</h3><p>按「開始分析」，查看實際模型輸出。</p></div>';
  $('animalCount').textContent='尚未分析';
  $('pairs').innerHTML='<p class="muted">等待分析結果。缺少有效右眼的配對不會計算距離。</p>';
  $('overlayTab').disabled=true;$('candidates').disabled=true;$('candidates').checked=false;
  $('downloadJson').removeAttribute('href');$('downloadCsv').removeAttribute('href');$('downloadImage').removeAttribute('href');
  document.querySelectorAll('.download').forEach(a=>{a.classList.add('disabled');a.setAttribute('aria-disabled','true')});
  $('originalTab').classList.add('active');$('originalTab').setAttribute('aria-pressed','true');
  $('overlayTab').classList.remove('active');$('overlayTab').setAttribute('aria-pressed','false');
  $('ruler').setAttribute('aria-pressed','false');$('rulerValue').textContent='在圖片點兩下，人工核對距離';
  $('imageWrap').classList.remove('ruler-mode');draw();
}
function updateUploadPreview() {
  const selected=!!state.file;
  $('uploadPrompt').hidden=selected;
  $('uploadPreview').hidden=!selected;
  if(selected){
    $('uploadedImage').src=state.url;
    $('uploadedImage').alt=state.file.name;
    $('uploadedName').textContent=state.file.name;
  }else{
    $('uploadedImage').removeAttribute('src');
    $('uploadedName').textContent='';
  }
}
function removeUpload() {
  if(state.busy)return;
  if(state.url)URL.revokeObjectURL(state.url);
  state.file=null;state.sample=null;state.url=null;state.originalUrl=null;
  $('upload').value='';clearResult();updateUploadPreview();
  $('preview').removeAttribute('src');$('preview').alt='尚未選擇影像';
  $('imageMeta').textContent='—';$('sourceLabel').textContent='選擇一張圖片開始';
  $('ruler').disabled=true;$('clearRuler').disabled=true;
  $('animals').innerHTML='<div class="empty"><h3>尚未選擇圖片</h3><p>上傳圖片或選擇範例後開始分析。</p></div>';
  document.querySelectorAll('.sample').forEach(b=>{b.classList.remove('selected');b.setAttribute('aria-pressed','false')});
  message('圖片已移除，請上傳圖片或選擇範例。');setBusy(false);
  $('upload').focus();
}
function selectSample(sample) {
  if(state.busy)return;
  if(state.url){URL.revokeObjectURL(state.url);state.url=null}
  state.sample=sample;state.file=null;$('upload').value='';clearResult();updateUploadPreview();
  $('preview').src=sample.url;$('preview').alt=sample.name;
  $('sourceLabel').textContent=`COCO val2017 · ${sample.file} · 原始標註 ${sample.animal_count} 隻動物`;
  document.querySelectorAll('.sample').forEach(b=>{b.classList.toggle('selected',b.dataset.id===sample.id);b.setAttribute('aria-pressed',b.dataset.id===sample.id)});
  message('範例已就緒。按「開始分析」執行既有模型。');setBusy(false);
}
function selectFile(file) {
  if(!file||state.busy)return;
  if(file.size>10*1024*1024){message('圖片不可超過 10 MB。',true);return}
  if(!['image/jpeg','image/png','image/webp'].includes(file.type)){message('請選擇 JPG、PNG 或 WebP 圖片。',true);return}
  if(state.url)URL.revokeObjectURL(state.url);
  state.file=file;state.sample=null;state.url=URL.createObjectURL(file);clearResult();updateUploadPreview();
  $('preview').src=state.url;$('preview').alt=file.name;
  $('sourceLabel').textContent=`本機圖片 · ${file.name}`;
  document.querySelectorAll('.sample').forEach(b=>{b.classList.remove('selected');b.setAttribute('aria-pressed','false')});
  message('圖片已就緒。按「開始分析」檢查輪廓與眼點。');setBusy(false);
}
async function jsonResponse(response) {
  let data;try{data=await response.json()}catch{throw new Error('服務回應格式不正確，請確認本地服務仍在執行。')}
  if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:`請求未完成（${response.status}），請檢查輸入。`);
  return data;
}
async function analyze() {
  if(state.busy)return;
  const form=new FormData();
  if(state.file)form.append('image',state.file);else if(state.sample)form.append('sample_id',state.sample.id);else return;
  form.append('conf',$('detectConf').value);form.append('eye_conf',$('eyeConf').value);
  form.append('match_iou',$('matchIou').value);form.append('device',$('device').value);
  if(state.result)$('preview').src=state.originalUrl;
  clearResult();setBusy(true);message('正在提交圖片與分析參數…');
  try {
    const created=await jsonResponse(await fetch('/api/jobs',{method:'POST',body:form}));state.job=created.id;
    let failures=0;
    while(true) {
      await new Promise(r=>setTimeout(r,650));
      let job;
      try {job=await jsonResponse(await fetch(`/api/jobs/${state.job}`));failures=0}
      catch(error){failures++;if(failures>=5)throw error;message('連線暫時中斷，正在重新取得同一筆分析進度…');continue}
      if(job.status==='complete'){
        state.result=job.result;state.points=[];renderResult();
        const valid=job.result.animals.filter(a=>a.inter_eye_distance_px!=null).length;
        const count=job.result.animals.length;
        message(`分析完成 · ${count} 隻動物，${valid} 組有效雙眼量測${count<2?'。未偵測到至少兩隻動物。':'。'}`);
        break;
      }
      if(job.status==='failed')throw new Error(job.error);
      message(job.status==='queued'?'工作已建立，等待分析…':'正在執行模型與眼點檢查，請稍候…');
    }
  } catch(error) {message(`分析未完成：${error.message}`,true)}
  finally {setBusy(false)}
}
function renderResult() {
  const result=state.result;
  $('overlayTab').disabled=false;$('candidates').disabled=false;
  $('animalCount').textContent=`${result.animals.length} 隻動物`;
  $('deviceStatus').textContent=`${result.runtime.device==='mps'?'Apple MPS':result.runtime.device.toUpperCase()} · 本地運算`;
  $('animals').innerHTML=result.animals.length?result.animals.map(animal=>{
    const sideHtml=side=>{
      const eye=animal.eyes[side];
      return `<div class="eye-row"><span>${side==='left'?'左眼':'右眼'}</span><span class="badge ${eye.status==='valid'?'valid':''}">${esc(labels[eye.status]||eye.status)}</span></div><div class="eye-info">${eye.confidence==null?'信心 —':`信心 ${eye.confidence.toFixed(3)}`}${eye.point?` · (${eye.point[0].toFixed(1)}, ${eye.point[1].toFixed(1)})`:''}</div>`;
    };
    return `<article class="animal" data-animal="${animal.id}"><div class="animal-heading"><button class="select-animal" data-id="${animal.id}">#${animal.id} ${esc(species[animal.class_name]||animal.class_name)}</button><small>偵測 ${(animal.confidence*100).toFixed(1)}%</small></div>${sideHtml('left')}${sideHtml('right')}<div class="distance-row"><span>雙眼距離</span><strong class="${animal.inter_eye_distance_px==null?'unavailable':''}">${px(animal.inter_eye_distance_px)}</strong></div><div class="animal-meta">眼點來源：Animal Pose · 骨架 ${animal.pose?'已配對':'未配對'} · 眼點 ${animal.eye_match.status==='matched'?`配對 IoU ${animal.eye_match.match_iou.toFixed(2)}`:'未配對'}</div><a class="mask-link" href="${artifact(animal.mask)}" target="_blank" rel="noopener">檢視動物 Mask ↗</a></article>`;
  }).join(''):'<div class="empty"><h3>未偵測到動物</h3><p>可換用 COCO 範例，或檢查影像與偵測門檻。</p></div>';
  document.querySelectorAll('.select-animal').forEach(button=>button.onclick=()=>{
    state.selected=state.selected===Number(button.dataset.id)?null:Number(button.dataset.id);
    document.querySelectorAll('.animal').forEach(a=>a.classList.toggle('selected',Number(a.dataset.animal)===state.selected));draw();
  });
  $('pairs').innerHTML=result.right_eye_distances.length?result.right_eye_distances.map(pair=>`<div class="pair"><strong>#${pair.animal_ids[0]} ↔ #${pair.animal_ids[1]}</strong><span>${pair.distance_px==null?'不可量測 · 缺少有效右眼':px(pair.distance_px)}</span></div>`).join(''):'<p class="muted">至少需要兩個動物實例才能建立配對。</p>';
  for(const [id,name] of [['downloadJson','result.json'],['downloadCsv','measurements.csv'],['downloadImage','overlay.jpg']]){
    const link=$(id);link.href=artifact(name);link.download=name;link.classList.remove('disabled');link.setAttribute('aria-disabled','false');
  }
  // The server normalizes EXIF orientation. Use that exact image for pixel comparisons.
  state.originalUrl=artifact('source.jpg');setView('overlay');
}
function setView(view) {
  state.view=view;
  if(state.result)$('preview').src=view==='overlay'?artifact('overlay.jpg'):state.originalUrl;
  for(const [id,value] of [['originalTab','original'],['overlayTab','overlay']]){
    $(id).classList.toggle('active',view===value);$(id).setAttribute('aria-pressed',view===value);
  }
  draw();
}
function svgElement(name,attributes,text){
  const el=document.createElementNS('http://www.w3.org/2000/svg',name);
  for(const [k,v] of Object.entries(attributes))el.setAttribute(k,v);
  if(text!=null)el.textContent=text;return el;
}
function draw() {
  const svg=$('annotations');svg.replaceChildren();
  const width=state.result?.image_size.width||$('preview').naturalWidth;
  const height=state.result?.image_size.height||$('preview').naturalHeight;
  if(!width||!height)return;
  svg.setAttribute('viewBox',`0 0 ${width} ${height}`);
  const radius=Math.max(width/150,3);
  if(state.result&&$('candidates').checked){
    for(const animal of state.result.animals){
      for(const side of ['left','right']){
        const eye=animal.eyes[side];if(!eye.point)continue;
        const [x,y]=eye.point;const valid=eye.status==='valid';const color=valid?'#16865b':'#d68a12';
        const circle=svgElement('circle',{cx:x,cy:y,r:radius,fill:valid?color:'#ffffffaa',stroke:color,'stroke-width':2,'vector-effect':'non-scaling-stroke'});
        circle.append(svgElement('title',{},`#${animal.id} ${side==='left'?'左眼':'右眼'}：${labels[eye.status]||eye.status}`));svg.append(circle);
        svg.append(svgElement('rect',{x:x-radius*2,y:y-radius*2,width:radius*4,height:radius*4,fill:'none',stroke:color,'stroke-width':1,'stroke-dasharray':valid?'none':'4 3','vector-effect':'non-scaling-stroke'}));
        svg.append(svgElement('text',{x:x+radius*3,y:y-radius*2,fill:color,'font-size':Math.max(width/48,12),'font-weight':600,stroke:'#fff','stroke-width':2,'paint-order':'stroke'},`#${animal.id} ${side==='left'?'L':'R'}`));
      }
    }
  }
  if(state.selected!=null&&state.result){
    const animal=state.result.animals.find(a=>a.id===state.selected);
    if(animal){const [x,y,x2,y2]=animal.bbox;svg.append(svgElement('rect',{x,y,width:x2-x,height:y2-y,class:'selected-highlight'}))}
  }
  for(let i=0;i<state.points.length;i++){
    const [x,y]=state.points[i];svg.append(svgElement('circle',{cx:x,cy:y,r:radius,fill:'#2f70bb',stroke:'#fff','stroke-width':1.5,'vector-effect':'non-scaling-stroke'}));
    svg.append(svgElement('text',{x:x+radius*2,y:y-radius*2,fill:'#2f70bb','font-size':Math.max(width/45,13),'font-weight':650,stroke:'#fff','stroke-width':2,'paint-order':'stroke'},i===0?'A':'B'));
  }
  if(state.points.length===2){
    const [a,b]=state.points;svg.prepend(svgElement('line',{x1:a[0],y1:a[1],x2:b[0],y2:b[1],stroke:'#2f70bb','stroke-width':2,'stroke-dasharray':'5 4','vector-effect':'non-scaling-stroke'}));
  }
}
$('preview').onload=()=>{
  const img=$('preview');$('imageMeta').textContent=`${img.naturalWidth} × ${img.naturalHeight} px`;
  $('ruler').disabled=false;$('clearRuler').disabled=false;draw();
};
$('preview').onerror=()=>message('圖片無法顯示，請重新選擇有效圖片。',true);
$('annotations').addEventListener('click',event=>{
  if(!state.ruler||state.busy)return;
  const svg=$('annotations');const pt=svg.createSVGPoint();pt.x=event.clientX;pt.y=event.clientY;
  const point=pt.matrixTransform(svg.getScreenCTM().inverse());
  const bounds=svg.viewBox.baseVal;if(point.x<0||point.y<0||point.x>=bounds.width||point.y>=bounds.height)return;
  if(state.points.length===2)state.points=[];
  state.points.push([point.x,point.y]);
  $('rulerValue').textContent=state.points.length===1?`A (${point.x.toFixed(1)}, ${point.y.toFixed(1)}) · 請選第二點`:`人工距離 ${Math.hypot(state.points[0][0]-point.x,state.points[0][1]-point.y).toFixed(2)} px · 再點可重測`;
  draw();
});
$('ruler').onclick=()=>{state.ruler=!state.ruler;$('ruler').setAttribute('aria-pressed',state.ruler);$('imageWrap').classList.toggle('ruler-mode',state.ruler)};
$('clearRuler').onclick=()=>{state.points=[];$('rulerValue').textContent='在圖片點兩下，人工核對距離';draw()};
$('candidates').onchange=draw;$('originalTab').onclick=()=>setView('original');$('overlayTab').onclick=()=>setView('overlay');
$('upload').onchange=event=>selectFile(event.target.files[0]);
$('run').onclick=analyze;
$('removeUpload').onclick=removeUpload;
for(const [id,output] of [['eyeConf','eyeValue'],['detectConf','detectValue'],['matchIou','matchValue']])$(id).oninput=()=>{
  $(output).textContent=Number($(id).value).toFixed(2);
  if(state.result)message('參數已調整，畫面仍是上次結果。請重新分析套用新門檻。');
};
$('device').onchange=()=>{if(state.result)message('裝置已調整，重新分析後套用。')};
const settingsDialog=$('settingsDialog');
$('openSettings').onclick=()=>settingsDialog.showModal();
$('closeSettings').onclick=()=>settingsDialog.close();
$('doneSettings').onclick=()=>settingsDialog.close();
settingsDialog.addEventListener('click',event=>{
  const r=settingsDialog.getBoundingClientRect();
  if(event.target===settingsDialog&&(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom))settingsDialog.close();
});
const drop=$('dropZone');
for(const name of ['dragenter','dragover'])drop.addEventListener(name,e=>{e.preventDefault();drop.classList.add('drag')});
for(const name of ['dragleave','drop'])drop.addEventListener(name,e=>{e.preventDefault();drop.classList.remove('drag')});
drop.addEventListener('drop',event=>selectFile(event.dataTransfer.files[0]));
async function init(){
  try{
    const [samples,health]=await Promise.all([fetch('/api/samples').then(jsonResponse),fetch('/api/health').then(jsonResponse)]);
    $('samples').innerHTML=samples.map(sample=>`<button class="sample" data-id="${sample.id}" aria-pressed="false"><img src="${sample.url}" alt=""><span>${esc(sample.name)}<small>COCO #${sample.id}</small></span></button>`).join('');
    document.querySelectorAll('.sample').forEach(b=>b.onclick=()=>selectSample(samples.find(s=>s.id===b.dataset.id)));
    state.ready=health.ready;
    $('deviceStatus').textContent=health.ready?`${health.runtime.device==='mps'?'Apple MPS':health.runtime.device.toUpperCase()} · 模型已就緒`:'模型或裝置未就緒';
    if(samples.length)selectSample(samples[0]);
    if(!health.ready)message('模型或裝置未就緒。請檢查 models 目錄與服務紀錄。',true);
  }catch(error){message(`無法連線本地服務：${error.message}`,true);$('deviceStatus').textContent='服務未連線'}
}
init();
