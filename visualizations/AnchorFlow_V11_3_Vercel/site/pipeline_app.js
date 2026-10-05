/* Real V11_3 readout inspector. No simulated fine field, teacher or model in browser. */
(async () => {
  'use strict';
  const $=id=>document.getElementById(id),M=JetMath,scenes=PIPELINE_SCENES,baseline=PIPELINE_BASELINE;
  const FW=1216,FH=352,N4=304*88,N2=608*176;
  const stages=[
    {name:'Coarse Jet NODE',group:'D4 / GEOMETRY',before:'D0',after:'D4',label:'D0 → D4',sub:'Bosh3 · 1/4 grid',clip:5,
      description:'Từ coarse metric depth, seed jet và tích phân chart đến T=1. Field refresh reaction, sparse source và transported neighbors ở mỗi RHS call.'},
    {name:'Phase lift 4 → 2',group:'D2 / RECONSTRUCTION',before:'D4',after:'D2_base',label:'D4 → D2_base',sub:'Convex phases · 2×',clip:5,
      description:'3×3 metric-depth hypotheses + learned phase weights/residual tạo half-resolution depth. Đây là learned upsampling, không thêm NODE calls.'},
    {name:'Five-jet consensus',group:'D2 / SUBPIXEL GEOMETRY',before:'D2_base',after:'D2_query',label:'Base → query',sub:'Center / E / W / S / N',clip:2,
      description:'Năm local surfaces query cùng target phase. Trộn inverse depth rồi reciprocal. Bước này chưa bao gồm learned metric-innovation correction.'},
    {name:'Metric innovation',group:'D2 / ABSOLUTE CORRECTION',before:'D2_query',after:'D2_pre_fine',label:'Query → metric readout',sub:'Sparse innovation / gating',clip:2,
      description:'Consensus được blend/gate với base; tiny metric residual dùng sparse innovation sửa absolute bias. Cặp này là toàn readout sau query, không chỉ một head độc lập.'},
    {name:'Fine Jet NODE',group:'D2 / FINE GEOMETRY',before:'D2_pre_fine',after:'D2',label:'D2 before → after NODE',sub:'Fixed midpoint RK2 · 2 NFE',clip:1,
      description:'Seed jet lại trên D2 grid. Một fixed midpoint step đánh giá cùng tiny field ở initial và RK-mid state. Không adaptive solve và không copy derivatives từ D4.'},
    {name:'Phase lift 2 → 1',group:'D1 / FULL-RESOLUTION READOUT',before:'D2',after:'D1_base',label:'D2 → D1_base',sub:'Phase lift + context',clip:2,
      description:'Learned phase upsampling dùng p2/g2/sensor context và quarter-grid rich feature. Output là scalar depth full-resolution; không full-res learned feature CNN.'},
    {name:'Learned detail',group:'D1 / LOCAL METRIC DETAIL',before:'D1_base',after:'D1_pre_pir',label:'D1_base → detail',sub:'Half-grid CNN / shuffle',clip:.5,
      description:'Existing half-grid detail head dự đoán four-phase residual, PixelShuffle thành scalar correction. Đây là detail trước phase-aware innovation residual mới.'},
    {name:'Phase innovation · PIR',group:'D1 / PHASE-AWARE SPARSE CORRECTION',before:'D1_pre_pir',after:'D1',label:'Detail → PIR',sub:'+952 params · no NODE',clip:.2,
      description:'Mean/std/density của donor sparse error theo từng phase + log-depth contrast. Tiny head đưa correction có gate vào D1 trước sensor fusion.'},
    {name:'Sensor reliability fusion',group:'DFULL / OUTPUT POLICY',before:'D1',after:'D_full',label:'D1 → Dfull',sub:'Learned soft sensor fusion',clip:.5,
      description:'Tại sparse sensor pixels, gate dùng learned trust và sensor/depth conflict để blend; ngoài sensor support giữ D1. Không GT/teacher hoặc compulsory hard anchor.'}
  ];
  let T,depths,gt,jets,fineChart,coarseV,fineValueChart,terms,sensorMask,sensorGate;
  let scene=0,stage=0,frame=40,fineFrame=2,zoom=1,mapMode='delta',clip=5,selected={x:290,y:206};
  let dirty=true,loading=false,playing=false,finePlaying=false,pipelinePlaying=false,lastTime=0,lastRender=0,yaw=-.62,pitch=.95;
  let pipelinePosition=1,pipelineTween=1,fineInternal=false;
  let fullCoarse=null,fullFine=null,fullPending=null,jetRegion=3,jetGain=2,compareMode='previous',wipe=.5;
  let rangeKey='',fixedRange=.01,fixedRef=.05;
  let mapKey='',beforeData,afterData,deltaStats,nativeStats;
  const rgb=new Image(),maps={};
  for(const key of ['before','after','delta','error','gate'])maps[key]=document.createElement('canvas');
  const labels=['v','gₓ','gᵧ','hₓₓ','hₓᵧ','hᵧᵧ'];
  const lineColors=['#087b76','#c66544','#447db1','#9867af','#a88939','#64934e'];
  const fmt=(x,p=3)=>Number(x).toFixed(p),sci=x=>Number(x).toExponential(3);
  function unpack(text,Type=Float32Array){const raw=atob(text),bytes=new Uint8Array(raw.length);for(let k=0;k<raw.length;k++)bytes[k]=raw.charCodeAt(k);return new Type(bytes.buffer);}
  function install(trace){
    if(trace.meta.source_sha256!==baseline.source_sha256||trace.meta.checkpoint_sha256!==baseline.checkpoint.sha256)throw new Error('Model/source SHA không đúng V11_3 baseline');
    T=trace;const a=T.arrays;jets=a.coarse_jets_roi.data;coarseV=a.coarse_v.data;fineChart=a.fine_chart_roi.data;fineValueChart=a.fine_value_chart.data;terms=a.terms.data;
    fullCoarse=null;fullFine=null;fullPending=null;rangeKey='';
    depths=Object.fromEntries(Object.entries(a).filter(([k])=>k.startsWith('depth/')).map(([k,b])=>[k.slice(6),{h:b.shape[0],w:b.shape[1],data:b.data}]));
    gt=Object.fromEntries(['4','2','1'].map(k=>[k,{h:a['gt/'+k].shape[0],w:a['gt/'+k].shape[1],data:a['gt/'+k].data,mask:a['gtmask/'+k].data}]));
    sensorMask=a.sensor_mask.data;sensorGate=a.sensor_gate.data;
    if(coarseV.length!==41*N4||fineValueChart.length!==3*N2)throw new Error('Missing pixel value trajectories');
    for(const b of Object.values(depths))if(b.data.length!==b.h*b.w)throw new Error('Depth array shape mismatch');
  }
  install(await PipelineLoader.decode(window.JET_PIPELINE_PACK));
  const stops=[[0,[161,25,48]],[.15,[233,72,47]],[.3,[254,186,73]],[.45,[218,235,151]],[.6,[82,201,189]],[.75,[44,136,210]],[1,[28,37,125]]];
  function color(d){const x=M.clamp(d/120,0,1);let a=0;while(a<stops.length-2&&x>stops[a+1][0])a++;const u=(x-stops[a][0])/(stops[a+1][0]-stops[a][0]);return stops[a][1].map((v,k)=>Math.round(v*(1-u)+stops[a+1][1][k]*u));}
  const rgba=(r,g,b)=>(255<<24)|(b<<16)|(g<<8)|r;
  const LUT=Uint32Array.from({length:4097},(_,k)=>rgba(...color(k*120/4096)));
  function diverge(value,range,negative,positive){const f=M.clamp(Math.abs(value)/range,0,1),to=value<0?negative:positive;return to.map((v,k)=>Math.round([247,249,250][k]*(1-f)+v*f));}
  const deltaLut=Uint32Array.from({length:8193},(_,k)=>rgba(...diverge((k-4096)/4096,1,[48,112,175],[204,101,72])));
  const errorLut=Uint32Array.from({length:8193},(_,k)=>rgba(...diverge((k-4096)/4096,1,[42,150,119],[158,88,158])));
  function target(b){return gt[String(FW/b.w)];}
  function indexAt(b,p=selected){return M.clamp(Math.floor(p.y*b.h/FH),0,b.h-1)*b.w+M.clamp(Math.floor(p.x*b.w/FW),0,b.w-1);}
  function crop(){const w=FW/zoom,h=FH/zoom;return {x:M.clamp(selected.x+.5-w/2,0,FW-w),y:M.clamp(selected.y+.5-h/2,0,FH-h),w,h};}
  function arrayJet(family,x,y,at){
    if(x<0||y<0||x>=(family==='fine'?608:304)||y>=(family==='fine'?176:88))return null;
    const full=family==='fine'?fullFine:fullCoarse;
    if(full){const fine=family==='fine',N=fine?N2:N4,k=y*(fine?608:304)+x,a=Math.floor(at),b=Math.min(fine?2:40,a+1),u=at-a;
      const value=(fine?fineValueChart:coarseV)[a*N+k]*(1-u)+(fine?fineValueChart:coarseV)[b*N+k]*u;
      const rest=Array.from({length:5},(_,c)=>full[(a*5+c)*N+k]*(1-u)+full[(b*5+c)*N+k]*u);
      return fine?M.decode([value,...rest]):[value,...rest];}
    const fine=family==='fine',r=fine?[130,90,32,24]:[65,45,16,12],N=r[2]*r[3],data=fine?fineChart:jets;
    if(x<r[0]||y<r[1]||x>=r[0]+r[2]||y>=r[1]+r[3])return null;
    const k=(y-r[1])*r[2]+x-r[0],a=Math.floor(at),b=Math.min(fine?2:40,a+1),u=at-a;
    const value=Array.from({length:6},(_,c)=>data[(a*6+c)*N+k]*(1-u)+data[(b*6+c)*N+k]*u);
    return fine?M.decode(value):value;
  }
  function fineDenseJet(x,y,t){if(fullFine){const k=y*608+x,states=[0,1,2].map(a=>[fineValueChart[a*N2+k],...Array.from({length:5},(_,c)=>fullFine[(a*5+c)*N2+k])]);return M.decode(M.midpointDense(...states,t));}
    const r=[130,90,32,24],N=32*24;if(x<130||y<90||x>=162||y>=114)return null;
    const k=(y-90)*32+x-130,states=[0,1,2].map(a=>Array.from({length:6},(_,c)=>fineChart[(a*6+c)*N+k]));return M.decode(M.midpointDense(...states,t));}
  function currentAfter(){
    const b=depths[stages[stage].after];
    if(stage===0&&frame===0)return depths.D0;
    if(stage===4&&fineFrame===0&&!fineInternal)return depths.D2_pre_fine;
    if(stage===0&&frame!==40){const a=Math.floor(frame),bb=Math.min(40,a+1),u=frame-a;const data=new Float32Array(N4);
      for(let k=0;k<N4;k++)data[k]=1/(coarseV[a*N4+k]*(1-u)+coarseV[bb*N4+k]*u);return {...b,data};}
    if(stage===4&&(fineFrame!==2||fineInternal)){const data=new Float32Array(N2),t=fineFrame/2;
      for(let k=0;k<N2;k++){const z0=fineValueChart[k],mid=fineValueChart[N2+k],end=fineValueChart[2*N2+k],z=fineInternal?mid:z0+2*(t-t*t)*(mid-z0)+t*t*(end-z0);data[k]=Math.exp(-Math.log(1/120)-Math.log(1200)*M.sigmoid(z));}return {...b,data};}
    if(stage!==0&&stage!==4&&pipelineTween<1){const before=depths[stages[stage].before],data=new Float32Array(b.w*b.h),u=pipelineTween;
      for(let y=0;y<b.h;y++)for(let x=0;x<b.w;x++){const k=y*b.w+x,bk=Math.floor(y*before.h/b.h)*before.w+Math.floor(x*before.w/b.w);data[k]=before.data[bk]*(1-u)+b.data[k]*u;}return {...b,data};}
    return b;
  }
  async function acceptFullJets(binary,chosenScene=scene){const pack=FULL_JET_PACKS[chosenScene];if(pack.meta.sample_id!==scenes[chosenScene].meta.sample_id||pack.meta.checkpoint_sha256!==baseline.checkpoint.sha256)throw new Error('Full jet sample mismatch');
    const result=await PipelineLoader.decode(pack,binary);if(chosenScene!==scene)return;
    fullCoarse=result.arrays.coarse_derivatives.data;fullFine=result.arrays.fine_chart_derivatives.data;rangeKey='';dirty=true;
    $('jetDataStatus').textContent='Full-image 6 coefficients · Float32 lossless · SHA verified';}
  async function ensureFullJets(){if(fullCoarse||fullPending)return fullPending;
    if(location.protocol==='file:'){$('jetDataStatus').textContent=`Offline: nạp web_traces/scene_${String(scene+1).padStart(2,'0')}.jets.bin.gz; hoặc dùng npm start.`;return;}
    const k=scene;$('jetDataStatus').textContent='Đang nạp full-image jets…';
    fullPending=(async()=>{const response=await fetch(`web_traces/scene_${String(k+1).padStart(2,'0')}.jets.bin.gz`);if(!response.ok)throw new Error('Missing full jet archive');await acceptFullJets(await response.arrayBuffer(),k);})().catch(e=>{if(k===scene)$('jetDataStatus').textContent=e.message;}).finally(()=>{if(k===scene)fullPending=null;});return fullPending;}
  function selectedRegion(){const family=stage>=4?'fine':'coarse',scale=family==='fine'?2:4,W=FW/scale,H=FH/scale,x=Math.floor(selected.x/scale),y=Math.floor(selected.y/scale);
    const size=jetRegion,rx=M.clamp(x-Math.floor(size/2),0,W-size),ry=M.clamp(y-Math.floor(size/2),0,H-size);
    return {family,scale,x,y,rx,ry,size,sx:rx*scale,sy:ry*scale,w:size*scale,h:size*scale};}
  function nativeMetric(b){const g=target(b);let n=0,sse=0,abs=0;for(let k=0;k<b.data.length;k++)if(g.mask[k]){const e=b.data[k]-g.data[k];n++;sse+=e*e;abs+=Math.abs(e);}return {rmse:Math.sqrt(sse/n),mae:abs/n,n};}
  function rebuildMaps(){
    const key=[scene,stage,stage===0?frame.toFixed(4):40,stage===4?fineFrame.toFixed(4):2,fineInternal,pipelineTween.toFixed(4),clip].join(':');if(key===mapKey)return;
    beforeData=depths[stages[stage].before];afterData=currentAfter();
    for(const [key,b]of [['before',beforeData],['after',afterData]]){
      const c=maps[key];c.width=b.w;c.height=b.h;const ctx=c.getContext('2d'),img=ctx.createImageData(b.w,b.h);
      const pixels=new Uint32Array(img.data.buffer);for(let k=0;k<b.data.length;k++)pixels[k]=LUT[M.clamp(Math.round(b.data[k]/120*4096),0,4096)];ctx.putImageData(img,0,0);
    }
    const images={};for(const key of ['delta','error','gate']){maps[key].width=afterData.w;maps[key].height=afterData.h;images[key]=maps[key].getContext('2d').createImageData(afterData.w,afterData.h);}
    const pixels=Object.fromEntries(Object.entries(images).map(([k,v])=>[k,new Uint32Array(v.data.buffer)]));
    const gray=rgba(224,230,233),g=target(afterData),N=afterData.w*afterData.h;let meanAbs=0,min=Infinity,max=-Infinity,better=0,worse=0,valid=0,errorChange=0;
    for(let y=0;y<afterData.h;y++)for(let x=0;x<afterData.w;x++){
      const k=y*afterData.w+x,bk=Math.floor(y*beforeData.h/afterData.h)*beforeData.w+Math.floor(x*beforeData.w/afterData.w);
      const delta=afterData.data[k]-beforeData.data[bk];meanAbs+=Math.abs(delta)/N;min=Math.min(min,delta);max=Math.max(max,delta);
      pixels.delta[k]=deltaLut[M.clamp(Math.round(delta/clip*4096)+4096,0,8192)];
      if(g.mask[k]){const e=Math.abs(afterData.data[k]-g.data[k])-Math.abs(beforeData.data[bk]-g.data[k]);
        valid++;errorChange+=e;if(e<0)better++;if(e>0)worse++;
        pixels.error[k]=errorLut[M.clamp(Math.round(e*4096)+4096,0,8192)];
      }else pixels.error[k]=gray;
      if(stage===8&&sensorMask[k]){const gate=sensorGate[k];pixels.gate[k]=rgba(Math.round(30+210*gate),Math.round(85+112*gate),Math.round(120-70*gate));}
      else pixels.gate[k]=gray;
    }
    for(const key of ['delta','error','gate'])maps[key].getContext('2d').putImageData(images[key],0,0);
    deltaStats={meanAbs,min,max,better,worse,valid,errorChange:errorChange/valid};nativeStats={before:nativeMetric(beforeData),after:nativeMetric(afterData)};mapKey=key;
  }
  function line(ctx,x,y,xx,yy,c='#ccd8dc',width=1){ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(xx,yy);ctx.strokeStyle=c;ctx.lineWidth=width;ctx.stroke();}
  function text(ctx,value,x,y,size=13,c='#142d36',weight=400){ctx.fillStyle=c;ctx.font=`${weight} ${size}px Segoe UI,Arial,sans-serif`;ctx.fillText(value,x,y);}
  function arrow(ctx,x,y,xx,yy,c='#667981'){line(ctx,x,y,xx,yy,c,1.4);const a=Math.atan2(yy-y,xx-x);ctx.beginPath();ctx.moveTo(xx,yy);ctx.lineTo(xx-7*Math.cos(a-.4),yy-7*Math.sin(a-.4));ctx.lineTo(xx-7*Math.cos(a+.4),yy-7*Math.sin(a+.4));ctx.closePath();ctx.fillStyle=c;ctx.fill();}
  function drawMap(ctx,key,x,y,w,h,overlay=true){
    const b=maps[key],cr=crop();ctx.save();ctx.imageSmoothingEnabled=false;
    ctx.drawImage(b,cr.x*b.width/FW,cr.y*b.height/FH,cr.w*b.width/FW,cr.h*b.height/FH,x,y,w,h);
    if(overlay){ctx.beginPath();ctx.rect(x,y,w,h);ctx.clip();const scale=FW/b.width;
      if($('grid').checked&&zoom>=4){ctx.beginPath();ctx.lineWidth=.7;ctx.strokeStyle='rgba(255,255,255,.23)';
        for(let gx=Math.ceil(cr.x/scale)*scale;gx<cr.x+cr.w;gx+=scale){const px=x+(gx-cr.x)*w/cr.w;ctx.moveTo(px,y);ctx.lineTo(px,y+h);}
        for(let gy=Math.ceil(cr.y/scale)*scale;gy<cr.y+cr.h;gy+=scale){const py=y+(gy-cr.y)*h/cr.h;ctx.moveTo(x,py);ctx.lineTo(x+w,py);}ctx.stroke();}
      const region=selectedRegion(),rrx=x+(region.sx-cr.x)*w/cr.w,rry=y+(region.sy-cr.y)*h/cr.h;
      ctx.fillStyle='rgba(255,219,91,.12)';ctx.fillRect(rrx,rry,region.w*w/cr.w,region.h*h/cr.h);ctx.strokeStyle='#ffdc68';ctx.lineWidth=2.2;ctx.strokeRect(rrx,rry,region.w*w/cr.w,region.h*h/cr.h);
      const jetScale=region.scale,sx=Math.floor(selected.x/jetScale)*jetScale,sy=Math.floor(selected.y/jetScale)*jetScale,px=x+(sx-cr.x)*w/cr.w,py=y+(sy-cr.y)*h/cr.h;
      ctx.strokeStyle='#142d36';ctx.lineWidth=3;ctx.strokeRect(px,py,jetScale*w/cr.w,jetScale*h/cr.h);
      ctx.strokeStyle='#fff';ctx.lineWidth=1.4;ctx.strokeRect(px,py,jetScale*w/cr.w,jetScale*h/cr.h);
    }ctx.restore();
  }
  function updateReadouts(){
    const bk=indexAt(beforeData),ak=indexAt(afterData),before=beforeData.data[bk],after=afterData.data[ak],delta=after-before,g=target(afterData);
    $('cellLabel').textContent=`u=${selected.x}, v=${selected.y}`;
    $('cellBefore').textContent=fmt(before)+' m';$('cellAfter').textContent=fmt(after)+' m';$('cellDelta').textContent=(delta>=0?'+':'')+fmt(delta)+' m';
    const a=nativeStats.before,b=nativeStats.after;
    $('sampleMetrics').innerHTML=`<table><thead><tr><th>Riêng scene · native GT</th><th>Trước</th><th>Sau</th></tr></thead><tbody><tr><td>RMSE · m</td><td>${fmt(a.rmse,4)}</td><td>${fmt(b.rmse,4)}</td></tr><tr><td>MAE · m</td><td>${fmt(a.mae,4)}</td><td>${fmt(b.mae,4)}</td></tr><tr><td>Valid cells</td><td>${a.n.toLocaleString('vi-VN')}</td><td>${b.n.toLocaleString('vi-VN')}</td></tr></tbody></table><p>CPU FP32 scene audit, không phải global400 hoặc anonymous KITTI test.</p>`;
    const same=beforeData.w===afterData.w;
    $('comparisonRule').textContent=same?'Cùng native grid / pooled-GT support. Bấm lên ảnh để chọn pixel; zoom giữ cùng vùng trước/sau.':'Đổi resolution: native metrics KHÔNG cùng target/support. Δdepth và Δ|lỗi| dùng nearest-expanded trước trên target grid sau, chỉ để inspect upsampling.';
    let note=g.mask[ak]?`GT target=${fmt(g.data[ak])} m · Δ|e| tại cell=${fmt(Math.abs(after-g.data[ak])-Math.abs(before-g.data[ak]))} m`:'Cell này không có GT target support.';
    if(stage===8)note+=sensorMask[ak]?` · Sensor gate=${fmt(sensorGate[ak])}`:' · Không có sparse sensor, gate=0.';
    $('impactStats').textContent=`Mean |ΔD| toàn grid=${fmt(deltaStats.meanAbs,4)} m. ${note}`;
    $('impactLegend').innerHTML=mapMode==='delta'?`<i class="legend-dot" style="background:#3070af"></i>− / gần hơn <i class="legend-dot" style="background:#cc6548"></i>+ / xa hơn · color ±${clip} m; số không clip. Δdepth không phải lỗi GT.`:
      mapMode==='error'?`<i class="legend-dot" style="background:#2a9677"></i>Âm: lỗi nhỏ hơn <i class="legend-dot" style="background:#9e589e"></i>Dương: lỗi lớn hơn · color ±1 m · xám: no GT. ${fmt(deltaStats.better/deltaStats.valid*100,1)}% target cells tốt hơn; không là ablation.`:
      'Gate0→1 trên sparse support. Xám: không sensor. Learned reliability, không ground-truth confidence.';
    const s=stages[stage],vb=baseline.native_stages[s.before],va=baseline.native_stages[s.after];
    $('valStageMetrics').textContent=`Full400 native RMSE: ${fmt(vb.rmse_m,4)} → ${fmt(va.rmse_m,4)} m ${same?'· cùng support':'· KHÁC scale/support'}`;
    $('solverNote').textContent=`Scene coarse: ${T.meta.solver.nfe} NFE / ${T.meta.solver.accepted_steps} accepted · fine:2 NFE · T=1 · weights ${T.meta.checkpoint_sha256.slice(0,10)}…`;
    $('observationNote').textContent=stage===0?'41 dense-output observations ≠41 neural calls. Giữa mốc, nội suy decoded jet chỉ để display. Full400 metrics ở terminal; current scene audit tại state hiển thị.':
      stage===4?(fineInternal?'Đang inspect internal RK-mid, KHÔNG exact z(0.5) và KHÔNG dense-half state.':'Fine progress: continuous extension RK2 bậc2 trong chart, reuse k1/k2; không exact ODE solution, không thêm NFE. Full400 numbers ở terminal.'):
      pipelineTween<1?'Đang blend hai recorded snapshots chỉ để trình bày; không là intermediate prediction của network. Scene GT audit là của ảnh blend, không một benchmark mới.':
      'Actual endpoint forward snapshots. Effect cùng checkpoint không chứng minh matched retraining gain.';
  }
  function selectedJetInfo(){
    const family=stage>=4?'fine':'coarse',scale=family==='fine'?2:4,x=Math.floor(selected.x/scale),y=Math.floor(selected.y/scale);
    const at=family==='fine'?(stage===4?fineFrame:2):(stage===0?frame:40);
    return {family,scale,x,y,at,j:family==='fine'?(stage===4&&fineInternal?arrayJet(family,x,y,1):fineDenseJet(x,y,at/2)):arrayJet(family,x,y,at),initial:arrayJet(family,x,y,0)};
  }
  function patch(ctx,j,cx,cy,size,ref,range,ghost=false,ox=0,oy=0,axes=true,emphasis=false){
    function project(s,t,v){s+=ox;t+=oy;const xx=Math.cos(yaw)*s-Math.sin(yaw)*t,yy=Math.sin(yaw)*s+Math.cos(yaw)*t,zz=(v-ref)/range*.6*jetGain;
      return [cx+xx*size,cy+(Math.cos(pitch)*yy-Math.sin(pitch)*zz)*size,Math.sin(pitch)*yy+Math.cos(pitch)*zz];}
    const n=ghost?6:jetRegion===1?24:jetRegion===3?12:8;
    const pol=[];for(let y=0;y<n;y++)for(let x=0;x<n;x++){
      const s=-.5+x/n,t=-.5+y/n,pts=[[s,t],[s+1/n,t],[s+1/n,t+1/n],[s,t+1/n]].map(([a,b])=>project(a,b,M.polynomial(j,a,b)));
      pol.push({pts,z:pts.reduce((v,p)=>v+p[2]/4,0),v:M.polynomial(j,s+.5/n,t+.5/n)});
    }pol.sort((a,b)=>b.z-a.z);
    for(const p of pol){ctx.beginPath();p.pts.forEach((v,k)=>k?ctx.lineTo(v[0],v[1]):ctx.moveTo(v[0],v[1]));ctx.closePath();ctx.fillStyle=ghost?'rgba(82,140,143,.035)':`rgb(${color(1/Math.max(1/120,p.v))})`;ctx.fill();ctx.strokeStyle=ghost?'rgba(8,123,118,.27)':'rgba(21,62,72,.22)';ctx.lineWidth=.55;ctx.stroke();}
    const boundary=[[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5],[-.5,-.5]].map(([s,t])=>project(s,t,M.polynomial(j,s,t)));
    ctx.beginPath();boundary.forEach((p,k)=>k?ctx.lineTo(p[0],p[1]):ctx.moveTo(p[0],p[1]));ctx.strokeStyle=ghost?'rgba(8,123,118,.45)':emphasis?'#ffdc68':'#365c68';ctx.lineWidth=emphasis&&!ghost?2.4:1;ctx.stroke();
    if(!ghost){const center=project(0,0,j[0]);ctx.beginPath();ctx.arc(center[0],center[1],emphasis?4:2,0,Math.PI*2);ctx.fillStyle='#142d36';ctx.fill();
      if(!axes)return;
      const origin=project(-.66,-.66,ref),sx=project(.66,-.66,ref),sy=project(-.66,.66,ref),vv=project(-.66,-.66,ref+range);
      arrow(ctx,origin[0],origin[1],sx[0],sx[1]);arrow(ctx,origin[0],origin[1],sy[0],sy[1]);arrow(ctx,origin[0],origin[1],vv[0],vv[1],'#087b76');
      text(ctx,'s',sx[0]+5,sx[1]+8,14);text(ctx,'t',sy[0]-7,sy[1]+15,14);text(ctx,'v · m⁻¹',vv[0]+7,vv[1]-5,13,'#087b76',600);
      text(ctx,fmt(ref+range,4),vv[0]+7,vv[1]+10,10,'#667981');text(ctx,fmt(ref,4),origin[0]-12,origin[1]+17,10,'#667981');}
  }
  function drawJet(){
    const info=selectedJetInfo(),j=info.j,j0=info.initial,c=$('surface'),ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);ctx.save();ctx.scale(2,2);
    const region=selectedRegion();$('jetFootprint').textContent=`Ô vàng: ${region.size}×${region.size} ${info.family==='fine'?'D2':'D4'} native cells = ${region.w}×${region.h} full-image pixels (${region.w*region.h} pixels), u ${region.sx}…${region.sx+region.w-1}, v ${region.sy}…${region.sy+region.h-1}. Ô trắng: cell tâm. Patch subdivisions chỉ là mesh hiển thị, không extra neural pixels.`;
    if(!j){text(ctx,'Đang chờ full-image coefficients của pixel này.',35,175,18);text(ctx,'HTTP/Vercel tự nạp; file offline dùng nút nạp .jets.bin.gz.',35,205,14,'#667981');ctx.restore();$('jetTitle').textContent='Chưa nạp full jets · không fabricate geometry';$('coefficients').innerHTML='';$('trajectory').getContext('2d').clearRect(0,0,1240,420);$('fieldNote').textContent='Dữ liệu full Float32 có sẵn cho cả 10 scenes; chọn đúng file của scene đang xem khi chạy file://.';return;}
    const cells=[];for(let y=region.ry;y<region.ry+region.size;y++)for(let x=region.rx;x<region.rx+region.size;x++){const initial=arrayJet(info.family,x,y,0),now=info.family==='fine'?(stage===4&&fineInternal?arrayJet('fine',x,y,1):fineDenseJet(x,y,info.at/2)):arrayJet('coarse',x,y,info.at);if(initial&&now)cells.push({x,y,initial,now});}
    const key=[scene,info.family,info.x,info.y,region.rx,region.ry,region.size,Boolean(fullCoarse)].join(':');
    if(key!==rangeKey){fixedRef=j0[0];fixedRange=Math.max(.0001,j0[0]*.06);for(const cell of cells)for(let k=0;k<(info.family==='fine'?33:41);k++){const a=info.family==='fine'?fineDenseJet(cell.x,cell.y,k/32):arrayJet('coarse',cell.x,cell.y,k);for(const s of [-.5,0,.5])for(const t of [-.5,0,.5])fixedRange=Math.max(fixedRange,Math.abs(M.polynomial(a,s,t)-fixedRef));}fixedRange*=1.12;rangeKey=key;}
    const size=226/Math.max(region.size,jetGain*1.4),ox=(region.size-1)/2;cells.sort((a,b)=>(Math.sin(yaw)*(b.x-region.rx)+Math.cos(yaw)*(b.y-region.ry))-(Math.sin(yaw)*(a.x-region.rx)+Math.cos(yaw)*(a.y-region.ry)));
    for(const cell of cells){const dx=cell.x-region.rx-ox,dy=cell.y-region.ry-ox,central=cell.x===info.x&&cell.y===info.y;patch(ctx,cell.initial,380,240,size,fixedRef,fixedRange,true,dx,dy,false);patch(ctx,cell.now,380,240,size,fixedRef,fixedRange,false,dx,dy,central,central);}
    text(ctx,`Initial ghost → current · τ=${fmt(info.family==='fine'?info.at/2:info.at/40)} · height gain ${jetGain}×`,20,28,13,'#087b76',600);
    text(ctx,`Fixed v reference ${fmt(fixedRef,5)} m⁻¹; range ±${fmt(fixedRange,5)} m⁻¹`,20,420,12,'#667981');ctx.restore();
    $('jetTitle').textContent=`${info.family==='fine'?'D2 jet':'D4 jet'} · cell (${info.x},${info.y}) · ${info.family==='fine'?(fineInternal&&stage===4?'internal RK-mid':`dense τ=${fmt(info.at/2)}`):`τ=${fmt(info.at/40)}`}${stage>4?' · upstream reference, không D1 jet':''}`;
    $('coefficients').innerHTML='<thead><tr><th>Coefficient</th><th>Initial</th><th>Selected</th><th>Δ</th></tr></thead><tbody>'+labels.map((name,k)=>`<tr><td style="color:${lineColors[k]}">${name}</td><td>${sci(j0[k])}</td><td>${sci(j[k])}</td><td>${sci(j[k]-j0[k])}</td></tr>`).join('')+'</tbody>';
    const tc=$('trajectory'),cc=tc.getContext('2d');cc.clearRect(0,0,tc.width,tc.height);cc.save();cc.scale(2,2);
    const count=info.family==='fine'?33:41,values=Array.from({length:count},(_,k)=>(info.family==='fine'?fineDenseJet(info.x,info.y,k/32):arrayJet(info.family,info.x,info.y,k)).map(v=>v/j0[0]));
    let low=-.15,high=1.25;values.flat().forEach(v=>{low=Math.min(low,v);high=Math.max(high,v);});
    const xx=k=>40+k/(count-1)*552,yy=v=>164-(v-low)/(high-low)*122;
    for(let k=0;k<5;k++){const y=42+k*122/4;line(cc,40,y,592,y,'#e2eaed');text(cc,fmt(high-k*(high-low)/4,1),6,y+4,10,'#667981');}
    for(let k=0;k<6;k++){cc.beginPath();values.forEach((v,index)=>index?cc.lineTo(xx(index),yy(v[k])):cc.moveTo(xx(index),yy(v[k])));cc.strokeStyle=lineColors[k];cc.lineWidth=1.8;cc.stroke();text(cc,labels[k],42+k*85,22,12,lineColors[k],600);}
    line(cc,xx(info.family==='fine'?info.at*16:info.at),37,xx(info.family==='fine'?info.at*16:info.at),169,'#142d36');text(cc,'τ=0',40,191,10,'#667981');text(cc,info.family==='fine'?'dense τ=0.5 ≠ internal RK-mid':'τ=0.5',235,191,10,'#667981');text(cc,'τ=1',550,191,10,'#667981');cc.restore();
    const rx=info.x-T.meta.roi.x,ry=info.y-T.meta.roi.y;
    if(info.family==='coarse'&&rx>=0&&ry>=0&&rx<16&&ry<12){const a=Math.floor(info.at),b=Math.min(40,a+1),u=info.at-a,idx=ry*16+rx,at=k=>terms[(a*13+k)*192+idx]*(1-u)+terms[(b*13+k)*192+idx]*u;
      $('fieldNote').textContent=`Current value contributions trước normalization: R=${sci(at(0))}, Q=${sci(at(1))}, transport=${sci(at(2))}; dz0/dτ=${sci(at(3))}. Đây không bằng dv/dτ; physical derivative cần Jacobian decode. Plot chia coefficients cho v(initial), units slope/Hessian vẫn khác nhau.`;
    }else $('fieldNote').textContent=info.family==='fine'?'Fine curve: z(τ)=z0+(τ−τ²)k1+τ²k2, sau đó decode jet. Continuous approximation bậc2 từ hai RHS đã record, không exact solution. Nếu đang chọn internal RK-mid, bảng là state interne còn curve vẫn là dense extension. Slopes theo half-grid units.':
      'Sáu jet coefficients có ở mọi cell; R/Q/transport diagnostics chỉ record trong ROI x65–80 / y45–56. Các coefficients chưa được enforce global integrability.';
  }
  function updateStage(){
    const s=stages[stage];$('stageGroup').textContent=s.group;$('stageTitle').textContent=`${String(stage+1).padStart(2,'0')} · ${s.name}`;$('stageDescription').textContent=s.description;
    $('beforeLabel').textContent=s.before;$('afterLabel').textContent=s.after;
    $('beforeShape').textContent=`${depths[s.before].h} × ${depths[s.before].w}`;$('afterShape').textContent=`${depths[s.after].h} × ${depths[s.after].w}`;
    $('coarseControls').hidden=stage!==0;$('fineControls').hidden=stage!==4;$('staticNotice').hidden=stage===0||stage===4;
    $('gateTab').hidden=stage!==8;if(stage!==8&&mapMode==='gate')setMap('delta');
    $('previous').disabled=stage===0;$('next').disabled=stage===8;
    document.querySelectorAll('[data-stage]').forEach(b=>{const active=Number(b.dataset.stage)===stage;b.classList.toggle('active',active);if(active)b.setAttribute('aria-current','step');else b.removeAttribute('aria-current');});
    revealHorizontal($('stages'),$('stages').children[stage]);
    document.querySelectorAll('[data-jump]').forEach(b=>b.classList.toggle('active',Number(b.dataset.jump)===stage));
  }
  function stopPlayback(){playing=false;finePlaying=false;pipelinePlaying=false;$('play').textContent='▶ Dynamics';$('playFine').textContent='▶ Fine NODE';$('playPipeline').textContent='▶ Toàn pipeline';}
  function setStage(value,keepPlaying=false){if(loading)return;stage=M.clamp(Math.floor(value),0,8);if(!keepPlaying)stopPlayback();frame=40;fineFrame=2;fineInternal=false;pipelineTween=1;pipelinePosition=stage+1;clip=stages[stage].clip;$('deltaScale').value=clip;mapKey='';updateStage();dirty=true;}
  function setPipeline(value,keepPlaying=false){if(loading)return;if(!keepPlaying)stopPlayback();const p=M.clamp(value,0,9),k=M.clamp(Math.ceil(p)-1,0,8),u=p-k;if(k!==stage)setStage(k,true);pipelinePosition=p;pipelineTween=u;fineInternal=false;if(k===0)frame=u*40;if(k===4)fineFrame=u*2;dirty=true;}
  function setFinePosition(value,internal=false){fineFrame=M.clamp(value,0,2);fineInternal=internal;pipelinePosition=4+fineFrame/2;pipelineTween=1;dirty=true;}
  function setMap(value){mapMode=value;$('deltaScaleLabel').hidden=value!=='delta';document.querySelectorAll('[data-map]').forEach(b=>b.classList.toggle('active',b.dataset.map===value));dirty=true;}
  function setCell(x,y){selected={x:M.clamp(Math.floor(x),0,FW-1),y:M.clamp(Math.floor(y),0,FH-1)};dirty=true;}
  function revealHorizontal(container,child){if(container.scrollWidth>container.clientWidth+1)container.scrollLeft=child.offsetLeft-container.offsetLeft-(container.clientWidth-child.clientWidth)/2;}
  function stateLabel(){return stage===0?`coarse τ=${fmt(frame/40)}`:stage===4?(fineInternal?'internal RK-mid ≠ dense half':`fine dense τ=${fmt(fineFrame/2)}`):pipelineTween<1?`display blend ${fmt(pipelineTween*100,1)}%`:'recorded endpoint';}
  function pixelSeries(){return [depths.D0,...stages.map(s=>depths[s.after])].map(b=>b.data[indexAt(b)]);}
  function drawCinema(){
    {
    const c=$('cinemaCanvas'),ctx=c.getContext('2d'),cr=crop(),split=wipe*c.width;
    if(compareMode==='rgb'&&rgb.complete&&rgb.naturalWidth){ctx.imageSmoothingEnabled=false;ctx.drawImage(rgb,cr.x,cr.y,cr.w,cr.h,0,0,c.width,c.height);}else drawMap(ctx,'before',0,0,c.width,c.height);
    ctx.save();ctx.beginPath();ctx.rect(split,0,c.width-split,c.height);ctx.clip();drawMap(ctx,'after',0,0,c.width,c.height);ctx.restore();
    // Draw selection on both sides, even RGB: same native-cell footprint.
    ctx.save();ctx.beginPath();ctx.rect(0,0,c.width,c.height);ctx.clip();const rr=selectedRegion(),x=(rr.sx-cr.x)*c.width/cr.w,y=(rr.sy-cr.y)*c.height/cr.h;
    ctx.strokeStyle='#ffdc68';ctx.lineWidth=3;ctx.strokeRect(x,y,rr.w*c.width/cr.w,rr.h*c.height/cr.h);ctx.restore();
    $('wipeHandle').style.left=`${wipe*100}%`;$('wipeHandle').setAttribute('aria-valuenow',fmt(wipe*100,1));$('wipeSlider').value=wipe*100;
    $('wipeLeft').textContent=compareMode==='rgb'?'RGB input':stages[stage].before+' · previous';$('wipeRight').textContent=stages[stage].after+' · current';
    }
    const s=stages[stage],values=pixelSeries(),current=afterData.data[indexAt(afterData)];
    $('cinemaTitle').textContent=`${s.before} → ${s.after} · ${stateLabel()}`;
    $('pipelineProgress').value=pipelinePosition;$('pipelinePercent').value=fmt(pipelinePosition/9*100,1)+'%';
    $('cinemaPixel').textContent=`Pixel (${selected.x}, ${selected.y}) · D0 ${fmt(values[0])} m → đang xem ${fmt(current)} m → Dfull ${fmt(values[9])} m`;
    $('cinemaSemantics').textContent=stage===0?'Coarse: 41 recorded dense observations; giữa mốc nội suy display.':stage===4?(fineInternal?'Internal numerical RK-mid; không nằm trên dense curve nói chung.':'Fine: RK2 continuous extension bậc 2, không thêm RHS/NFE.'):'CNN/readout: endpoints thật; chuyển tiếp chỉ là blend hiển thị.';
    document.querySelectorAll('[data-progress]').forEach(b=>{b.classList.toggle('active',Number(b.dataset.progress)===stage+1);b.classList.toggle('done',Number(b.dataset.progress)<pipelinePosition);});
    const tc=$('pixelTrajectory'),ctx=tc.getContext('2d');ctx.clearRect(0,0,tc.width,tc.height);
    const names=['D0','D4','D2 base','query','pre-fine','D2','D1 base','detail','PIR','Dfull'],lo=Math.min(...values,current),hi=Math.max(...values,current),span=Math.max(hi-lo,.02);
    const x=k=>55+k/9*(tc.width-110),y=v=>130-(v-lo)/span*90;
    line(ctx,55,130,tc.width-55,130,'#dce6e9');ctx.beginPath();values.forEach((v,k)=>k?ctx.lineTo(x(k),y(v)):ctx.moveTo(x(k),y(v)));ctx.strokeStyle='#99b7ba';ctx.lineWidth=2;ctx.stroke();
    values.forEach((v,k)=>{ctx.beginPath();ctx.arc(x(k),y(v),4,0,Math.PI*2);ctx.fillStyle='#087b76';ctx.fill();text(ctx,fmt(v),x(k)-23,y(v)-12,15);text(ctx,names[k],x(k)-24,160,14,'#667981');});
    const px=x(pipelinePosition),py=y(current);line(ctx,px,24,px,133,'#c66544',1);ctx.beginPath();ctx.arc(px,py,6,0,Math.PI*2);ctx.fillStyle='#c66544';ctx.fill();
    text(ctx,'m · điểm = recorded snapshots; đường nối = guide',55,18,12,'#667981');
  }
  function render(){if(loading)return;rebuildMaps();for(const [id,key]of [['before','before'],['after','after'],['impact',mapMode]]){const c=$(id);drawMap(c.getContext('2d'),key,0,0,c.width,c.height);}
    updateReadouts();drawCinema();if($('jetDetails').open)drawJet();$('timeLabel').value=fmt(frame/40);$('time').value=frame;
    $('fineTime').value=fineFrame;$('fineTimeLabel').value=fmt(fineFrame/2);
    const jt=stage===0?frame/40:stage===4?fineFrame/2:1;$('jetTime').value=jt;$('jetTimeLabel').value=fmt(jt);$('jetTime').disabled=stage!==0&&stage!==4;$('jetPlay').disabled=stage!==0&&stage!==4;$('jetPlay').textContent=playing||finePlaying?'Ⅱ Pause':'▶ Jet dynamics';
    document.querySelectorAll('[data-fine]').forEach(b=>b.classList.toggle('active',!fineInternal&&Number(b.dataset.fine)===fineFrame));$('inspectFineMid').classList.toggle('active',fineInternal);
    if(stage===0)$('afterLabel').textContent=`D4 geo · τ=${fmt(frame/40)}`;
    if(stage===4)$('afterLabel').textContent=`D2 · ${stateLabel()}`;dirty=false;
  }
  function metadata(){
    $('sampleId').textContent=T.meta.sample_id;$('sceneStatus').textContent=loading?'Đang đổi scene…':`${T.meta.solver.nfe}+2 NFE · replay parity ${T.meta.replay_max_difference_m} m`;
    $('rgb').src=scenes[scene].rgb;$('rgb').alt=T.meta.sample_id;
    document.querySelectorAll('.scene-button').forEach((b,k)=>{b.setAttribute('aria-selected',String(k===scene));b.tabIndex=k===scene?0:-1;b.disabled=loading;});
    revealHorizontal($('scenes'),$('scenes').children[scene]);
  }
  function loadRgb(path){return new Promise((resolve,reject)=>{rgb.onload=resolve;rgb.onerror=()=>reject(new Error('RGB missing'));const source=scenes.find(s=>s.rgb===path);rgb.src=source.rgb_data_url;});}
  async function selectScene(value){
    if(loading||value===scene&&window.PIPELINE_LAB.ready)return;
    if(!Number.isInteger(value)||value<0||value>=scenes.length)throw new Error('Invalid scene index');
    loading=true;stopPlayback();window.PIPELINE_LAB.ready=false;metadata();
    try{const data=await new Promise((resolve,reject)=>{const script=document.createElement('script');script.src=scenes[value].trace;
      script.onload=()=>{script.remove();resolve(window.JET_PIPELINE_PACK);};script.onerror=()=>{script.remove();reject(new Error('Missing trace file'));};document.body.appendChild(script);});
      if(data.meta.sample_id!==scenes[value].meta.sample_id)throw new Error('RGB/trace sample mismatch');
      install(await PipelineLoader.decode(data));scene=value;await loadRgb(scenes[value].rgb);window.PIPELINE_LAB.proof=T.meta;window.PIPELINE_LAB.ready=true;if($('jetDetails').open)ensureFullJets();
    }catch(e){loading=false;metadata();$('sceneStatus').textContent=e.message;throw e;}finally{loading=false;mapKey='';if(window.PIPELINE_LAB.ready){metadata();updateStage();dirty=true;}}
  }
  function download(blob,name){const a=document.createElement('a'),url=URL.createObjectURL(blob);a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),15000);}
  async function png(){
    rebuildMaps();const c=document.createElement('canvas');c.width=3840;c.height=2160;const ctx=c.getContext('2d');ctx.scale(3840/1440,2160/810);
    ctx.fillStyle='#f3f6f7';ctx.fillRect(0,0,1440,810);text(ctx,'ANCHORFLOW V11.3 / STAGE IMPACT',30,30,11,'#087b76',700);
    text(ctx,`${scene+1}. ${stages[stage].name} · ${stages[stage].before} → ${stages[stage].after}`,30,69,29,'#142d36',650);
    text(ctx,`${T.meta.sample_id} · best-policy epoch9 · CPU FP32 · ${stateLabel()} · zoom ${zoom}×`,30,94,12,'#667981');
    const panel=(x,y,w,h)=>{ctx.fillStyle='#fff';ctx.fillRect(x,y,w,h);};
    panel(30,113,680,251);text(ctx,'RGB · same scene',45,142,15,'#142d36',600);if(rgb.complete&&rgb.naturalWidth)ctx.drawImage(rgb,45,163,650,650*FH/FW);
    panel(730,113,680,251);text(ctx,'What this stage does',746,142,15,'#142d36',600);
    const words=stages[stage].description.split(' ');let row='',y=173;
    for(const word of words){const next=row+word+' ';if(ctx.measureText(next).width>626){text(ctx,row,746,y,14,'#667981');row='';y+=23;}row+=word+' ';}text(ctx,row,746,y,14,'#667981');
    const a=nativeStats.before,b=nativeStats.after;
    text(ctx,`Scene native RMSE: ${fmt(a.rmse,4)} → ${fmt(b.rmse,4)} m`,746,282,18,'#087b76',600);
    text(ctx,`Grid ${beforeData.h}×${beforeData.w} → ${afterData.h}×${afterData.w}; ${beforeData.w===afterData.w?'same GT support':'different native targets/support'}`,746,310,13,'#667981');
    text(ctx,'Not full400 benchmark or matched retraining ablation.',746,337,12,'#667981');
    for(const [x,key,label]of [[30,'before','Before'],[730,'after','After']]){panel(x,380,680,245);text(ctx,`${label}: ${stages[stage][key]}`,x+15,411,17,'#142d36',600);drawMap(ctx,key,x+15,434,650,650*FH/FW);}
    panel(30,643,680,135);text(ctx,`${mapMode==='delta'?'Signed Δdepth':mapMode==='error'?'Change in absolute GT error':'Sensor gate'}`,45,670,15,'#142d36',600);drawMap(ctx,mapMode,45,693,266,77,false);
    const bv=beforeData.data[indexAt(beforeData)],av=afterData.data[indexAt(afterData)];text(ctx,`Selected (${selected.x},${selected.y})`,330,709,13,'#667981');text(ctx,`${fmt(bv)} → ${fmt(av)} m`,330,738,20,'#087b76',600);text(ctx,`ΔD=${fmt(av-bv)} m; mean |ΔD|=${fmt(deltaStats.meanAbs,4)} m`,330,761,11,'#667981');
    text(ctx,'Full400 baseline: RMSE0.986517m / iRMSE3.169735km⁻¹',746,674,16,'#142d36',600);
    text(ctx,`Current scene solver: ${T.meta.solver.nfe} coarse +2 fine NFE; T=1`,746,703,13,'#667981');
    text(ctx,mapMode==='delta'?`Correction color saturation ±${clip}m; raw numbers not clipped.`:mapMode==='error'?'Green means error decreased; purple increased; gray has no GT.':'Gate is learned sensor reliability, not GT confidence.',746,731,12,'#667981');
    text(ctx,'Cross-grid deltas use nearest-expanded previous grid, display reference only.',746,757,11,'#667981');
    text(ctx,'4K figure does not change model pixel resolution. Fine RK-mid is an internal numerical stage, not an exact half-time solution.',30,799,10,'#667981');
    return new Promise(resolve=>c.toBlob(resolve,'image/png'));
  }
  scenes.forEach((s,k)=>{const b=document.createElement('button');b.className='scene-button';b.setAttribute('role','tab');b.setAttribute('aria-controls','rgbReference');b.setAttribute('aria-label',`Ảnh${k+1}, val index${s.meta.validation_index}`);
    b.innerHTML=`<img src="${s.rgb}" width="1216" height="352" alt="${s.meta.sample_id}"><div class="scene-caption"><b>${s.label}</b><span>val${s.meta.validation_index} · ${s.meta.sample_id.match(/drive_\d+/)[0]}</span></div>`;
    b.onclick=()=>selectScene(k).catch(e=>{$('sceneStatus').textContent=e.message;});b.onkeydown=e=>{if(e.key==='ArrowRight'||e.key==='ArrowLeft'){e.preventDefault();const next=(k+(e.key==='ArrowRight'?1:scenes.length-1))%scenes.length;$('scenes').children[next].focus();$('scenes').children[next].click();}};$('scenes').appendChild(b);});
  stages.forEach((s,k)=>{const b=document.createElement('button');b.dataset.stage=k;b.className='stage-button';b.innerHTML=`<span class="number">${String(k+1).padStart(2,'0')}</span><span><b>${s.name}</b><small>${s.sub}</small></span>`;b.onclick=()=>setStage(k);$('stages').appendChild(b);});
  document.querySelectorAll('[data-jump]').forEach(b=>b.onclick=()=>setStage(Number(b.dataset.jump)));
  document.querySelectorAll('[data-map]').forEach(b=>b.onclick=()=>setMap(b.dataset.map));
  stages.forEach((s,k)=>{const b=document.createElement('button');b.dataset.progress=k+1;b.textContent=`${k+1}. ${s.after}`;b.onclick=()=>setPipeline(k+1);$('pipelineChapters').appendChild(b);});
  function setCoarsePosition(value){frame=M.clamp(value,0,40);pipelinePosition=frame/40;pipelineTween=1;dirty=true;}
  document.querySelectorAll('[data-time]').forEach(b=>b.onclick=()=>{stopPlayback();setCoarsePosition(Number(b.dataset.time));});
  document.querySelectorAll('[data-fine]').forEach(b=>b.onclick=()=>{stopPlayback();setFinePosition(Number(b.dataset.fine));});
  $('previous').onclick=()=>setStage(stage-1);$('next').onclick=()=>setStage(stage+1);
  for(const id of ['zoom','cinemaZoom'])$(id).onchange=e=>{zoom=Number(e.target.value);$('zoom').value=zoom;$('cinemaZoom').value=zoom;dirty=true;};$('grid').onchange=()=>dirty=true;
  $('deltaScale').onchange=e=>{clip=Number(e.target.value);dirty=true;};$('time').oninput=e=>{stopPlayback();setCoarsePosition(Number(e.target.value));};
  $('fineTime').oninput=e=>{stopPlayback();setFinePosition(Number(e.target.value));};$('inspectFineMid').onclick=()=>{stopPlayback();setFinePosition(1,true);};
  $('pipelineProgress').oninput=e=>setPipeline(Number(e.target.value));$('resetPipeline').onclick=()=>setPipeline(0);
  $('cinemaOnly').onclick=()=>{const box=document.querySelector('.workspace');box.hidden=!box.hidden;$('cinemaOnly').setAttribute('aria-pressed',String(box.hidden));$('cinemaOnly').textContent=box.hidden?'Mở so sánh':'Ẩn chi tiết';};
  $('play').onclick=()=>{const start=!playing;stopPlayback();playing=start;if(start&&frame>=40)setCoarsePosition(0);lastTime=performance.now();$('play').textContent=start?'Ⅱ Pause':'▶ Dynamics';dirty=true;};
  $('playFine').onclick=()=>{const start=!finePlaying;stopPlayback();finePlaying=start;if(start&&(fineFrame>=2||fineInternal))setFinePosition(0);lastTime=performance.now();$('playFine').textContent=start?'Ⅱ Pause':'▶ Fine NODE';dirty=true;};
  $('playPipeline').onclick=()=>{const start=!pipelinePlaying;stopPlayback();if(start&&pipelinePosition>=9)setPipeline(0,true);pipelinePlaying=start;lastTime=performance.now();$('playPipeline').textContent=start?'Ⅱ Pause':'▶ Toàn pipeline';dirty=true;};
  for(const id of ['before','after','impact','cinemaCanvas']){const c=$(id);c.tabIndex=0;c.onclick=e=>{const r=c.getBoundingClientRect(),cr=crop();setCell(cr.x+(e.clientX-r.left)/r.width*cr.w,cr.y+(e.clientY-r.top)/r.height*cr.h);};
    c.onkeydown=e=>{const step=FW/afterData.w,offset={ArrowLeft:[-step,0],ArrowRight:[step,0],ArrowUp:[0,-step],ArrowDown:[0,step]}[e.key];if(offset){e.preventDefault();setCell(selected.x+offset[0],selected.y+offset[1]);}};}
  $('rgb').onclick=e=>{const r=e.target.getBoundingClientRect();setCell((e.clientX-r.left)/r.width*FW,(e.clientY-r.top)/r.height*FH);};
  $('jetDetails').ontoggle=()=>{dirty=true;if($('jetDetails').open)ensureFullJets();};
  $('loadFullJets').onclick=()=>{if(location.protocol==='file:')$('fullJetFile').click();else ensureFullJets();};$('loadFullJets').hidden=location.protocol!=='file:';
  $('fullJetFile').onchange=async e=>{const file=e.target.files[0];if(file)try{await acceptFullJets(await file.arrayBuffer());}catch(error){$('jetDataStatus').textContent=error.message;}e.target.value='';};
  $('jetRegion').onchange=e=>{jetRegion=Number(e.target.value);rangeKey='';dirty=true;};$('jetGain').onchange=e=>{jetGain=Number(e.target.value);dirty=true;};
  $('jetTime').oninput=e=>{stopPlayback();if(stage===0)setCoarsePosition(Number(e.target.value)*40);else if(stage===4)setFinePosition(Number(e.target.value)*2);};$('jetPlay').onclick=()=>{if(stage===0)$('play').click();else if(stage===4)$('playFine').click();};
  $('compareMode').onchange=e=>{compareMode=e.target.value;dirty=true;};$('wipeSlider').oninput=e=>{wipe=Number(e.target.value)/100;dirty=true;};
  let wiping=false;const handle=$('wipeHandle');handle.onpointerdown=e=>{e.preventDefault();e.stopPropagation();wiping=true;handle.setPointerCapture(e.pointerId);};handle.onpointermove=e=>{if(wiping){const r=$('cinemaCanvas').getBoundingClientRect();wipe=M.clamp((e.clientX-r.left)/r.width,0,1);dirty=true;}};handle.onpointerup=handle.onpointercancel=()=>wiping=false;handle.onclick=e=>e.stopPropagation();handle.onkeydown=e=>{if(['ArrowLeft','ArrowRight','Home','End'].includes(e.key)){e.preventDefault();wipe=e.key==='Home'?0:e.key==='End'?1:M.clamp(wipe+(e.key==='ArrowRight'?.02:-.02),0,1);dirty=true;}};
  $('focusRoi').onclick=()=>setCell(290,206);
  let drag=null;$('surface').onpointerdown=e=>{drag={x:e.clientX,y:e.clientY,yaw,pitch};e.target.setPointerCapture(e.pointerId);};$('surface').onpointermove=e=>{if(drag){yaw=drag.yaw+(e.clientX-drag.x)*.008;pitch=M.clamp(drag.pitch+(e.clientY-drag.y)*.006,.25,1.45);dirty=true;}};$('surface').onpointerup=()=>drag=null;$('surface').onpointercancel=()=>drag=null;$('resetView').onclick=()=>{yaw=-.62;pitch=.95;dirty=true;};
  $('exportPng').onclick=async()=>{download(await png(),`V11_3_scene${scene+1}_stage${stage+1}_${stages[stage].after}_4K.png`);$('exportStatus').textContent='PNG 3840×2160 đã xuất; model resolution không thay đổi.';};
  $('allStageMetrics').innerHTML='<thead><tr><th>Stage</th><th>Native grid</th><th>RMSE m</th><th>MAE m</th><th>Valid cells</th></tr></thead><tbody>'+Object.entries(baseline.native_stages).map(([k,v])=>`<tr><td>${k}</td><td>${depths[k]?depths[k].h+'×'+depths[k].w:k==='D16'?'22×76':k==='D8'?'44×152':k.startsWith('D4')?'88×304':'352×1216'}</td><td>${fmt(v.rmse_m,6)}</td><td>${fmt(v.mae_m,6)}</td><td>${v.pixels.toLocaleString('vi-VN')}</td></tr>`).join('')+'</tbody>';
  function loop(now){const dt=M.clamp((now-lastTime)/1000,0,.3);if(!loading){if(pipelinePlaying){setPipeline(pipelinePosition+dt/3,true);if(pipelinePosition>=9)stopPlayback();}else if(playing){setCoarsePosition(frame+dt*5);if(frame>=40)stopPlayback();}else if(finePlaying){setFinePosition(fineFrame+dt/3);if(fineFrame>=2)stopPlayback();}}lastTime=now;if(dirty&&(!(playing||finePlaying||pipelinePlaying)||now-lastRender>66)){render();lastRender=now;}requestAnimationFrame(loop);}
  window.addEventListener('resize',()=>{revealHorizontal($('scenes'),$('scenes').children[scene]);revealHorizontal($('stages'),$('stages').children[stage]);dirty=true;});
  window.PIPELINE_LAB={ready:false,selectScene,setStage,setMap,setCell,setPipeline,setFrame:f=>{stopPlayback();setCoarsePosition(f);},setFineFrame:f=>{stopPlayback();setFinePosition(f);},inspectFineMid:()=>{stopPlayback();setFinePosition(1,true);},render,png,pixelSeries,
    getState:()=>({scene,stage,frame,fineFrame,fineInternal,pipelinePosition,pipelineTween,playing,finePlaying,pipelinePlaying,zoom,mapMode,selected,loading,jetRegion,jetGain,compareMode,wipe,fullJets:!!fullCoarse}),
    proof:T.meta,rawJet:arrayJet,fineDenseJet,selectedRegion,ensureFullJets,acceptFullJets,selectedJetInfo,stages,
    jetVisualState:()=>({fixedRange,fixedRef,rangeKey,region:selectedRegion()}),
    values:()=>({before:beforeData.data[indexAt(beforeData)],after:afterData.data[indexAt(afterData)],deltaStats,nativeStats}),
    baseline,math:M};
  metadata();updateStage();render();requestAnimationFrame(loop);
  loadRgb(scenes[0].rgb).then(()=>{window.PIPELINE_LAB.ready=true;dirty=true;if($('jetDetails').open)ensureFullJets();}).catch(e=>{$('sceneStatus').textContent=e.message;});
})().catch(e=>{document.getElementById('sceneStatus').textContent='Viewer error: '+e.message;console.error(e);});
